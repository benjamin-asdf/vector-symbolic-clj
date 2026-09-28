(ns vsc.machine
  "Stage S2: a vector CEK machine. No host `veval`.

  The whole machine state is one vector, a pointer s in the working memory
  W with

    W: s ↦ ν(C⊗control + E⊗env + K⊗continuation + M⊗mode)

  mode ∈ {EVAL, RET, APPLY, HALT, ERROR}. In EVAL, control is an expression;
  in RET, a value being returned to the top frame; in APPLY, a function whose
  argument list sits in the E field. The continuation is a linked list of
  frames in W, each frame a record ν(T⊗type + A⊗a + B⊗b + E⊗env + N⊗next).

  Transition rules live in a rule memory R, keyed by the shape of the state,
  and are chosen by similarity (nearest key). Each rule's value is a pointer
  to a microprogram: a list of instruction vectors in a code memory, written
  in a fixed register ISA of nine instructions. The key itself is computed by
  a microprogram (`fetch`), from VSA operations only.

  The only host control code is the loop at the bottom of this file:
  fetch, select a rule by cleanup, run its microprogram, repeat until the
  mode is HALT. It knows nothing about Lisp; every Lisp-specific decision is
  in the rule table and the microprograms, which are vectors."
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
   [libpython-clj2.python :as py]
   [vsc.core :as vsc]
   [vsc.hdc :as h]))

;; ---------------------------------------------------------------------------
;; the core's data layer (reused, not rewritten)

(defn- core [] @@#'vsc.core/machine)
(defn- cm [k] (get (core) k))
(defn- sp [] (cm :space))
(defn- prim-atom [name] (get @(cm :atoms) [:prim name]))

;; ---------------------------------------------------------------------------
;; the vocabulary: every name below is one vector in the operand memory

(def registers '[S C E K M T A B N X Y Z KEY])
(def opcodes '[FIELD CLEAN BIND UNBIND REC MOV JMPEQ PRIM DEF])
(def clean-spaces '[VAL CONST KIND CLASS F])     ;; where a probe is cleaned up
(def rec-spaces '[W LIST FN NONE])               ;; where a record is stored
(def modes '[EVAL RET APPLY HALT ERROR])
(def frame-types '[halt-fr if-fr do-fr def-fr let-fr cond-fr and-fr or-fr args-fr])
(def features '[FM FK FH FI FF])                 ;; roles of the rule key
(def kind-names
  '{:sym k-sym :kw k-kw :str k-str :num k-num :nil k-nil :true k-true
    :false k-false :prim k-prim :list k-list :vec k-vec :map k-map :set k-set
    :fn k-fn :box k-box})
(def empty-kind-names '{:list k-empty-list :vec k-empty-vec :map k-empty-map :set k-empty-set})
(def n-fields 13)                                ;; OP + 12 operand positions

;; ---------------------------------------------------------------------------
;; the program: microcode routines and the rule table, as Clojure data.
;; Assembled into vectors at init; after that, only the vectors are used.
;;
;; Instruction set (d: destination register, x/y: register or constant,
;; r: role (immediate), s: space, l: code label):
;;   FIELD d x r s   d ← clean_s(r ⊘ deref(x))       read a record field
;;   CLEAN d x s     d ← clean_s(x)                   cleanup / classify
;;   BIND d x y      d ← x ⊗ y
;;   UNBIND d x y    d ← x ⊘ y
;;   REC d s r x …   d ← store_s(ν(Σ r⊗x))            build a record
;;   MOV d x         d ← x
;;   JMPEQ x y l     if sim(x, y) > θ, jump to l
;;   PRIM d f x      d ← host primitive f applied to argument list x
;;   DEF x y         global memory F ↞ (x ↦ y)
;; Assembler macros: (JMP l) = (JMPEQ NIL NIL l),
;;                   (PACK) = (REC S W C C E E K K M M), the next state.

(def fetch-program
  "Unpack the state and compute its shape: the rule key
  ν(FM⊗mode + FK⊗kind(C) + FH⊗kind(C)⊗class(car C) + FI⊗kind(C)⊗class(C) + FF⊗type(top frame))."
  '[:fetch
    [FIELD M S M CONST] [FIELD C S C VAL] [FIELD E S E VAL] [FIELD K S K VAL]
    [FIELD T K T CONST]
    [CLEAN X C KIND]
    [FIELD Y C L VAL] [CLEAN Y Y CLASS] [BIND Y X Y]
    [CLEAN Z C CLASS] [BIND Z X Z]
    [REC KEY NONE FM M FK X FH Y FI Z FF T]])

(def boot-program
  '[:boot [REC K W T halt-fr] [MOV E EMPTY] [MOV M EVAL] [PACK]])

(def routines
  '[[:ret [MOV M RET] [PACK]]
    [:eval [MOV M EVAL] [PACK]]
    [:error [MOV M ERROR] [PACK]]
    [:ret-nil [MOV C NIL] [JMP :ret]]
    ;; evaluate the body list X in env E; the last form is in tail position
    [:body [CLEAN A X KIND] [JMPEQ A k-empty-list :ret-nil]
     [FIELD C X L VAL] [FIELD B X R VAL]
     [CLEAN A B KIND] [JMPEQ A k-empty-list :eval]
     [REC K W T do-fr A B E E N K] [JMP :eval]]
    ;; let: remaining bindings A, body B, env E
    [:let-next [CLEAN X A KIND] [JMPEQ X k-empty-vec :let-body] [JMPEQ X k-empty-list :let-body]
     [FIELD Y A R VAL] [FIELD C Y L VAL]
     [REC K W T let-fr A A B B E E N K] [JMP :eval]
     :let-body [MOV X B] [JMP :body]]
    ;; and/or: remaining forms X (non-empty), frame type T
    [:seq-next [FIELD C X L VAL] [FIELD A X R VAL]
     [CLEAN Y A KIND] [JMPEQ Y k-empty-list :eval]
     [REC K W T T A A E E N K] [JMP :eval]]
    ;; cond: remaining clauses X
    [:cond-next [CLEAN Y X KIND] [JMPEQ Y k-empty-list :ret-nil]
     [FIELD C X L VAL] [FIELD A X R VAL]
     [REC K W T cond-fr A A E E N K] [JMP :eval]]])

(def rules
  "Each rule: :key (the state features it matches) and :code. A rule matches
  by similarity, so a key naming fewer features acts as a default for more
  specific ones."
  '[;; ---- EVAL: control is an expression
    {:name self-evaluating :key {:mode EVAL} :code [[JMP :ret]]}
    {:name unknown-vector :key {:mode EVAL :kind k-unknown} :code [[JMP :error]]}
    {:name symbol :key {:mode EVAL :kind k-sym}
     :code [[MOV X E]
            :lk-loop [CLEAN A X KIND] [JMPEQ A k-empty-list :lk-global]
            [FIELD Y X L VAL] [FIELD Z Y L VAL] [JMPEQ Z C :lk-found]
            [FIELD X X R VAL] [JMP :lk-loop]
            :lk-found [FIELD C Y R VAL] [JMP :ret]
            :lk-global [BIND Y L C] [CLEAN Y Y F] [UNBIND Z L Y] [JMPEQ Z C :lk-ok]
            [JMP :error]
            :lk-ok [UNBIND Y R Y] [CLEAN C Y VAL] [JMP :ret]]}
    {:name call :key {:mode EVAL :kind k-list}
     :code [[FIELD B C R VAL] [REC K W T args-fr A EMPTY B B E E N K]
            [FIELD C C L VAL] [JMP :eval]]}
    {:name vector-literal :key {:mode EVAL :kind k-vec}
     :code [[REC K W T args-fr A EMPTY B C E E N K] [MOV C %vector] [JMP :ret]]}
    {:name map-literal :key {:mode EVAL :kind k-map}
     :code [[REC X W L C R EMPTY] [PRIM B %seq X] [REC B W L EMPTY-MAP R B]
            [REC K W T args-fr A EMPTY B B E E N K] [MOV C %conj] [JMP :ret]]}
    {:name set-literal :key {:mode EVAL :kind k-set}
     :code [[REC X W L C R EMPTY] [PRIM B %seq X] [REC B W L EMPTY-SET R B]
            [REC K W T args-fr A EMPTY B B E E N K] [MOV C %conj] [JMP :ret]]}
    {:name quote :key {:mode EVAL :kind k-list :head quote}
     :code [[FIELD C C R VAL] [FIELD C C L VAL] [JMP :ret]]}
    {:name if :key {:mode EVAL :kind k-list :head if}
     :code [[FIELD X C R VAL] [FIELD C X L VAL] [FIELD A X R VAL]
            [REC K W T if-fr A A E E N K] [JMP :eval]]}
    {:name do :key {:mode EVAL :kind k-list :head do}
     :code [[FIELD X C R VAL] [JMP :body]]}
    {:name def :key {:mode EVAL :kind k-list :head def}
     :code [[FIELD X C R VAL] [FIELD A X L VAL] [FIELD X X R VAL] [FIELD C X L VAL]
            [REC K W T def-fr A A N K] [JMP :eval]]}
    {:name defn :key {:mode EVAL :kind k-list :head defn}
     :code [[FIELD X C R VAL] [FIELD A X L VAL]
            [REC Y LIST L FN-SYM R X] [REC Y FN L Y R EMPTY]
            [DEF A Y] [MOV C A] [JMP :ret]]}
    {:name fn :key {:mode EVAL :kind k-list :head fn}
     :code [[REC C FN L C R E] [JMP :ret]]}
    {:name let :key {:mode EVAL :kind k-list :head let}
     :code [[FIELD X C R VAL] [FIELD A X L VAL] [FIELD B X R VAL] [JMP :let-next]]}
    {:name cond :key {:mode EVAL :kind k-list :head cond}
     :code [[FIELD X C R VAL] [JMP :cond-next]]}
    {:name and :key {:mode EVAL :kind k-list :head and}
     :code [[FIELD X C R VAL] [MOV T and-fr]
            [CLEAN Y X KIND] [JMPEQ Y k-empty-list :and-empty] [JMP :seq-next]
            :and-empty [MOV C TRUE] [JMP :ret]]}
    {:name or :key {:mode EVAL :kind k-list :head or}
     :code [[FIELD X C R VAL] [MOV T or-fr]
            [CLEAN Y X KIND] [JMPEQ Y k-empty-list :ret-nil] [JMP :seq-next]]}

    ;; ---- RET: control is a value, returned to the top frame
    {:name bad-frame :key {:mode RET} :code [[JMP :error]]}
    {:name halt :key {:mode RET :frame halt-fr} :code [[MOV M HALT] [PACK]]}
    {:name if-branch :key {:mode RET :frame if-fr}
     :code [[FIELD A K A VAL] [FIELD E K E VAL] [FIELD K K N VAL]
            [JMPEQ C NIL :if-else] [JMPEQ C FALSE :if-else]
            [FIELD C A L VAL] [JMP :eval]
            :if-else [FIELD A A R VAL] [CLEAN X A KIND] [JMPEQ X k-empty-list :ret-nil]
            [FIELD C A L VAL] [JMP :eval]]}
    {:name do-next :key {:mode RET :frame do-fr}
     :code [[FIELD X K A VAL] [FIELD E K E VAL] [FIELD K K N VAL] [JMP :body]]}
    {:name def-store :key {:mode RET :frame def-fr}
     :code [[FIELD A K A VAL] [FIELD K K N VAL] [DEF A C] [MOV C A] [JMP :ret]]}
    {:name let-bind :key {:mode RET :frame let-fr}
     :code [[FIELD A K A VAL] [FIELD B K B VAL] [FIELD E K E VAL] [FIELD K K N VAL]
            [FIELD X A L VAL] [REC X LIST L X R C] [REC E LIST L X R E]
            [FIELD A A R VAL] [FIELD A A R VAL] [JMP :let-next]]}
    {:name cond-test :key {:mode RET :frame cond-fr}
     :code [[FIELD A K A VAL] [FIELD E K E VAL] [FIELD K K N VAL]
            [JMPEQ C NIL :cond-no] [JMPEQ C FALSE :cond-no]
            [FIELD C A L VAL] [JMP :eval]
            :cond-no [FIELD X A R VAL] [JMP :cond-next]]}
    {:name and-next :key {:mode RET :frame and-fr}
     :code [[FIELD X K A VAL] [FIELD E K E VAL] [FIELD K K N VAL]
            [JMPEQ C NIL :ret] [JMPEQ C FALSE :ret] [MOV T and-fr] [JMP :seq-next]]}
    {:name or-next :key {:mode RET :frame or-fr}
     :code [[FIELD X K A VAL] [FIELD E K E VAL] [FIELD K K N VAL]
            [JMPEQ C NIL :or-go] [JMPEQ C FALSE :or-go] [JMP :ret]
            :or-go [MOV T or-fr] [JMP :seq-next]]}
    {:name arg :key {:mode RET :frame args-fr}
     :code [[FIELD A K A VAL] [REC A W L C R A]
            [FIELD B K B VAL] [FIELD E K E VAL] [FIELD N K N VAL]
            [CLEAN X B KIND] [JMPEQ X k-empty-list :args-done] [JMPEQ X k-empty-vec :args-done]
            [FIELD C B L VAL] [FIELD B B R VAL]
            [REC K W T args-fr A A B B E E N N] [JMP :eval]
            ;; all evaluated: reverse the list, then apply its head to its tail
            :args-done [MOV Y EMPTY]
            :args-rev [CLEAN X A KIND] [JMPEQ X k-empty-list :args-go]
            [FIELD Z A L VAL] [REC Y W L Z R Y] [FIELD A A R VAL] [JMP :args-rev]
            :args-go [FIELD C Y L VAL] [FIELD E Y R VAL] [MOV K N] [MOV M APPLY] [PACK]]}

    ;; ---- APPLY: control is a function, E its argument list
    {:name not-a-function :key {:mode APPLY} :code [[JMP :error]]}
    {:name primitive :key {:mode APPLY :kind k-prim} :code [[PRIM C C E] [JMP :ret]]}
    {:name apply :key {:mode APPLY :kind k-prim :ident %apply}
     :code [[FIELD C E L VAL] [FIELD X E R VAL] [PRIM E %spread X] [MOV M APPLY] [PACK]]}
    {:name eval :key {:mode APPLY :kind k-prim :ident %eval}
     :code [[FIELD C E L VAL] [MOV E EMPTY] [JMP :eval]]}
    {:name keyword :key {:mode APPLY :kind k-kw}
     :code [[FIELD X E R VAL] [REC X W L C R X] [FIELD Y E L VAL] [REC X W L Y R X]
            [PRIM C %get X] [JMP :ret]]}
    {:name map :key {:mode APPLY :kind k-map}
     :code [[REC X W L C R E] [PRIM C %get X] [JMP :ret]]}
    {:name set :key {:mode APPLY :kind k-set}
     :code [[REC X W L C R E] [PRIM C %get X] [JMP :ret]]}
    {:name closure :key {:mode APPLY :kind k-fn}
     :code [[FIELD X C L VAL] [FIELD Y C R VAL]        ;; form (fn …), captured env
            [FIELD X X R VAL] [FIELD Z X L VAL]        ;; (name? params body…)
            [CLEAN A Z KIND] [JMPEQ A k-sym :fn-named]
            [FIELD X X R VAL] [JMP :fn-bind]
            ;; (fn name [..] ..) sees itself under its name
            :fn-named [REC A LIST L Z R C] [REC Y LIST L A R Y]
            [FIELD X X R VAL] [FIELD Z X L VAL] [FIELD X X R VAL]
            ;; bind params Z to args E onto env Y
            :fn-bind [CLEAN A Z KIND] [JMPEQ A k-empty-vec :fn-body] [JMPEQ A k-empty-list :fn-body]
            [FIELD A Z L VAL] [JMPEQ A AMP :fn-rest]
            [FIELD B E L VAL] [REC B LIST L A R B] [REC Y LIST L B R Y]
            [FIELD Z Z R VAL] [FIELD E E R VAL] [JMP :fn-bind]
            :fn-rest [FIELD A Z R VAL] [FIELD A A L VAL] [PRIM B %list E]
            [REC B LIST L A R B] [REC Y LIST L B R Y]
            :fn-body [MOV E Y] [JMP :body]]}])

;; ---------------------------------------------------------------------------
;; the machine's own memories (resources/vsc/vsc_machine.py)

(defonce ^:private vm (atom nil))
(defn- v [k] (get @vm k))
(defn- W [] (v :work))

(def ^:const theta-jump 0.6)   ;; JMPEQ: exact vectors give ~1 or ~0; a global slot ~0.71
(def ^:const theta-rule 0.4)   ;; weakest correct match (a one-feature default) is 1/√5 ≈ 0.45

(defn- py-module []
  @@#'vsc.hdc/module
  ;; reload, so that a rebuilt machine picks up an edited vsc_machine.py
  (py/call-attr (py/import-module "importlib") "reload" (py/import-module "vsc_machine")))

(defn- opm-add!
  "Add `vec` to the operand memory under `name`; returns its index."
  [name vec]
  (let [i (py/call-attr (py/get-attr (W) "opm") "add" vec nil 0)]
    (swap! vm (fn [s] (-> s (update :names assoc name i) (update :consts assoc i vec)
                          (update :name-of assoc i name))))
    i))

(defn- named [x] (get (v :consts) (get (v :names) x)))

(defn- constants
  "The non-register constants: data atoms of the core, plus fresh atoms."
  []
  (merge {'L ((cm :roles) :L) 'R ((cm :roles) :R)
          'NIL (cm :nil) 'TRUE (cm :true) 'FALSE (cm :false)
          'EMPTY ((cm :tags) :list) 'EMPTY-MAP ((cm :tags) :map) 'EMPTY-SET ((cm :tags) :set)
          'FN-SYM (cm :fn-sym) 'AMP (cm :amp)
          '%vector (prim-atom 'vector) '%conj (prim-atom 'conj) '%seq (prim-atom 'seq)
          '%list (prim-atom 'list) '%get (prim-atom 'get)
          '%apply (prim-atom 'apply) '%eval (prim-atom 'eval)}
         (into {} (for [n (concat opcodes clean-spaces rec-spaces modes frame-types features
                                  (vals kind-names) (vals empty-kind-names)
                                  '[k-unknown k-wcell none %spread])]
                    [n (h/unitary (sp))]))))

;; ---------------------------------------------------------------------------
;; assembler: Clojure data → instruction vectors in the code memory

(defn- expand [[op & xs :as instr]]
  (case op
    JMP ['JMPEQ 'NIL 'NIL (first xs)]
    PACK '[REC S W C C E E K K M M]
    instr))

(defn- positions
  "[[labels instr] …] of a flat program vector."
  [prog]
  (:out (reduce (fn [{:keys [pending out]} x]
                  (if (keyword? x)
                    {:pending (conj pending x) :out out}
                    {:pending [] :out (conj out [pending (expand x)])}))
                {:pending [] :out []} prog)))

(defn- assemble!
  "Store `prog` as a list of instruction cells; returns its entry pointer.
  Labels must already be in the operand memory."
  [prog]
  (let [w (W)
        cells (positions prog)]
    (reduce (fn [nxt [labels [op & args]]]
              (let [fields (concat [(named op)]
                                   (map named args)
                                   (repeat (- n-fields 1 (count args)) nil))
                    instr (py/call-attr w "asm_instr" (py/->py-list fields))
                    key (if (seq labels) (named (first labels)) (h/unitary (sp)))]
                (py/call-attr w "asm_cell" key instr nxt)))
            (v :end) (reverse cells))))

(defn- declare-labels!
  "Give every labelled position a code pointer, known to the operand memory."
  [prog]
  (doseq [[labels _] (positions prog)
          :when (seq labels)]
    (let [key (h/unitary (sp))]
      (doseq [l labels] (opm-add! l key)))))

(defn- feature-key
  "The rule key for a feature map, built like the fetch program builds it."
  [{:keys [mode kind head ident frame]}]
  (let [w (W)
        b #(h/bind (sp) %1 %2)
        cls #(if (symbol? %) (or (named %) (vsc/encode %)) %)
        parts (cond-> [(named 'FM) (named mode)]
                kind (into [(named 'FK) (named kind)])
                head (into [(named 'FH) (b (named kind) (cls head))])
                ident (into [(named 'FI) (b (named kind) (cls ident))])
                frame (into [(named 'FF) (named frame)]))]
    (apply py/call-attr w "record" parts)))

(defn- class-members
  "What the CLASS space recognises: the special forms, and the two
  primitives the machine must intercept because they re-enter evaluation."
  []
  (concat (map vsc/encode vsc/special-forms) [(prim-atom 'apply) (prim-atom 'eval)]))

(defn- spread
  "(apply f a b [c d]) spreads its last argument: (a b c d)."
  [args]
  (vsc/mk-list (concat (butlast args) (#'vsc.core/as-cells (last args)))))

(defn- build!
  "Create the machine's memories and assemble fetch, boot, routines and rules."
  []
  (let [s (sp)
        w (py/call-attr (py-module) "Work" (cm :C) (cm :F))]
    (reset! vm {:work w :names {} :consts {} :name-of {}})
    (doseq [r registers] (opm-add! r (h/unitary s)))
    (doseq [[n x] (sort-by (comp str key) (constants))] (opm-add! n x))
    (let [roles (fn [ns] (py/->py-list (map named ns)))
          fields (vec (repeatedly n-fields #(h/unitary s)))]
      (py/call-attr w "set_roles" (named 'L) (named 'R) (h/unitary s) (h/unitary s)
                    (py/->py-list fields) (roles '[C E K M T A B N L R])))
    (let [label->kind @#'vsc.core/label->kind
          empties (for [[k tag] (cm :tags)] [(first (h/nearest (cm :M) tag)) (named (empty-kind-names k))])]
      (py/call-attr w "set_kinds"
                    (py/->py-list (keys label->kind))
                    (py/->py-list (map #(named (kind-names %)) (vals label->kind)))
                    (py/->py-list (map first empties)) (py/->py-list (map second empties))
                    (named 'k-unknown) (named 'k-wcell) (named 'none))
      (py/call-attr w "set_list_labels" (py/->py-list [10 11])))
    (doseq [x (class-members)] (py/call-attr (py/get-attr w "cls") "add" x nil 0))
    (swap! vm assoc :end (py/call-attr w "asm_end")
           :internal [[(named '%spread) spread]])
    (let [progs (concat [fetch-program boot-program] routines (map :code rules))]
      (doseq [p progs] (declare-labels! p))
      (swap! vm assoc :fetch (assemble! fetch-program) :boot (assemble! boot-program))
      (doseq [p routines] (assemble! p))
      (doseq [[i {:keys [key code]}] (map-indexed vector rules)]
        (py/call-attr (py/get-attr w "R") "add" (feature-key key) (assemble! code) i)))
    (swap! vm assoc :rule-names (mapv :name rules)
           :reg (zipmap registers (range))
           :space (into {} (map (fn [n i] [(get (v :names) n) i]) clean-spaces (range)))
           :rec-space (into {} (map (fn [n] [(get (v :names) n) (keyword n)]) rec-spaces))
           :opcode (into {} (map (fn [n] [(get (v :names) n) n]) opcodes)))
    :built))

;; ---------------------------------------------------------------------------
;; the datapath: what each instruction does to vectors (one substrate call each)

(defn- field [x r s] (py/call-attr (W) "field" x r s))
(defn- clean [x s] (py/call-attr (W) "clean" x s))
(defn- sim ^double [a b] (h/sim (sp) a b))

(defn- store
  "Build ν(Σ rᵢ⊗xᵢ) from `pairs` [r₁ x₁ r₂ x₂ …] and store it in `space`."
  [space pairs]
  (case space
    :W (apply py/call-attr (W) "alloc" pairs)
    :NONE (apply py/call-attr (W) "record" pairs)
    :LIST (apply py/call-attr (W) "intern" 10 pairs)
    :FN (apply py/call-attr (W) "intern" 14 pairs)))

(defn- items [x]
  (let [xs (py/call-attr (W) "items_of" x)]
    (mapv #(py/get-item xs %) (range (py/call-attr xs "__len__")))))

(defn- prim [f args]
  (let [[i _ s] (h/recall (cm :P) f)
        g (if (> s vsc/theta-atom)
            (:f (nth (cm :prims) i))
            (second (apply max-key #(sim f (first %)) (v :internal))))]
    (g (items args))))

(defn- decode
  "[field indices, next code pointer] of the instruction at `pc`, fetched
  from the code memory and decoded by cleanup. With the instruction cache on,
  each code pointer is decoded once and its fields are then reused (the code
  memory is immutable after build!, and every pc the machine meets is an
  object that decode or select handed out, so identity is a valid key)."
  [pc]
  (let [ic ^java.util.Map (v :icache)]
    (or (and ic (.get ic pc))
        (let [r (py/call-attr (W) "decode" pc)
              d [(py/->jvm (py/get-item r 0)) (py/get-item r 1)]]
          (when ic (.put ic pc d))
          d))))

(defn icache!
  "Turn the decoded-instruction cache on or off."
  [on?]
  (swap! vm assoc :icache (when on? (java.util.IdentityHashMap.))))

;; ---------------------------------------------------------------------------
;; the host loop: the only host control code (ISA interpreter + driver)

(def ^:private n-reg (count registers))
(def ^:dynamic *trace* nil)      ;; (fn [step rule-index regs]) called after fetch
(defonce ^:private counters (atom {:steps 0 :instrs 0}))

(defn- exec!
  "Execute one decoded instruction on register file `regs`; returns a jump
  target or nil."
  [^objects regs ops]
  (let [a #(nth ops %)
        val #(let [i (a %)] (if (< i n-reg) (aget regs i) (get (v :consts) i)))
        imm #(get (v :consts) (a %))
        put! #(do (aset regs (a 1) %) nil)]
    (case ((v :opcode) (a 0))
      FIELD (put! (field (val 2) (imm 3) ((v :space) (a 4))))
      CLEAN (put! (clean (val 2) ((v :space) (a 3))))
      BIND (put! (py/call-attr (W) "bind" (val 2) (val 3)))
      UNBIND (put! (py/call-attr (W) "unbind" (val 2) (val 3)))
      REC (put! (store ((v :rec-space) (a 2))
                       (mapcat (fn [k] [(imm k) (val (inc k))])
                               (take-while #(>= (a %) 0) (range 3 n-fields 2)))))
      MOV (put! (val 2))
      JMPEQ (when (> (sim (val 1) (val 2)) theta-jump) (imm 3))
      PRIM (put! (prim (val 2) (val 3)))
      DEF (do (vsc/define! (val 1) (val 2)) nil))))

(defn- run-micro!
  "Run the microprogram at code pointer `pc` to its end."
  [regs pc]
  (loop [pc pc]
    (when pc
      (let [[ops nxt] (decode pc)]
        (swap! counters update :instrs inc)
        (recur (or (exec! regs ops) nxt))))))

(defn- reg [^objects regs r] (aget regs ((v :reg) r)))

(defn- run-state!
  "Drive the machine from the state in register S until it halts:
  fetch, select a rule by nearest key, run it. Returns the control register."
  [^objects regs]
  (loop [step 0]
    (run-micro! regs (v :fetch))
    (let [r (py/call-attr (W) "select" (reg regs 'KEY))
          i (py/get-item r 0)]
      (when (< (py/get-item r 1) theta-rule)
        (throw (ex-info "no rule matches the machine state" {:sim (py/get-item r 1)})))
      (when *trace* (*trace* step i regs))
      (run-micro! regs (or (get-in @vm [:entries i])
                           (let [pc (py/get-item r 2)] (swap! vm assoc-in [:entries i] pc) pc)))
      (swap! counters update :steps inc)
      (py/call-attr (W) "gc" (reg regs 'S) false)
      (cond
        (> (sim (reg regs 'M) (named 'HALT)) theta-jump) (reg regs 'C)
        (> (sim (reg regs 'M) (named 'ERROR)) theta-jump)
        (throw (ex-info (str "machine error at " (pr-str (vsc/decode (reg regs 'C)))) {}))
        :else (recur (inc step))))))

;; ---------------------------------------------------------------------------
;; entry points

(defn eval-form
  "Encode host `form`, run it on the machine, return the result vector."
  [form]
  (let [regs (object-array n-reg)]
    (aset regs ((v :reg) 'C) (vsc/encode form))
    (run-micro! regs (v :boot))
    (try (run-state! regs)
         (finally (py/call-attr (W) "gc" nil true)))))

(defn run
  "Evaluate a host form on the vector machine and decode the result."
  [form]
  (vsc/decode (eval-form form)))

(defn run-string
  "Evaluate every form in `src` on the machine; the decoded value of the last."
  [src]
  (reduce (fn [_ form] (run form)) nil (vsc/read-forms src)))

(defn init!
  "A fresh core (space, memories, primitives), the machine's memories and
  microcode, then the prelude, evaluated by the machine."
  ([] (init! {}))
  ([opts]
   (vsc/init! (assoc opts :prelude? false))
   (build!)
   (icache! (:icache? opts true))
   (when (:prelude? opts true)
     (run-string (slurp (io/resource "vsc/prelude.clj"))))
   :ready))
