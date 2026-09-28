(ns vsc.worlds
  "W1: superposition (\"many worlds\") programming for the vector-symbolic
  dialect. Design: docs/W0-superposition-design.md, section 6; results and
  limits: docs/results-W.md.

  A value may be a weighted superposition of ordinary values (worlds),
  x = Σ wᵢ vᵢ with wᵢ ≥ 0 and Σ wᵢ = 1. The weights are the linear
  coefficients themselves (not squared amplitudes), so they read directly as
  probabilities. Everything runs on the :proj memory (hdc.py, Proj), which is
  linear on the worlds it detects and snaps everything else away; on a
  single world it is the codebook.

  Semantics, in one line each:
    - destructors lift linearly: first, rest, env lookup (so let/fn
      parameters never snap), inc/dec/+/- (binding distributes over sums),
      get with a superposed keyword key; every cell read multiplies by the
      trace's stored field gain, so worlds keep their weights at any depth
    - everything else that takes a superposed argument enumerates its worlds:
      f(x) = Σ wᵢ f(xᵢ), and several superposed arguments give the product
      of their distributions (run-time choice, below)
    - constructors enumerate: (cons (amb 1 2) ()) = (amb (1) (2)), so that
      worlds are proper values and equal worlds merge (their weights add)
    - if/cond/and/or on a superposed test evaluate both branches: weights
      π : 1−π, π read from the test's coefficients; each branch gets only
      its own worlds of the test's superposed variable (the split), which is
      what makes recursion whose depth differs between worlds terminate; a
      branch no world needs (π < ε) is not evaluated
    - choice is run-time: every occurrence of a superposed variable chooses
      independently, (let [x (amb 1 2)] (+ x x)) = {2 ¼, 3 ½, 4 ¼}.
      for-worlds is call-time choice by explicit enumeration:
      (for-worlds [x (amb 1 2)] (+ x x)) = {2 ½, 4 ½}

  Dialect primitives added by init!: amb superpose worlds weight collapse
  sample assert support without, and the special form for-worlds. The
  dialect has no floats, so weights inside the dialect are integers: relative
  weights for superpose, per mille for worlds and weight. The host API
  (worlds, weight, run) uses doubles."
  (:require
   [vsc.core :as core]
   [vsc.hdc :as h]))

;; ---------------------------------------------------------------------------
;; access to the core machine

(defn- m [k] (get @@#'core/machine k))
(defn- sp [] (m :space))
(defn- cfg [k] (get (m :worlds) k))

(def ^:private NUM 4)
(defn- label->kind [label] (get @#'core/label->kind label))

(defn- nil-vec [] (m :nil))
(defn- false-vec [] (m :false))
(defn- bool [x] (if x (m :true) (m :false)))
(defn- num-vec [n] (h/num-vec (m :nums) n))
(defn- truthy? [v] (#'core/truthy? v))
(defn- cells [c] (#'core/cells c))
(defn- env-extend [env sym v] (#'core/env-extend env sym v))

;; ---------------------------------------------------------------------------
;; capacity

(defn capacity
  "How many worlds `worlds` reads back reliably at the machine's dimension:
  {:worlds n :ints n}. Pointers and atoms: about D/100 (weights within TV
  0.1 after a memory step, W0 section 1; docs/results-W.md). Integers:
  about D/400, because residue integers are correlated with each other
  (W0 section 2)."
  []
  (let [d (m :dim)]
    {:worlds (max 2 (quot d 100)) :ints (max 2 (quot d 400))}))

(defn- over-capacity [what n cap]
  (throw (ex-info (str "superposition capacity exceeded: " n " " what
                       " worlds, capacity " cap " at D=" (m :dim)
                       "; init! with a larger :dim")
                  {:worlds n :capacity cap :what what})))

;; ---------------------------------------------------------------------------
;; readout

(defn- world-vec [[label i _]]
  (if (= label NUM) (num-vec i) (h/mem-get (m :M) i)))

(defn- single?
  "Is v one clean world (a stored row or an exact integer)? The bar is
  high: a 1% world next to a 99% one leaves a similarity of 0.99995."
  [v]
  (let [[_ _ s] (h/recognize (m :C) v)]
    (> s 0.999999)))

(defn readout
  "The raw readout of v: [[label index weight] ...], heaviest first, index
  = n for integers. Throws beyond capacity or when the worlds found do not
  explain v (a residual energy fraction above 1/2 with several worlds)."
  [v]
  (let [[ws resid] (h/readout (m :C) v {:int-lo (cfg :int-lo) :int-hi (cfg :int-hi)})
        ws (vec (sort-by (fn [[_ _ w]] (- w)) ws))
        {cap-w :worlds cap-i :ints} (capacity)
        n-int (count (filter #(= NUM (first %)) ws))
        n-ptr (- (count ws) n-int)]
    (when (> n-ptr cap-w) (over-capacity "value" n-ptr cap-w))
    (when (> n-int cap-i) (over-capacity "integer" n-int cap-i))
    (when (and (> (count ws) 1) (> resid 0.5))
      (throw (ex-info (str "unreadable superposition: " (count ws) " worlds leave "
                           (Math/round (* 100 resid)) "% of the energy unexplained")
                      {:residual resid})))
    ws))

(defn worlds-of
  "[[world-vector weight] ...] of v. A clean single world is [[v 1.0]], with
  v itself (so a noisy single value is passed on exactly as before)."
  [v]
  (if (single? v)
    [[v 1.0]]
    (let [ws (readout v)]
      (if (<= (count ws) 1)
        [[v 1.0]]
        (mapv (fn [w] [(world-vec w) (nth w 2)]) ws)))))

(defn- superposed? [v] (> (count (worlds-of v)) 1))

(defn superpose-vecs
  "Σ wᵢ vᵢ over [[v w] ...], weights normalised to sum 1. A single world
  comes back as itself."
  [pairs]
  (let [pairs (filter (fn [[v w]] (and v (pos? w))) pairs)
        t (reduce + (map second pairs))]
    (when-not (pos? t)
      (throw (ex-info "superposition of no worlds (every weight is 0)" {})))
    (if (= 1 (count pairs))
      (ffirst pairs)
      (h/lincomb (sp) (map first pairs) (map #(/ (second %) t) pairs)))))

(defn- product
  "Joint worlds of several independent superpositions: [[[v ...] w] ...]."
  [wss]
  (reduce (fn [acc ws]
            (for [[vs w] acc [v w2] ws] [(conj vs v) (* w w2)]))
          [[[] 1.0]] wss))

(defn- enumerate
  "Σ over the joint worlds of `args` of weight · f(world args). Singles pass
  through untouched; with every argument single this is just (f args)."
  [f args]
  (let [wss (mapv worlds-of args)]
    (if (every? #(= 1 (count %)) wss)
      (f args)
      (let [n (reduce * (map count wss))
            cap (:worlds (capacity))]
        (when (> n (* cap cap)) (over-capacity "joint" n (* cap cap)))
        (superpose-vecs (for [[vs w] (product wss)] [(f vs) w]))))))

;; ---------------------------------------------------------------------------
;; host API

(defn- round4 [x] (/ (Math/round (* 1e4 (double x))) 1e4))

(defn worlds
  "The host map {decoded-value weight} of vector v (weights sum to 1)."
  [v]
  (->> (worlds-of v)
       (map (fn [[wv w]] {(core/decode wv) w}))
       (apply merge-with +)))

(defn weight "Weight of host value `x` in vector v." [v x]
  (get (worlds v) x 0.0))

(defn tv-distance
  "Total variation distance between two host distributions {value p}."
  [p q]
  (* 0.5 (reduce + (for [k (into (set (keys p)) (keys q))]
                     (Math/abs (- (double (get p k 0.0)) (double (get q k 0.0))))))))

(defn eval-form "Evaluate host form, return the result vector." [form] (core/eval-form form))

(defn run-worlds
  "Evaluate host `form` and return its worlds as a host map."
  [form]
  (worlds (core/eval-form form)))

(defn run "Evaluate host `form` and decode it (superpositions as #worlds)." [form]
  (core/run form))

(defn run-string [src] (core/run-string src))

;; ---------------------------------------------------------------------------
;; hooks

(defn- decode-hook [v]
  (let [ws (worlds-of v)]
    (if (= 1 (count ws))
      (core/decode-plain (ffirst ws))
      (tagged-literal 'worlds
                      (->> ws
                           (map (fn [[wv w]] {(core/decode wv) w}))
                           (apply merge-with +)
                           (into {} (map (fn [[k w]] [k (round4 w)]))))))))

(defn- construct-hook
  "Constructors enumerate: a list/vector/map/set with superposed elements is
  the superposition of the collections built from each joint world."
  [k f xs]
  (let [xs (vec xs)
        flat (if (= k :map) (vec (mapcat identity xs)) xs)]
    (if (every? single? flat)
      (f xs)
      (enumerate (fn [vs] (f (if (= k :map) (mapv vec (partition 2 vs)) vs))) flat))))

(def ^:private native-prims
  "Primitives that take superpositions as they are: arithmetic is binding,
  which distributes over sums; the substrate ops are linear or are
  measurements; the worlds primitives handle superpositions themselves."
  '#{inc dec + - bind unbind bundle cleanup similarity degrade memory-size
     println apply eval
     amb superpose worlds weight collapse sample assert support without
     union relation follow})

(defn- cells-of-one-kind?
  "Are all worlds non-empty cells of one sequence kind (so first/rest lift
  linearly through the memory)?"
  [ws]
  (let [labels (map first ws)
        kinds (set (map label->kind labels))
        empties (m :empty-idx)]
    (and (= 1 (count kinds))
         (#{:list :vec} (first kinds))
         (not-any? (fn [[l i]] (and (not= l NUM) (contains? empties i))) ws))))

(defn- apply-prim [f args]
  (let [nm (core/prim-name f)
        plain #(core/apply-plain f %)]
    (cond
      (contains? native-prims nm) (plain args)
      (every? single? args) (plain args)
      (#{'first 'rest} nm)
      (let [ws (readout (first args))]
        (if (and (> (count ws) 1) (cells-of-one-kind? ws))
          ;; straight through the memory: the kind checks of the plain
          ;; primitive see only the heaviest world
          ((if (= 'first nm) core/car core/cdr) (first args))
          (enumerate plain args)))
      :else (enumerate plain args))))

(defn- apply-hook [f args]
  (let [fws (worlds-of f)]
    (if (> (count fws) 1)
      ;; a superposed function: apply each world of it
      (superpose-vecs (for [[fv w] fws] [(core/apply-fn fv args) w]))
      (case (first (core/kind f))
        ;; closures run on superposed arguments as they are: the body is
        ;; evaluated once, in an environment that holds the superpositions
        :fn (core/apply-plain f args)
        :prim (apply-prim f args)
        (enumerate #(core/apply-plain f %) args)))))

;; -- if ----------------------------------------------------------------------

(defn- truth
  "π: the weight of the worlds of test value tv that are truthy."
  [tv]
  (let [s-nil (h/sim (sp) tv (nil-vec))
        s-false (h/sim (sp) tv (false-vec))
        floor (/ 4.5 (Math/sqrt (m :dim)))]
    (cond
      (and (< s-nil floor) (< s-false floor)) 1.0
      (or (> s-nil 0.995) (> s-false 0.995)) 0.0
      :else (reduce + (for [[l _ w] (readout tv)
                            :when (not (#{:nil :false} (label->kind l)))]
                        w)))))

(defn- form-symbols
  "The distinct symbols of expression e outside quote (candidate variables
  a test depends on)."
  [e]
  (let [walk (fn walk [e]
               (let [[k e?] (core/kind e)]
                 (cond
                   (= k :sym) [e]
                   e? []
                   (= k :list) (let [xs (cells e)]
                                 (if (core/eq? (first xs) (core/encode 'quote)) [] (mapcat walk xs)))
                   (= k :vec) (mapcat walk (cells e))
                   :else [])))]
    (reduce (fn [acc s] (if (some #(core/eq? s %) acc) acc (conj acc s))) [] (walk e))))

(defn- superposed-vars
  "[[sym worlds] ...] for the symbols of test-form bound to superpositions."
  [env test-form]
  (vec (for [s (form-symbols test-form)
             :let [v (try (#'core/lookup s env) (catch clojure.lang.ExceptionInfo _ nil))]
             :when v
             :let [ws (worlds-of v)]
             :when (> (count ws) 1)]
         [s ws])))

(defn- mix
  "Σ w·value over [[value w] ...] for the branches that ran."
  [pairs]
  (superpose-vecs (filter first pairs)))

(defn- restrict
  "The truthy (or falsy) worlds of test value tv, renormalised: the test
  value as one branch sees it (and/or return it)."
  [tv truthy-part?]
  (let [ws (worlds-of tv)
        kept (filter (fn [[v _]] (= truthy-part? (boolean (truthy? v)))) ws)]
    (if (or (= 1 (count ws)) (empty? kept)) tv (superpose-vecs kept))))

(declare branch-hook)

(defn- split-branch [env test-form tv pi then-fn else-fn]
  (let [eps (cfg :eps)
        vars (superposed-vars env test-form)]
    (case (count vars)
      ;; the test is superposed but depends on no superposed variable (it
      ;; is, say, (amb true false)): both branches in the same environment
      0 (mix [[(then-fn env (restrict tv true)) pi]
              [(else-fn env (restrict tv false)) (- 1.0 pi)]])
      ;; the split: each world of the variable goes to the branch its test
      ;; value takes (to both, weighted, if the test is itself uncertain
      ;; within that world)
      1 (let [[sym ws] (first vars)
              per (vec (for [[v w] ws]
                         [v w (truth (core/veval test-form (env-extend env sym v)))]))
              p-t (reduce + (for [[_ w t] per] (* w t)))
              p-f (reduce + (for [[_ w t] per] (* w (- 1.0 t))))
              part (fn [f] (superpose-vecs (for [[v w t] per] [v (* w (f t))])))
              run (fn [f truthy-part? weight-of]
                    (let [env' (env-extend env sym (part weight-of))]
                      (f env' (restrict (core/veval test-form env') truthy-part?))))
              a (when (> p-t eps) (run then-fn true identity))
              b (when (> p-f eps) (run else-fn false #(- 1.0 %)))]
          (mix [[a p-t] [b p-f]]))
      ;; several superposed variables: without world identity the test
      ;; cannot be split per variable, so enumerate their joint worlds (they
      ;; are independent: run-time choice) and decide each one
      (let [syms (map first vars)]
        (mix (for [[vs w] (product (map second vars))]
               (let [env' (reduce (fn [e [s v]] (env-extend e s v)) env (map vector syms vs))]
                 [(branch-hook env' test-form (core/veval test-form env') then-fn else-fn) w])))))))

(defn- branch-hook
  "if with a superposed test: π : 1−π, lazy at ε, the test's variable split."
  [env test-form tv then-fn else-fn]
  (let [pi (truth tv)
        eps (cfg :eps)]
    (cond
      (> pi (- 1.0 eps)) (then-fn env tv)
      (< pi eps) (else-fn env tv)
      :else (split-branch env test-form tv pi then-fn else-fn))))

(defn- global-hook
  "A global's value: F holds ν(L⊗sym + R⊗val), so R⊘slot carries the
  crosstalk R⊘L⊗sym. For a superposed value, subtract the known L field
  first, so the least squares weights are exact (as for cells)."
  [sym slot]
  (let [L (get (m :roles) :L) R (get (m :roles) :R)
        l (h/bind (sp) L sym)]
    (core/cleanup (h/unbind (sp) R (h/lincomb (sp) [slot l] [1.0 (- (h/coef slot l))])))))

;; -- for-worlds -----------------------------------------------------------------

(defn- for-worlds*
  "(for-worlds [v x w y ...] body ...): call-time choice. Each binding is
  enumerated, and inside body v is one world of x; the result is
  Σ weight · body."
  [bindings body env]
  (if-let [[sym xe & more] (seq bindings)]
    (superpose-vecs (for [[v w] (worlds-of (core/veval xe env))]
                      [(for-worlds* more body (env-extend env sym v)) w]))
    (#'core/eval-body body env)))

(defn- special-hook [sf args env]
  (case sf
    for-worlds (for-worlds* (vec (cells (core/car args))) (core/cdr args) env)
    (throw (ex-info (str "unknown special form " sf) {}))))

;; ---------------------------------------------------------------------------
;; dialect primitives

(defn- int-of [v] (#'core/int-of v))
(defn- permille [w] (num-vec (Math/round (* 1000.0 (double w)))))

(defn- prim-superpose
  "(superpose {v w, ...}): integer relative weights."
  [[mv]]
  (superpose-vecs (for [[k x] (#'core/map-entries mv)] [k (double (int-of x))])))

(defn- prim-amb [xs]
  (when (empty? xs) (throw (ex-info "(amb) of no worlds" {})))
  (superpose-vecs (for [x xs] [x 1.0])))

(defn- prim-worlds
  "(worlds x): a map {world weight-per-mille}."
  [[x]]
  (core/mk-map (for [[v w] (worlds-of x)] [v (permille w)])))

(defn- world-weight [x y]
  (let [[yv] (first (worlds-of y))]
    (reduce + 0.0 (for [[v w] (worlds-of x) :when (core/eq? v yv)] w))))

(defn- prim-weight
  "(weight x v): the weight of world v in x, per mille."
  [[x y]]
  (permille (world-weight x y)))

(defn- prim-collapse
  "(collapse x): the heaviest world (measurement). (collapse x β): partial
  collapse, weights ∝ exp(β·wᵢ/|w|₂). This distorts the weights (W0
  section 4) and is labelled as collapse, not as a soft superposition."
  [[x beta]]
  (let [ws (worlds-of x)]
    (if (nil? beta)
      (first (apply max-key second ws))
      (let [b (double (int-of beta))
            n2 (Math/sqrt (reduce + (map #(* (second %) (second %)) ws)))
            top (apply max (map second ws))]
        (superpose-vecs (for [[v w] ws] [v (Math/exp (* b (/ (- w top) n2)))]))))))

(defn- prim-sample
  "(sample x): one world, drawn with probability ∝ weight (seeded)."
  [[x]]
  (let [ws (worlds-of x)
        r (.nextDouble ^java.util.Random (cfg :rng))]
    (loop [[[v w] & more] ws acc 0.0]
      (if (or (empty? more) (< r (+ acc w))) v (recur more (+ acc w))))))

(defn- prim-assert
  "(assert p x): post-selection. Each world v of x keeps weight w·[p v],
  renormalised; throws if no world survives. Classical conditioning."
  [[p x]]
  (let [kept (for [[v w] (worlds-of x)
                   :let [t (truth (core/apply-fn p [v]))]
                   :when (pos? t)]
               [v (* w t)])]
    (when (empty? kept)
      (throw (ex-info "assert: no world satisfies the predicate" {})))
    (superpose-vecs kept)))

(defn- prim-support
  "(support x): the uniform superposition over the worlds of x (re-binarise:
  set semantics, as in a BFS frontier)."
  [[x]]
  (superpose-vecs (for [[v _] (worlds-of x)] [v 1.0])))

(defn- n-worlds
  "The number of worlds of a uniform superposition y (as `support` makes
  them), from its norm alone: |y|₂² = 1/n. No readout of y."
  [y]
  (let [c (h/coef y y)]
    (max 1.0 (/ 1.0 (* c c)))))

(defn- prim-without
  "(without x y): the worlds of x that are not worlds of y, weights
  renormalised; nil when none is left. y is never read out: a world v of x
  is in y when its coefficient v·y clears half of what one world of a
  uniform y carries (1/n, n from y's norm). So y may hold more worlds than
  `worlds` could read (a BFS visited set), within that SNR."
  [[x y]]
  (let [half (/ 0.5 (n-worlds y))
        kept (remove (fn [[v _]] (> (h/coef y v) half)) (worlds-of x))]
    (if (empty? kept) (nil-vec) (superpose-vecs kept))))

(defn- prim-union
  "(union x y): the uniform superposition over the worlds of x and of y,
  for uniform, disjoint x and y (a BFS visited set and a new frontier),
  weighted by their world counts estimated from their norms: no readout."
  [[x y]]
  (let [nx (n-worlds x) ny (n-worlds y)]
    (h/lincomb (sp) [x y] [(/ nx (+ nx ny)) (/ ny (+ nx ny))])))

(def ^:private relation-count (atom 0))

(defn- prim-relation
  "(relation {k #{v ...} ...}), or a seq of [k [v ...]] pairs (no map
  capacity limit): a heteroassociative relation held in the item memory,
  one row per key, R⊗k ↦ the uniform superposition of its v (so `follow`
  is one step of a random walk). Returns R, a fresh atom (it prints as a
  symbol relation-N). Keys should be atoms (keywords, symbols): integer keys
  are correlated in the residue encoding."
  [[spec]]
  (let [entries (if (= :map (first (core/kind spec)))
                  (#'core/map-entries spec)
                  (map #(vec (cells %)) (cells spec)))
        r (#'core/intern-atom! :sym (symbol (str "relation-" (swap! relation-count inc))))]
    (doseq [[k vs] entries
            :let [members (case (first (core/kind vs))
                            :set (#'core/set-elems vs)
                            (:list :vec) (cells vs)
                            :nil []
                            [vs])]
            :when (seq members)]
      (h/associate! (m :M) (h/bind (sp) r k)
                    (h/lincomb (sp) members (repeat (count members) (/ 1.0 (count members)))) 16))
    r))

(defn- prim-follow
  "(follow r x): the successors of every world of x in relation r, in one
  lookup: cleanup(deref(R⊗x)), weights ∝ Σ over the worlds u of x of
  w(u)/|succ(u)| per successor. Worlds of x that are not keys drop out; nil
  if none is."
  [[r x]]
  (or (h/follow (m :C) (h/bind (sp) r x)) (nil-vec)))

(def ^:private prims
  [['amb prim-amb]
   ['superpose prim-superpose]
   ['worlds prim-worlds]
   ['weight prim-weight]
   ['collapse prim-collapse]
   ['sample prim-sample]
   ['assert prim-assert]
   ['support prim-support]
   ['without prim-without]
   ['union prim-union]
   ['relation prim-relation]
   ['follow prim-follow]])

;; ---------------------------------------------------------------------------
;; boot

(defn init!
  "A machine for superposition programming: vsc.core/init! with the :proj
  memory (unless :memory says otherwise), then the hooks, primitives and
  for-worlds. Extra options:
    :eps        pruning weight: worlds and branches below it are dropped (0.02)
    :int-range  integers are read as worlds only in [-r, r] (1024)"
  ([] (init! {}))
  ([opts]
   (let [opts (merge {:memory :proj} opts)
         r (long (get opts :int-range 1024))]
     (core/init! opts)
     (swap! @#'core/machine assoc :worlds
            {:eps (double (get opts :eps 0.02))
             :int-lo (- r) :int-hi r
             :rng (java.util.Random. (long (get opts :seed 42)))})
     (h/configure-cleanup! (m :C) {:int-lo (- r) :int-hi r
                                   :roles [(get (m :roles) :L) (get (m :roles) :R)]})
     (doseq [[nm f] prims] (core/add-primitive! nm f))
     (core/add-special! 'for-worlds)
     (core/set-hooks! {:apply apply-hook
                       :branch branch-hook
                       :construct construct-hook
                       :decode decode-hook
                       :global global-hook
                       :special special-hook})
     :ready)))
