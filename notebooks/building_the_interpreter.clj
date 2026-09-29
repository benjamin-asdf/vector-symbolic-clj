^:kindly/hide-code
(ns building-the-interpreter
  (:require
   [scicloj.kindly.v4.kind :as kind]
   [vsc.hdc :as h]))

;; # Building a Vector-Symbolic Interpreter
;;
;; *The code behind [A Vector-Symbolic Clojure](../), step by step.
;; September 2026.*
;;
;; This notebook builds a small interpreter from nothing but vector
;; operations. At the end it runs `(fact 5)`, and every value along the way
;; is one vector. The finished interpreter
;; ([`src/vsc/core.clj`](https://github.com/benjamin-asdf/vector-symbolic-clj/blob/main/src/vsc/core.clj))
;; does the same with more care. The last section lists what it adds.
;;
;; The only library used is the substrate bridge `vsc.hdc`: thin Clojure
;; wrappers around numpy, reached through libpython-clj. A vector is an
;; opaque Python object. Clojure never looks inside one.

^:kindly/hide-code
(defn table
  "A small table from a seq of maps."
  [rows]
  (kind/table {:column-names (keys (first rows))
               :row-vectors (map vals rows)}))

;; ## 1. Random vectors are almost orthogonal
;;
;; A space of D = 2048 dimensions, and a way to make random "unitary"
;; vectors: vectors whose binding (step 2) can be undone exactly.

(def D 2048)

(def S (h/space D 7))

(defn random-vec [] (h/unitary S))

(defn sim [a b] (h/sim S a b))

;; A vector is fully similar to itself, and nearly unrelated to any other.
;; Across many random pairs, the similarity scatters around 0 with a spread
;; of about 1/√D = 0.022.

(let [sims (repeatedly 500 #(sim (random-vec) (random-vec)))
      mean (/ (reduce + sims) (count sims))
      sd (Math/sqrt (/ (reduce + (map #(Math/pow (- % mean) 2) sims)) (count sims)))]
  {:self (let [v (random-vec)] (sim v v))
   :random-pairs-mean mean
   :random-pairs-sd sd
   :one-over-sqrt-d (/ 1 (Math/sqrt D))})

;; ## 2. Binding, unbinding, bundling
;;
;; Binding glues two vectors into one that looks like neither. Unbinding
;; with one of them gives the other back, exactly.

(def role (random-vec))
(def value (random-vec))
(def glued (h/bind S role value))

{:glued-vs-role (sim glued role)
 :glued-vs-value (sim glued value)
 :unbound-vs-value (sim (h/unbind S role glued) value)}

;; Bundling adds vectors. The sum is similar to each part and to nothing
;; else, so it is a set. With three parts each shares 1/√3 ≈ 0.577.

(let [[a b c d] (repeatedly 4 random-vec)
      bag (h/normalize S (h/bundle S a b c))]
  {:bag-vs-a (sim bag a) :bag-vs-b (sim bag b) :bag-vs-outsider (sim bag d)})

;; ## 3. A cleanup memory, and symbols
;;
;; Unbinding from a bundle gives a noisy vector. The cleanup memory M snaps
;; it to the nearest vector it has stored. A symbol is a random vector
;; stored in M. Clojure keeps a lexicon only for printing: from a row of M
;; back to a name.

(def M (h/memory S))

(def lexicon (atom {}))
(def names (atom {}))

(defn sym
  "The vector for symbol `s`, made and stored on first use."
  [s]
  (or (@lexicon s)
      (let [i (h/mem-add! M (random-vec) 0)
            v (h/mem-get M i)]
        (swap! lexicon assoc s v)
        (swap! names assoc i s)
        v)))

(defn name-of [v] (@names (first (h/nearest M v))))

;; Bury `a` under noise three times its own size. The noisy copy is only
;; about 30% similar to `a`, yet cleanup still finds it, because every
;; other stored symbol is near 0.

(let [noisy (h/degrade S (sym 'a) 3.0)]
  {:noisy-vs-a (sim noisy (sym 'a))
   :noisy-vs-b (sim noisy (sym 'b))
   :cleaned-up (name-of noisy)})

;; ## 4. Cons cells, first attempt
;;
;; A list cell says "first is a, rest is b". Bind a to a role L and b to a
;; role R, and bundle them. This is the encoding of the original paper.

(def L (random-vec))
(def R (random-vec))

(defn cell-trace [a b]
  (h/normalize S (h/bundle S (h/bind S L a) (h/bind S R b))))

(let [c (cell-trace (sym 'x) (sym 'y))]
  {:first (name-of (h/unbind S L c))
   :rest (name-of (h/unbind S R c))})

;; It works for one cell. Recursion, though, builds many nearly equal cells:
;; ((a . v₁) . ()), ((a . v₂) . ()), … Reading `rest` from a cell must pick
;; its own chain, not a look-alike. The experiment below builds 32 such
;; chains, reads `rest` from a cell on top of each, and measures by how much
;; the right chain beats the best look-alike. It also measures how often
;; the right chain still wins under noise.

(defn confusion
  "Margins of a rest read against look-alike chains, for superposed cells
  (:superposed) or cells behind their own random pointer (:pointer)."
  [encoding]
  (let [mem (h/memory S)
        store #(h/mem-get mem (h/mem-add! mem % 0))
        a (store (random-vec))
        empty (store (random-vec))
        cell (fn [x y]
               (let [t (cell-trace x y)]
                 (if (= encoding :superposed)
                   (store t)
                   (h/mem-get mem (h/intern! mem (random-vec) t 10)))))
        trace-of #(if (= encoding :superposed) % (h/deref-ptr mem %))
        chains (vec (for [_ (range 32)] (cell (cell a (store (random-vec))) empty)))
        probes (vec (for [ch chains]
                      (h/unbind S R (trace-of (cell (cell a (store (random-vec))) ch)))))
        margin (fn [i probe]
                 (- (sim probe (chains i))
                    (apply max (for [j (range 32) :when (not= i j)] (sim probe (chains j))))))
        wins (fn [sigma]
               (/ (count (filter (fn [i]
                                   (let [p (h/degrade S (probes i) sigma)]
                                     (= i (apply max-key #(sim p (chains %)) (range 32)))))
                                 (range 32)))
                  32.0))
        ms (map-indexed margin probes)]
    {:encoding encoding
     :look-alike-similarity (sim (chains 0) (chains 1))
     :mean-margin (/ (reduce + ms) 32)
     :right-chain-at-noise-4 (wins 4.0)
     :right-chain-at-noise-8 (wins 8.0)}))

(table [(confusion :superposed) (confusion :pointer)])

;; As bundles, the look-alike chains are 75% similar, and the right chain
;; wins by a quarter of the margin that pointers give. Under heavy noise
;; (σ = 8) the bundled cells find their own chain less than half the time.
;; A program makes thousands of such reads. The original paper's cells also
;; bundle a marker φ into every cell, which makes look-alikes even more
;; similar (89%, margin 0.05). This is failure mode 1 in the overview
;; notebook.

;; ## 5. Cons cells behind pointers
;;
;; The fix: a cell's identity is a fresh random *pointer*, and the memory
;; maps the pointer to the cell's trace (a heteroassociative memory). Two
;; cells with equal contents are *hash-consed* onto one pointer, so equality
;; is still one comparison. Pointers get the label 10 in M, which tells a
;; cell from a symbol.

(def CELL 10)

(defn kons [a b] (h/mem-get M (h/intern! M (random-vec) (cell-trace a b) CELL)))

;; Integers come from the substrate's residue encoding: n = Bⁿ⊗Z, cleaned up
;; by reading off each residue. General cleanup takes whichever is closer,
;; a stored row or an integer.

(def N (h/numbers S))

(defn num [n] (h/num-vec N n))

(defn cleanup [v]
  (let [[n s-num] (h/read-num N v)
        [i s-row] (h/nearest M v)]
    (if (> s-num s-row) (num n) (h/mem-get M i))))

(defn kar [p] (cleanup (h/unbind S L (h/deref-ptr M p))))
(defn kdr [p] (cleanup (h/unbind S R (h/deref-ptr M p))))

(let [p (kons (sym 'x) (num 7))]
  {:first (name-of (kar p))
   :rest-is-7 (first (h/read-num N (kdr p)))
   :same-pointer-for-equal-cells (sim p (kons (sym 'x) (num 7)))})

;; ## 6. What kind of value is this?
;;
;; A value's kind is whatever recognises it: the integer readout, the empty
;; list, a pointer row (label 10), or a symbol row.

(def EMPTY (sym '()))
(def TRUE (sym 'true))
(def FALSE (sym 'false))

(defn kind [v]
  (let [[_ s-num] (h/read-num N v)
        [i label s] (h/recall M v)]
    (cond
      (> s-num (max s 0.5)) :num
      (< s 0.5) :unknown
      (> (sim v EMPTY) 0.9) :empty
      (= label CELL) :cell
      :else :sym)))

(defn same? [a b] (> (sim a b) 0.9))

;; Encoding host data into vectors, and decoding back. This is the only
;; place where host data and vectors meet.

(defn encode [x]
  (cond
    (int? x) (num x)
    (true? x) TRUE
    (false? x) FALSE
    (symbol? x) (sym x)
    (seq? x) (reduce (fn [acc e] (kons (encode e) acc)) EMPTY (reverse x))
    (vector? x) (encode (apply list x))
    :else (throw (ex-info (str "cannot encode " (pr-str x)) {}))))

(defn items
  "The elements of a list cell chain, as a host seq of vectors."
  [c]
  (when (= :cell (kind c)) (cons (kar c) (lazy-seq (items (kdr c))))))

(defn decode [v]
  (case (kind v)
    :num (first (h/read-num N v))
    :empty ()
    :cell (apply list (map decode (items v)))
    :sym (name-of v)
    '?))

(decode (encode '(fact (dec n) [a b] 42)))

;; ## 7. Environments
;;
;; An environment is a list of (name . value) cells, as in Lisp 1.5.
;; Extending it is one `kons`. Looking a name up walks the list. Globals
;; are one more such list, kept in a host atom that `def` updates. (The real
;; interpreter keeps globals in their own memory instead.)

(defn extend-env [env s v] (kons (kons s v) env))

(def globals (atom EMPTY))

(defn lookup [s env]
  (loop [e env global? false]
    (cond
      (= :cell (kind e)) (if (same? (kar (kar e)) s) (kdr (kar e)) (recur (kdr e) global?))
      global? s
      :else (recur @globals true))))

;; An unbound symbol evaluates to itself. That is how primitives work: `+`
;; evaluates to the symbol `+`, and application recognises it.

;; ## 8. The evaluator
;;
;; Arithmetic is binding. Since n = Bⁿ⊗Z, the sum a + b is a⊗b⊘Z, and
;; `inc` binds one more B.

(def Z (h/num-offset N))
(def B (h/num-step N))

(defn truthy [x] (if x TRUE FALSE))

(def primitives
  {'+ (fn [a b] (h/unbind S Z (h/bind S a b)))
   '- (fn [a b] (h/bind S Z (h/unbind S b a)))
   'inc (fn [a] (h/bind S B a))
   'dec (fn [a] (h/unbind S B a))
   'zero? (fn [a] (truthy (same? a (num 0))))
   '= (fn [a b] (truthy (same? a b)))
   'cons kons
   'first kar
   'rest kdr
   'empty? (fn [a] (truthy (same? a EMPTY)))
   'list (fn [& xs] (reduce (fn [acc x] (kons x acc)) EMPTY (reverse xs)))})

;; Special forms are recognised by similarity: the head of a form is
;; compared with the vectors of `quote`, `if`, `fn` and `def`.

(def special-forms (into {} (for [s '[quote if fn def]] [s (sym s)])))

(defn special [head]
  (when (= :sym (kind head))
    (some (fn [[s v]] (when (same? head v) s)) special-forms)))

;; A closure is a list (closure params body env), tagged with the symbol
;; `closure`.

(def CLOSURE (sym 'closure))

(declare evaluate)

(defn apply-fn [f args]
  (cond
    (and (= :cell (kind f)) (same? (kar f) CLOSURE))
    (let [[params body env] (items (kdr f))
          env (reduce (fn [e [p a]] (extend-env e p a)) env (map vector (items params) args))]
      (evaluate body env))

    (and (= :sym (kind f)) (primitives (name-of f)))
    (apply (primitives (name-of f)) args)

    :else (throw (ex-info (str "not a function: " (decode f)) {}))))

(defn evaluate [e env]
  (case (kind e)
    (:num :empty) e
    :sym (lookup e env)
    :cell (let [head (kar e)
                args (kdr e)]
            (case (special head)
              quote (kar args)
              if (let [[c t f] (items args)]
                   (if (same? (evaluate c env) FALSE)
                     (evaluate f env)
                     (evaluate t env)))
              fn (let [[params body] (items args)]
                   (kons CLOSURE (kons params (kons body (kons env EMPTY)))))
              def (let [[s x] (items args)]
                    (swap! globals extend-env s (evaluate x env))
                    s)
              (apply-fn (evaluate head env) (map #(evaluate % env) (items args)))))
    (throw (ex-info "cannot evaluate an unrecognised vector" {}))))

(defn run [form] (decode (evaluate (encode form) EMPTY)))

;; ## 9. Running programs

^:kindly/hide-code
(defn examples [forms]
  (kind/table
   {:column-names ["form" "result" "ms"]
    :row-vectors (for [f forms]
                   (let [t (System/nanoTime)
                         r (try (run f) (catch Exception ex (str "error: " (ex-message ex))))]
                     [(kind/code (pr-str f))
                      (kind/code (pr-str r))
                      (format "%.0f" (/ (- (System/nanoTime) t) 1e6))]))}))

(examples '[(+ 40 2)
            (first (quote (a b c)))
            ((fn [x] (+ x x)) 21)
            (if (zero? 0) (quote yes) (quote no))
            (cons 1 (list 2 3))])

;; Multiplication and factorial, defined in the language itself:

(run '(def * (fn [a b] (if (zero? b) 0 (+ a (* a (dec b)))))))

(run '(def fact (fn [n] (if (zero? n) 1 (* n (fact (dec n)))))))

(examples '[(* 6 7) (fact 5)])

;; `(fact 5)` makes 120 additions through the recursion of `*`, and every
;; variable lookup walks an environment of vectors. The memory grew to:

(h/mem-size M)

;; ## 10. From here to the real interpreter
;;
;; This evaluator shows the idea in about 150 lines. The real one,
;; [`src/vsc/core.clj`](https://github.com/benjamin-asdf/vector-symbolic-clj/blob/main/src/vsc/core.clj),
;; adds:
;;
;; - **Decisions without fixed thresholds.** `same?` above says "similar
;;   enough" at 0.9. That caps how much noise the interpreter survives,
;;   whatever D is (failure mode 5 in the overview). The real `eq?` cleans
;;   up both sides and compares identities, and `kind` compares with the
;;   chance level 1/√D.
;; - **Maps and sets** as bundles: `{k v}` is ν(Σ k⊗⟨v⟩) plus the set of
;;   keys and the size. `get` is one unbind and one cleanup. Keys are listed
;;   by *explaining away*: clean up, subtract what was found, repeat.
;; - **A global memory** F instead of a global list. It is probed with L⊗name,
;;   as in the original paper.
;; - **The residue integers in full**
;;   ([`resources/vsc/hdc.py`](https://github.com/benjamin-asdf/vector-symbolic-clj/blob/main/resources/vsc/hdc.py),
;;   class `Numbers`), and integers boxed behind pointers inside maps and
;;   sets.
;; - **Closures, varargs, `let`, `cond`, `and`, `or`,** and a prelude
;;   written in the dialect.
;; - **A CEK machine** in which the evaluator's own control state and rules
;;   are vectors
;;   ([`src/vsc/machine.clj`](https://github.com/benjamin-asdf/vector-symbolic-clj/blob/main/src/vsc/machine.clj)).
;; - **Superposition programming**
;;   ([`src/vsc/worlds.clj`](https://github.com/benjamin-asdf/vector-symbolic-clj/blob/main/src/vsc/worlds.clj)).
;;
;; The tests compare each of them with real Clojure
;; ([`test/`](https://github.com/benjamin-asdf/vector-symbolic-clj/tree/main/test)).
