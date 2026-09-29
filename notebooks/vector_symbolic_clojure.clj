^:kindly/hide-code
(ns vector-symbolic-clojure
  (:require
   [clojure.java.io :as io]
   [scicloj.kindly.v4.kind :as kind]
   [vsc.core :as vsc]
   [vsc.machine :as machine]
   [vsc.worlds :as w]))

;; # A Vector-Symbolic Clojure
;;
;; *Every value is one high-dimensional vector. September 2026.*
;;
;; This notebook runs a small Clojure in which every value is a single
;; vector of 2048 numbers. That covers lists, maps, sets, integers,
;; closures, and the environment the evaluator runs in. The evaluator never
;; looks inside a vector. It only binds, unbinds, adds and compares vectors,
;; and cleans up noisy ones against a memory.
;;
;; It is the idea of Tomkins-Flanagan and Kelly's "Hey Pentti, We Did It!"
;; (2024), which builds Lisp 1.5 this way, carried over to Clojure and
;; then run until it broke. Every result below is computed live when the
;; page renders.
;;
;; Code, tests, experiments and the paper draft:
;; [github.com/benjamin-asdf/vector-symbolic-clj](https://github.com/benjamin-asdf/vector-symbolic-clj).

^:kindly/hide-code
(defn figure
  "An image from figures/, embedded so the page stands alone."
  [file caption]
  (let [bytes (java.nio.file.Files/readAllBytes (.toPath (io/file "figures" file)))]
    (kind/hiccup
     [:figure
      [:img {:src (str "data:image/png;base64,"
                       (.encodeToString (java.util.Base64/getEncoder) bytes))
             :style {:max-width "100%"}}]
      [:figcaption {:style {:font-size "0.9em"}} caption]])))

^:kindly/hide-code
(defn examples
  "A table of forms and what the vector machine returns for them."
  [run forms]
  (kind/table
   {:column-names ["form" "result" "ms"]
    :row-vectors (for [f forms]
                   (let [t (System/nanoTime)
                         r (try (run f) (catch Exception e (str "error: " (ex-message e))))]
                     [(kind/code (pr-str f))
                      (kind/code (pr-str r))
                      (format "%.0f" (/ (- (System/nanoTime) t) 1e6))]))}))

^:kindly/hide-code
(defn worlds
  "The worlds of `form`, with weights rounded to 4 digits."
  [form]
  (let [r (w/run-worlds form)]
    (if (map? r)
      (into (sorted-map) (map (fn [[v p]] [v (/ (Math/round (* 1e4 (double p))) 1e4)])) r)
      r)))

;; ## Hyperdimensional computing in four operations
;;
;; Pick random vectors in 2048 dimensions and any two of them are almost
;; orthogonal: their cosine is about ±0.02. So there is room for far more
;; symbols than dimensions. Four operations build everything else.
;;
;; - **Binding** a⊗b (circular convolution) makes a vector unlike both a
;;   and b, from which either can be recovered given the other.
;; - **Unbinding** a⊘c recovers b from c = a⊗b. It is exact for the random
;;   "unitary" vectors used here.
;; - **Bundling** a + b makes a vector similar to both. That is a set.
;; - **Cleanup** replaces a noisy vector with the nearest stored one.
;;
;; These are holographic reduced representations (Plate, 1995).

;; ## Every value is one vector
;;
;; Start a machine, then read a map into vector space. What comes back is
;; a numpy array, and that array is the entire map.

(vsc/init!)

(kind/code (str (vsc/eval-form '{:name "Ada" :lang :clojure})))

;; The encoding:
;;
;; | value | vector |
;; |---|---|
;; | symbol, keyword, string | a random atom, stored in the memory M with its kind |
;; | `(a . b)`, `[a b]` | a pointer p, with M: p ↦ ν(L⊗a + R⊗b) |
;; | `first`, `rest` | unbind L or R from what p points to, then clean up |
;; | `{k v …}` | p ↦ ν(Σ k⊗⟨v⟩) plus the set of keys and the size |
;; | `#{x …}` | p ↦ ν(Σ x) plus the size |
;; | integer n | Bⁿ⊗Z, so `+` is binding |
;; | closure | p ↦ ν(L⊗form + R⊗env) |
;;
;; L, R, B and Z are fixed random vectors, and ν normalises to length 1.
;; A map is a *record*: `get` is one unbind and one cleanup. Map equality
;; ignores key order for free, because addition commutes.

(examples vsc/run
          '[(cons 'a '(b c))
            (:name {:name "Ada" :lang :clojure})
            (= {:a 1 :b 2} {:b 2 :a 1})
            (contains? #{:x :y :z} :y)
            (assoc {:a 1} :b [1 2])
            (+ 40 2)
            (eval (list '+ 1 2))])

;; Programs work too. The prelude's `map`, `filter` and `reduce` are
;; themselves written in this dialect.

(vsc/run '(defn fact [n] (if (zero? n) 1 (* n (fact (dec n))))))

(examples vsc/run
          '[(fact 5)
            (map (fn [x] (* x x)) (range 6))
            (reduce + 0 [1 2 3 4])
            (let [add (fn [a] (fn [b] (+ a b)))] ((add 3) 4))])

;; A small program takes seconds, where native Clojure takes microseconds.
;; Speed is not the point.

;; ## Noise
;;
;; Add noise as large as the signal to a list, and its first element still
;; comes back, because every read ends in a cleanup.

(examples vsc/run
          '[(first (degrade (quote (a b c)) 100))
            (get (degrade {:a 1 :b 2} 60) :b)])

;; ## What breaks
;;
;; Running real programs turned up five failure modes that the algebra
;; hides.
;;
;; **1. Superposed cells are fragile.** Encoded as a plain sum, as in the
;; original paper, the cells ((a . 5) . ()) and ((a . 4) . ()) have cosine
;; 0.89 to each other. A `rest` then beats the sibling by a margin of only
;; 0.05, and tolerates 2.5× less noise than a cell behind its own random
;; pointer. Equal contents are hash-consed onto one pointer, so equality is
;; still one comparison.

^:kindly/hide-code
(figure "paper-f2-cell-confusion.png"
        "Left: the margin of a rest cleanup over the best sibling. Right: accuracy under probe noise.")

;; **2. Binding commutes.** In `{1 2, 2 3}` the entry 1 ↦ 2 contributes
;; 1⊗2 = 2⊗1, so `(get {1 2 2 3} 2)` returned 1. Each value now gets its
;; own pointer, which no key can equal.
;;
;; **3. Residue integers form chimeras.** Integers use a residue number
;; system: one frequency band per prime 5, 7, …, 19. That makes cleanup
;; algorithmic (72 checks instead of a table scan), following Kymn et al.
;; (2025) and Hanley et al. (2025). But in the sum B¹¹ + B¹², every integer
;; whose residues each match 11 or 12 matches on every band. There are 62
;; such chimeras, and each scores exactly like a true member.

^:kindly/hide-code
(figure "paper-f3-chimeras.png"
        "Every chimera has cosine 0.665 to ν(B¹¹ + B¹²), the same as 11 and 12.")

;; Integers inside maps and sets are therefore boxed behind pointers.
;;
;; **4. Zero was one dimension.** B⁰ is the identity of convolution, a
;; spike at index 0. Killing that one dimension broke all arithmetic, so 0
;; is now the dense vector Z. The same residue structure makes sim(B⁰, Bⁿ)
;; the number of moduli dividing n, over 6. A threshold-based test would
;; call 5005 = 5·7·11·13 zero:

(examples vsc/run '[(zero? 5005) (zero? 0)])

;; **5. Fixed thresholds cap robustness.** The first interpreter recognised
;; a value's kind when its cosine exceeded 0.5. A clean vector's cosine to
;; its own noisy probe is 1/√(1 + σ²), which crosses 0.5 at σ = √3 for
;; *every* dimension. So every structure failed at σ ≈ 1.7, while the
;; cleanup underneath survives σ ≈ 14. Deciding by argmax, or relative to
;; the chance level 1/√D, makes robustness grow with √D again. Fixed
;; thresholds are common practice: Nengo's associative memory defaults to
;; 0.3 in every dimension.

;; ## How robust it is
;;
;; Primitive cleanup matches its textbook model within 0.007 on average,
;; and its noise tolerance grows as √D.

^:kindly/hide-code
(figure "e1.png" "Cleanup accuracy against probe noise: measured, and theory dashed.")

;; Capacity is linear in D: about D/64 map entries and D/36 set members at
;; 90% reliability. Overloaded structures forget; they do not hallucinate
;; members.

^:kindly/hide-code
(figure "e3.png" "Lookups against the number of entries.")

;; Storage survives the loss of 60–80% of dimensions. Computation survives
;; about 4%, because every bind re-loses the dead dimensions until the next
;; cleanup.

^:kindly/hide-code
(figure "e4c.png" "Lesions: stored structures against programs.")

;; ## Hopfield networks do not help
;;
;; The paper suggests a modern Hopfield network as the cleanup memory. It
;; never beats a plain lookup table. At high β it *is* the table, only
;; slower. At lower β, and whenever it iterates, it is worse. The reason:
;; when a probe is one stored item plus noise, argmax is already the best
;; possible decision.

^:kindly/hide-code
(figure "p2-e1.png" "Noise tolerance for the lookup table, Hopfield settings and a linear memory.")

;; ## The interpreter as vectors
;;
;; The evaluator above is host code. The next step moves it into vectors: a
;; CEK machine whose whole state is one vector,
;; s ↦ ν(C⊗control + E⊗env + K⊗continuation + M⊗mode).
;; Its 35 transition rules sit in a memory and are picked by similarity.
;; Rule keys are *sums* of features, so the most specific matching rule
;; wins by nearest neighbour, and shorter keys act as defaults. The host
;; loop is 47 lines and knows no Lisp. Tail calls come for free.

^:kindly/hide-code
(figure "paper-f8-cek-machine.png" "The vector CEK machine.")

;; Each step below is decoded from the state vector: the mode, the control,
;; the rule the nearest-neighbour lookup selected, and the frames on the
;; continuation.

(machine/init!)

(kind/code (with-out-str (machine/print-trace '((fn [x] (if (zero? x) :zero (inc x))) 1))))

;; ## Superposition programming
;;
;; A weighted sum of values is also a vector. So a variable can hold
;; several *worlds* at once, and the program computes on all of them.
;; Destructors act on every world in one memory read, and integer
;; arithmetic lifts exactly, because binding distributes over addition.

(w/init! {:dim 4096})

(examples worlds
          '[(+ (amb 1 2) 10)
            (first (amb (quote (a b)) (quote (c d))))
            (let [x (amb 1 2)] (+ x x))
            (for-worlds [x (amb 1 2)] (+ x x))])

;; Look at the last two rows. A vector carries no labels saying which
;; world a value belongs to, so the two uses of `x` are independent draws.
;; That is *run-time choice*. Call-time choice is explicit, with
;; `for-worlds`.
;;
;; This needed a new kind of memory. A lookup table and a softmax (Hopfield)
;; memory both collapse a sum of worlds to one world. A linear memory keeps
;; the worlds but lets every other stored item through as well. What works
;; is a memory that keeps only the stored items that clearly stand out
;; above chance and solves least squares on them. It holds about D/100
;; worlds.
;;
;; `if` on a superposed test runs both branches and weights them by the
;; test. Each world goes down its own branch, so recursion ends at a
;; different depth in each world:

(w/run '(defn dbl [n] (if (zero? n) 0 (+ 2 (dbl (dec n))))))

(examples worlds '[(dbl (superpose {2 5 3 3 5 2}))])

;; A small Bayesian network. Rain makes the sprinkler less likely, and
;; either makes the grass wet. What is P(rain | wet)? The exact answer is
;; 0.36255.

(w/run-string (slurp "examples/worlds/sprinkler.clj"))

(examples worlds '[rain-given-wet rain-given-wet-naive])

;; The naive version uses `let` instead of `for-worlds`, so run-time choice
;; loses the correlation between rain and wet grass, and it returns the
;; prior, 0.2.

^:kindly/hide-code
(figure "w-readout.png" "Worlds read back exactly, against the number of worlds and D.")

;; There is no speedup: a superposed step still scans the memory. What
;; superposition buys is a programming model, within a capacity of about
;; D/100 worlds, and only about 5 integer worlds.

;; ## Limits
;;
;; - Seconds per small program. Every cleanup scans a memory that only
;;   grows.
;; - No macros, `loop`/`recur`, floats or strings.
;; - Records hold about D/64 entries; superpositions about D/100 worlds.

;; ## References
;;
;; - Tomkins-Flanagan, E., & Kelly, M. A. (2024). Hey Pentti, we did it!: A
;;   fully vector-symbolic Lisp. MathPsych/ICCM 2024. arXiv:2510.17889.
;; - Hanley, C., Tomkins-Flanagan, E., & Kelly, M. A. (2025). Hey Pentti,
;;   we did (more of) it!: A vector-symbolic Lisp with residue arithmetic.
;;   IJCNN 2025. arXiv:2511.08767.
;; - Kymn, C. J., et al. (2025). Computing with residue numbers in
;;   high-dimensional representation. *Neural Computation*, 37(1).
;; - Meier, C. (2023). Vector symbolic architectures in Clojure.
;;   Clojure/Conj 2023. https://github.com/gigasquid/vsa-clj
;; - Plate, T. A. (1995). Holographic reduced representations. *IEEE
;;   Transactions on Neural Networks*, 6(3), 623–641.
;; - Ramsauer, H., et al. (2021). Hopfield networks is all you need. ICLR
;;   2021.
