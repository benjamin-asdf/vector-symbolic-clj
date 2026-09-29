^:kindly/hide-code
(ns vector-symbolic-clojure
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
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
;; **Why care.** Brains, and the neuromorphic chips modelled on them, do not
;; store bits in addresses. They store patterns spread over many noisy
;; units. Vector-symbolic architectures are a mathematical model of how
;; symbols and structure could live in such patterns. A language whose
;; values are all such vectors asks the question concretely: can you
;; *program* on that kind of substrate, and what goes wrong when you try?
;; It also puts data and code in one format, and it allows a new way of
;; programming, on superpositions of values.
;;
;; **Three findings.**
;;
;; - It works, but only after fixing five failure modes that the algebra
;;   hides (below).
;; - Stored data survives heavy damage. Computation is far more fragile.
;; - A value can be a weighted mix of several values, and the program
;;   computes on all of them at once. Getting this to work needed a new kind
;;   of memory.
;;
;; How the interpreter is built, step by step, with all the code:
;; [Building a Vector-Symbolic Interpreter](building/).
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
;; A symbol is a random vector of D = 2048 numbers. Two random vectors in
;; that many dimensions are almost unrelated: their similarity (the cosine
;; of the angle between them, 1 for equal, 0 for unrelated) is about ±0.02.
;; So there is room for far more symbols than dimensions. Four operations
;; build everything else.
;;
;; - **Binding** a⊗b glues two vectors together, like a name tag onto a
;;   value. The result looks like neither, but given one you can get the
;;   other back. (Technically: circular convolution.)
;; - **Unbinding** a⊘c peels the tag a off c = a⊗b and gives back b.
;; - **Bundling** a + b throws vectors into one bag. The bag is similar to
;;   each thing in it. That is a set.
;; - **Cleanup** snaps a noisy vector to the nearest vector the memory
;;   knows, like autocorrect snapping a typo to a real word.
;;
;; These are holographic reduced representations (Plate, 1995).

;; ## Every value is one vector
;;
;; Start a machine, then read a map into vector space. What comes back is
;; a numpy array, and that array is the entire map.

(vsc/init!)

(kind/code (str (vsc/eval-form '{:name "Ada" :lang :clojure})))

;; How a list becomes one vector: a list cell says "first is a, rest is b".
;; Bind a to a fixed role label L and b to a role label R, and bundle the
;; two: L⊗a + R⊗b. To read the first element, unbind L and clean up. Every
;; such cell also gets its own random ID (a *pointer*), for a reason
;; explained under "What breaks". The rest of the encoding follows the same
;; pattern. In the table, ν means "rescale to length 1", and ⟨v⟩ is v behind
;; its own pointer.
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
;; **1. Similar lists get confused.** In the original paper, a list cell is
;; just the bundle of its parts. Recursion creates many nearly equal cells,
;; such as ((a . 5) . ()) and ((a . 4) . ()), and as bundles they are 89%
;; similar. Reading one of them beats its look-alike by a margin of only
;; 0.05, so a little noise returns the wrong cell. The fix: every cell gets
;; its own random ID, and cells with equal contents share one ID, so
;; checking equality is still one comparison. The right cell now wins by
;; 0.66, and reading survives 2.5× more noise. (Noise σ here means noise of
;; σ times the vector's own size.)

^:kindly/hide-code
(figure "paper-f2-cell-confusion.png"
        "Left: the margin of a rest cleanup over the best sibling. Right: accuracy under probe noise.")

;; **2. Binding commutes.** In `{1 2, 2 3}` the entry 1 ↦ 2 contributes
;; 1⊗2 = 2⊗1, so `(get {1 2 2 3} 2)` returned 1. Each value now gets its
;; own pointer, which no key can equal.
;;
;; **3. Numbers can blend into fake numbers.** We store an integer by its
;; remainders, like reading six clocks at once: 11 is "1 on a 5-hour clock,
;; 4 on a 7-hour clock, 0 on an 11-hour clock", and so on up to a 19-hour
;; clock. Each clock lives in its own share of the vector. That makes
;; numbers cheap to recognise: read each clock, then combine (Kymn et al.,
;; 2025; Hanley et al., 2025).
;;
;; Now put 11 and 12 into one set. The set shows both numbers' readings on
;; every clock. A number that shows 11's reading on some clocks and 12's on
;; the others, 13102 for example, matches the set exactly as well as 11 or
;; 12 do. It is a *chimera*: a fake member stitched together from pieces of
;; real ones. With six clocks there are 2⁶ − 2 = 62 of them.

^:kindly/hide-code
(figure "paper-f3-chimeras.png"
        "Left: all 62 fakes score exactly like the real members 11 and 12; random numbers score near 0. Right: each clock of the fake 13102 matches either 11 or 12.")

;; The fix: inside sets and maps, each number is stored behind its own
;; random ID, and IDs cannot be stitched together.
;;
;; **4. Zero lived in a single dimension.** The number 0 was a vector with
;; all its weight in one of the 2048 dimensions. When that one dimension
;; was damaged, all arithmetic broke. Zero is now spread over all
;; dimensions like every other value. The clocks cause a second trap: 5005
;; = 5·7·11·13 reads 0 on four of the six clocks, so it is 4/6 similar to
;; zero, and a test "similar enough to zero" would accept it. `zero?` now
;; asks for equality:

(examples vsc/run '[(zero? 5005) (zero? 0)])

;; **5. Fixed thresholds throw robustness away.** Bigger vectors should
;; tolerate more noise, but a fixed "similar enough" cutoff does not grow
;; with them. The first interpreter recognised a value's kind when its
;; cosine exceeded 0.5. A clean vector's cosine to
;; its own noisy probe is 1/√(1 + σ²), which crosses 0.5 at σ = √3 for
;; *every* dimension. So every structure failed at σ ≈ 1.7, while the
;; cleanup underneath survives σ ≈ 14. Deciding by argmax, or relative to
;; the chance level 1/√D, makes robustness grow with √D again. Fixed
;; thresholds are common practice: Nengo's associative memory defaults to
;; 0.3 in every dimension.

;; ## How robust it is
;;
;; Here D is the number of dimensions and σ the noise, relative to the
;; signal. A single cleanup matches its textbook model within 0.007 on
;; average, and the noise it tolerates grows as √D: four times the
;; dimensions, twice the noise.

^:kindly/hide-code
(figure "e1.png" "How often cleanup finds the right item, against noise, for D from 512 to 8192 and memories of 100 to 100,000 items. Dashed: theory.")

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
;; The paper suggests a modern Hopfield network as the cleanup memory. Instead
;; of returning the one best match, it returns a weighted mix of the stored
;; items, and can repeat that step. A parameter β sets how hard it favours
;; the best match: at large β the mix is almost only the winner.
;;
;; It never beats a plain lookup table. At large β it *is* the table, only
;; slower. At smaller β, and whenever it repeats the step, it is worse. The
;; reason: when a probe is one stored item plus noise, picking the single
;; best match is already the best possible decision.

^:kindly/hide-code
(figure "p2-e1-summary.png" "The noise each memory tolerates (higher is better): the lookup table against Hopfield settings and a linear memory.")

;; ## The interpreter as vectors
;;
;; The evaluator above is host code. The next step moves it into vectors. A
;; classic design for interpreters, the CEK machine, keeps three things:
;; **C**, what to evaluate now; **E**, which variable has which value; and
;; **K**, a to-do list of what comes after. Here all three, plus a mode, are
;; bundled into one vector:
;; s ↦ ν(C⊗control + E⊗env + K⊗to-do + M⊗mode).
;; Its 35 transition rules sit in a memory and are picked by similarity.
;; Rule keys are *sums* of features, so the most specific matching rule
;; wins by nearest neighbour, and shorter keys act as defaults. The host
;; loop is 47 lines and knows no Lisp. Tail calls come for free.

;; One step of the machine goes round this loop. Everything in the blue
;; boxes is a vector in memory; only the loop itself is host code.

^:kindly/hide-code
(defn cek-diagram []
  (let [blue {:fill "#eef4fb" :stroke "#2c6fbb" :stroke-width 1.5 :rx 10}
        box (fn [x y title lines]
              (into [:g [:rect (merge blue {:x x :y y :width 320 :height 128})]
                     [:text {:x (+ x 16) :y (+ y 28) :font-size 15 :font-weight 600
                             :fill "#1b3a5c"} title]]
                    (map-indexed
                     (fn [i [a b]]
                       [:text {:x (+ x 16) :y (+ y 54 (* i 20)) :font-size 13 :fill "#222"}
                        [:tspan {:font-family "monospace" :font-weight 600 :fill "#2c6fbb"} a]
                        (str "  " b)])
                     lines)))
        arrow (fn [x1 y1 x2 y2 label lx ly anchor]
                [:g [:line {:x1 x1 :y1 y1 :x2 x2 :y2 y2 :stroke "#555" :stroke-width 1.6
                            :marker-end "url(#arrowhead)"}]
                 [:text {:x lx :y ly :font-size 12.5 :font-style "italic" :fill "#555"
                         :text-anchor anchor} label]])]
    (kind/hiccup
     [:svg {:viewBox "0 0 780 380" :style {:width "100%" :max-width "780px"
                                           :font-family "system-ui, sans-serif"}}
      [:defs [:marker {:id "arrowhead" :markerWidth 10 :markerHeight 8 :refX 9 :refY 4
                       :orient "auto"} [:path {:d "M0,0 L10,4 L0,8 z" :fill "#555"}]]]
      (box 20 16 "state s: one vector"
           [["C" "what to evaluate now"] ["E" "variables → values"]
            ["K" "to-do list (9 kinds of frame)"]])
      (box 440 16 "key(s): the state's shape"
           [["fetch" "12 instructions read s"] ["Σ" "mode + kind + head"]
            ["+" "identity + next frame"]])
      (box 440 234 "rule memory R"
           [["35" "rules, each with a key"] ["nearest" "key wins"]
            ["⇒" "the most specific rule"]])
      (box 20 234 "the rule's microprogram"
           [["list" "of instruction vectors"] ["9 ops" "FIELD CLEAN BIND UNBIND"]
            ["" "REC MOV JMPEQ PRIM DEF"]])
      (arrow 340 80 440 80 "read its shape" 390 70 "middle")
      (arrow 600 144 600 234 "nearest key" 611 194 "start")
      (arrow 440 298 340 298 "run it" 390 288 "middle")
      (arrow 180 234 180 144 "new state" 169 194 "end")
      [:rect {:x 262 :y 172 :width 256 :height 36 :rx 18 :fill "#fff4e0"
              :stroke "#d98a00" :stroke-width 1.5}]
      [:text {:x 390 :y 195 :font-size 13.5 :font-weight 600 :fill "#7a4b00"
              :text-anchor "middle"} "host loop · 47 lines · no Lisp"]])))

^:kindly/hide-code
(cek-diagram)

;; **Why the most specific rule wins.** A state's key always has all five
;; features. A rule's key names only the features it cares about: a
;; default rule names one, the rule for `if` names three. When all of a
;; rule's features match, its similarity to the state's key is √(k/5) for k
;; features. So when several rules match, the one that names the most
;; features is nearest, and a rule that names fewer acts as a default.

(kind/table
 {:column-names ["features the rule names" "example" "similarity to the state's key"]
  :row-vectors (for [[k ex] [[1 "any value evaluates to itself"]
                             [2 "a list is a function call"]
                             [3 "a list headed by if is an if"]]]
                 [k ex (format "%.3f" (Math/sqrt (/ k 5.0)))])})

;; A run, step by step. Each row is decoded from the state vector: the mode,
;; what is being evaluated, the rule that nearest-neighbour lookup chose,
;; and the to-do list. At step 14 the `if` frame is already gone: the branch
;; runs as a tail call, so the to-do list does not grow.

^:kindly/hide-code
(kind/hidden (machine/init!))

^:kindly/hide-code
(let [{:keys [value steps]} (machine/trace '((fn [x] (if (zero? x) :zero (inc x))) 1))
      badge {"EVAL" ["#e3eefb" "#1b4f8a"] "RET" ["#e5f5e8" "#1d6b34"]
             "APPLY" ["#fdf0dc" "#8a5300"] "HALT" ["#eee" "#333"]}
      cell {:padding "3px 10px" :border-bottom "1px solid #e5e5e5" :white-space "nowrap"}]
  (kind/hiccup
   [:div {:style {:overflow-x "auto" :font-size "0.85em"}}
    [:table {:style {:border-collapse "collapse" :margin "0.5em 0"}}
     [:thead
      (into [:tr]
            (for [h ["step" "mode" "evaluating" "rule chosen" "to-do list"]]
              [:th {:style (merge cell {:text-align "left" :border-bottom "2px solid #999"})} h]))]
     (into [:tbody]
           (for [{:keys [step mode control rule frames]} steps
                 :let [[bg fg] (get badge (str mode) ["#eee" "#333"])]]
             [:tr
              [:td {:style (merge cell {:text-align "right" :color "#888"})} step]
              [:td {:style cell}
               [:span {:style {:background bg :color fg :border-radius "4px"
                               :padding "1px 6px" :font-weight 600 :font-size "0.9em"}}
                (str mode)]]
              [:td {:style (merge cell {:font-family "monospace"})} (pr-str control)]
              [:td {:style cell} (str rule)]
              [:td {:style (merge cell {:color "#555"})}
               (str/join " → " (map #(str/replace (str %) "-fr" "") frames))]]))]
    [:p {:style {:margin-top "0.4em"}} "Result: " [:code (pr-str value)]]]))

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
;; That is *run-time choice*. The name comes from nondeterministic
;; programming, where Hussmann (1993) distinguished it from *call-time
;; choice*, in which a variable picks one value and keeps it. Languages
;; like Curry use call-time choice. Here it is explicit, with `for-worlds`.
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

;; The weights behave like probabilities, so a textbook Bayesian network
;; works. Rain makes the sprinkler less likely, and either makes the grass
;; wet. What is P(rain | wet)? The exact answer is 0.36255. Probability with
;; such vectors is not new: Furlong and Eliasmith (2024) and the
;; "hyperdimensional transform" (Dewulf et al., 2023) do inference this way.
;; What is new here is that it runs inside an ordinary programming language.

(w/run-string (slurp "examples/worlds/sprinkler.clj"))

(examples worlds '[rain-given-wet rain-given-wet-naive])

;; The naive version uses `let` instead of `for-worlds`, so run-time choice
;; loses the link between rain and wet grass, and it returns the prior, 0.2.
;; This query still enumerates the worlds one by one, so it is no faster
;; than ordinary code.

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
;; - Dewulf, P., De Baets, B., & Stock, M. (2023). The hyperdimensional
;;   transform for distributional modelling, regression and classification.
;;   arXiv:2311.08150.
;; - Furlong, P. M., & Eliasmith, C. (2024). Modelling neural probabilistic
;;   computation using vector symbolic architectures. *Cognitive
;;   Neurodynamics*, 18(6).
;; - Hussmann, H. (1993). *Nondeterminism in Algebraic Specifications and
;;   Algebraic Programs*. Birkhäuser.
;; - Plate, T. A. (1995). Holographic reduced representations. *IEEE
;;   Transactions on Neural Networks*, 6(3), 623–641.
;; - Ramsauer, H., et al. (2021). Hopfield networks is all you need. ICLR
;;   2021.
