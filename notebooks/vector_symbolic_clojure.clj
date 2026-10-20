^:kindly/hide-code
(ns vector-symbolic-clojure
  (:require
   [clojure.java.io :as io]
   [clojure.pprint]
   [clojure.repl]
   [clojure.string :as str]
   [scicloj.kindly.v4.kind :as kind]
   [vsc.core :as vsc]
   [vsc.hdc :as h]
   [vsc.machine :as machine]
   [vsc.worlds :as w]))

;; # A Vector-Symbolic Clojure
;;
;; > Generated with Claude Code, and I am not ashamed. My own blog posts of
;; > this kind are part of the training data and have sculpted its style.
;;
;; *Every value is one high-dimensional vector. September 2026, updated
;; October 2026.*
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
;; **What we learned.** Each point is measured below.
;;
;; 1. **Similar is confusable.** A cleanup's margin is 1 minus the
;;    similarity to the nearest wrong item. Built as bundles of their parts,
;;    the cells of a recursion are 89% alike, and reads fail. Give every
;;    structure its own random pointer, shared by equal contents: reads then
;;    survive 2.8× more noise. Keep similarity only where you want it, as in
;;    vectors.
;; 2. **Never decide with a fixed threshold.** A cutoff like "cosine above
;;    0.5" fails at noise σ ≈ 1.7 in *every* dimension, while the cleanup
;;    underneath survives σ ≈ 17. Decide by the best match, or relative to
;;    chance (1/√D), and robustness grows with √D.
;; 3. **Convolution commutes, so key–value pairs need direction.** Store an
;;    entry as k⊗Π(v), with Π a fixed random permutation, as one operation.
;;    A cyclic shift does not work, since it commutes with convolution too.
;; 4. **Residue-number integers blend into fake ones.** A superposition of
;;    11 and 12 matches 62 other integers exactly as well as its members.
;;    Inside sets and maps, keep integers behind pointers.
;; 5. **Stored data is robust, computation is fragile.** Lists still read
;;    back with 60–80% of the dimensions dead; `(fact 4)` breaks at about
;;    4%, because every bind passes the damage on until the next cleanup.
;; 6. **A Hopfield network cannot beat a lookup table at exact recall.**
;;    For a probe that is one stored item plus noise, picking the best match
;;    is already optimal. Measured: it ties at best and loses on look-alikes.
;; 7. **Programs can compute on superpositions.** A value can be a weighted
;;    mix of values, and destructors, arithmetic and `if` act on all of them
;;    at once. This needs a memory that keeps several matches instead of
;;    one; it holds about D/100 worlds. It makes no program faster.
;; 8. **An interpreter can live in vector memory.** A CEK machine whose 35
;;    rules are vectors, picked by similarity, runs these programs from a
;;    host loop of under 50 lines that knows no Lisp.
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
;; How the interpreter is built, step by step, with all the code:
;; [Building a Vector-Symbolic Interpreter](building/).
;;
;; How the design got here, roughly in the order the traps turned up:
;; [History](history/).
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
;; explained under "What the algebra hides". The rest of the encoding follows the same
;; pattern. In the table, L, R, V, KEYS, SIZE, B and Z are fixed random vectors, Bⁿ is
;; B bound to itself n times, Π shuffles the dimensions in a fixed random
;; order, and ν rescales to length 1.
;;
;; | value | vector |
;; |---|---|
;; | symbol, keyword, string | a random atom, stored in the memory M with its kind |
;; | `(a . b)` | a pointer p, with M: p ↦ ν(L⊗a + R⊗b) |
;; | `first`, `rest` | unbind L or R from what p points to, then clean up |
;; | `[x₀ … xₙ₋₁]` | ν(Σᵢ Πⁱ(V⊗xᵢ)) itself, with no pointer; M records its length next to it |
;; | `{k v …}` | p ↦ ν(ν(Σ k⊗Π(v)) + ν(KEYS⊗Σ k) + SIZE⊗n): the entries, the set of keys, and the size n |
;; | `#{x …}` | p ↦ ν(ν(Σ x) + SIZE⊗n) |
;; | integer n | Bⁿ⊗Z, so `inc` is one more binding with B, and `+` binds two numbers and unbinds one Z (Bᵐ⊗Bⁿ = Bᵐ⁺ⁿ) |
;; | closure | p ↦ ν(L⊗form + R⊗env) |
;;
;; A map is a *record*: `get` is one unbind and one cleanup. Map equality
;; ignores key order for free, because addition commutes.
;;
;; Lists and vectors are encoded differently on purpose. A list is a chain
;; of pointers: robust and unbounded, but two lists that differ in one
;; element are unrelated vectors. A vector puts element i at position Πⁱ
;; and has no pointer, so vectors with the same elements in the same places
;; are similar vectors, and `nth` is one readout. The price: a vector holds
;; about D/48 elements, and it tolerates less noise than a list, the less
;; the longer it is, because each element gets a smaller share of the
;; vector. The trade-off is the programmer's choice. Need any number of
;; elements that read back reliably, even under heavy noise? Use a list.
;; Need sequences that are similar when their elements are? Use a vector.

(examples vsc/run
          '[(similarity [:a :b :c :d] [:a :b :c :e])
            (similarity (list :a :b :c :d) (list :a :b :c :e))
            (nth [:a :b :c :d] 3)])

;; `similarity` answers in percent, since the dialect has no floats. The
;; two vectors share three of four positions and are about 75% alike; the
;; two lists are as unrelated as random vectors.

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

;; ## What the algebra hides
;;
;; Seven traps hide in the algebra, and each one shaped the design.
;;
;; **1. Similar lists get confused.** In the original paper, a list cell is
;; just the bundle of its parts. Recursion creates many nearly equal cells,
;; such as ((a . 5) . ()) and ((a . 4) . ()), and as bundles they are 89%
;; similar. Reading one of them beats its look-alike by a margin of only
;; 0.05, so a little noise returns the wrong cell, and `(fact 5)` fails
;; with "unable to resolve symbol: a". So every cell gets its own random
;; ID, and cells with equal contents share one ID, so checking equality is
;; still one comparison. The right cell then wins by 0.66, and reading
;; survives 2.8× more noise (σ ≈ 11 instead of 4). (Noise σ here means
;; noise of σ times the vector's own size.)
;;
;; The lesson is not "never let things be similar". A cleanup's margin is
;; 1 minus the similarity to the nearest wrong item, so similarity and
;; confusability are the same number. Lists and the interpreter's own
;; structures take the margin, and vectors take the similarity.

^:kindly/hide-code
(figure "paper-f2-cell-confusion.png"
        "Left: the margin of a rest cleanup over the best sibling, bundled cells against cells with IDs. Right: accuracy under probe noise.")

;; **2. Binding commutes.** The original paper binds by circular
;; convolution, and that commutes: a⊗b = b⊗a. Its list cells do not mind,
;; because the labels L and R tell first from rest. A map entry has no such
;; labels. Stored as k⊗v, the entry 1 ↦ 2 is 1⊗2 = 2⊗1, so
;; `(get {1 2 2 3} 2)` would answer 1.
;;
;; So an entry is a directed bind, k⊗Π(v), with Π the fixed random
;; shuffle from the table above, and reading it unbinds k and undoes Π.
;; The cheapest permutation, shifting every number one place along, would
;; not do: a shift commutes with circular convolution, so
;; shift(1)⊗2 = shift(2)⊗1. Π goes on the value and not on the key,
;; because keys unbind exactly and a shuffled key no longer does. Pairs
;; without a direction, where a–b is the same as b–a, have their uses too,
;; a symmetric relation for one. They can be a second structure, built on
;; the plain bind.
;;
;; **3. Numbers can blend into fake numbers.** An integer is stored by its
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
;; real ones. With six clocks there are 2⁶ − 2 = 62 of them. Here is the
;; sum of 11 and 12, built with the interpreter's own number encoding,
;; against its members, a chimera and a few ordinary numbers:

^:kindly/hide-code
(let [S (h/space 2048 1)
      N (h/numbers S)
      both (h/bundle S (h/num-vec N 11) (h/num-vec N 12))
      [read-back s] (h/read-num N (h/normalize S both))]
  (kind/table
   {:column-names ["number" "similarity to 11 + 12"]
    :row-vectors (concat
                  (for [n [11 12 13102 42 777 -5000]]
                    [(str n (when (= n 13102) " (chimera)"))
                     (format "%.3f" (h/sim S both (h/num-vec N n)))])
                  [[(str "clock readout of 11 + 12: " read-back)
                    (format "%.3f" s)]])}))

;; The readout, asked which number the sum is, answers with a chimera.

^:kindly/hide-code
(figure "paper-f3-chimeras.png"
        "Left: all 62 fakes score exactly like the real members 11 and 12; random numbers score near 0. Right: each clock of the fake 13102 matches either 11 or 12.")

;; So inside sets and maps, each number is stored behind its own random
;; ID, and IDs cannot be stitched together.
;;
;; **4. Partly equal is not equal.** The clocks set a second trap: 5005 =
;; 5·7·11·13 reads 0 on four of the six clocks, so it is 4/6 similar to
;; zero, and a test "similar enough to zero" would accept it. `zero?`
;; asks for equality:

(examples vsc/run '[(zero? 5005) (zero? 0)])

;; **5. Fixed thresholds throw robustness away.** Bigger vectors should
;; tolerate more noise, but a fixed "similar enough" cutoff does not grow
;; with them. Say a value's kind counts as recognised when its cosine
;; exceeds 0.5. A clean vector's cosine to its own noisy probe is
;; 1/√(1 + σ²), which crosses 0.5 at σ = √3 for *every* dimension. So
;; every structure would fail at σ ≈ 1.7, while the cleanup underneath
;; survives σ ≈ 17 at the size of these memories. Deciding by argmax, or
;; relative to the chance level 1/√D, makes robustness grow with √D again,
;; and that is how the interpreter decides. Fixed thresholds are common
;; practice: Nengo's associative memory defaults to 0.3 in every dimension.
;;
;; **6. A shared field makes strangers similar.** If a vector stored its
;; length inside itself, any two vectors of the same length would share
;; that part and look 20% alike, even with no element in common. That is
;; far above chance, and the superposition memory below treats anything
;; above chance as present, so unrelated vectors would creep into results.
;; So the length lives next to the vector in the memory, not inside it.
;; Maps and sets keep their size in their trace, and that is fine: their
;; value is the random pointer p, so two maps of one size are as unrelated
;; as any two pointers.
;;
;; **7. A sum of tuples forgets which values went together.** A vector is
;; a sum over its positions, so [1 1] + [2 2] is exactly [1 2] + [2 1]. A
;; superposition of vectors remembers how likely each value is at each
;; position, but not which values occurred together. Tuples whose
;; correlations matter are therefore built as lists.

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

;; Stored structures survive heavy damage: lists still read back with 60–80%
;; of the dimensions dead, sets with 57–70%, maps with 23–52%. A long
;; computation such as `(fact 4)` survives about 4%, because every bind
;; re-loses the dead dimensions until the next cleanup. A short one,
;; `(+ 17 25)`, survives about half.

^:kindly/hide-code
(figure "e4c.png" "Lesions: stored structures against programs.")

;; ## Hopfield networks do not help
;;
;; The paper suggests a modern Hopfield network as the cleanup memory. Instead
;; of returning the one best match, it returns a weighted mix of the stored
;; items, and can repeat that step. A parameter β sets how hard it favours
;; the best match: at large β the mix is almost only the winner.
;;
;; It cannot beat a plain lookup table here, and the reason comes before
;; any experiment. Every probe the interpreter makes is one stored item
;; plus noise, and for such a probe picking the single best match is
;; already the best possible decision. The Hopfield network scores the same
;; stored items the same way, so its first step picks the table's winner
;; at every β, and repeating the step only sharpens whatever was ahead. It
;; can tie the table or lose to it, and the experiments measure where it
;; loses.
;;
;; With unrelated stored items, β = 64 *is* the table, only slower, and
;; smaller β is worse:

^:kindly/hide-code
(figure "p2-e1-summary.png" "The noise each memory tolerates on unrelated stored items, relative to the lookup table (1 = as good as the table).")

;; Stored look-alikes are where it loses. In groups of 16 items at
;; similarity 7/8, β = 64 still holds for three steps, but ten steps pull
;; the state towards the centre of the group; only β = 256 matches the
;; table there. Inside the interpreter, one soft step at β = 64, which
;; matched the table on unrelated items, passes the blend of look-alikes on
;; as the value. So a well-set β depends on the closest pair of stored
;; items, not only on how many there are. Damage to the stored items costs
;; every memory the same.
;;
;; | noise tolerated (σ₅₀, higher is better) | table | β = 64, 3 steps | β = 64, 10 steps | β = 64, 1 soft step |
;; |---|---|---|---|---|
;; | single cleanup, unrelated items | 14.1 | 14.1 | 14.1 | 14.1 |
;; | single cleanup, 16 look-alikes | 9.4 | 9.0 | 5.6 | 9.2 |
;; | reading a vector stored beside 15 look-alikes | 3.0 | 3.0 | | 1.8 |
;;
;; So the hope for a Hopfield network is probably not a better cleanup.
;; Exact recall is the one job a lookup table already does as well as
;; anything can. The hope is a different job: a memory of what the
;; computation *did*, not only of what it stored. Which values a function
;; returned, which code paths a program took. A new situation would then be
;; completed from similar old ones. The pull towards the centre of a group
;; of look-alikes, a loss in the table above, is the same mechanism seen
;; from the other side: it is how such a memory would form a prototype and
;; generalise. That would make the interpreter a neuro-symbolic system,
;; exact where it cleans up and associative where it remembers. The CEK
;; machine below already picks its rules by similarity; a memory of its
;; past runs would be the next step. We have not built it.

;; ## The interpreter as vectors
;;
;; The evaluator above is host code. The next step moves it into vectors. A
;; classic design for interpreters, the CEK machine, keeps three things:
;; the **C**ontrol, what to evaluate now; the **E**nvironment, which
;; variable has which value; and the **K**ontinuation, a to-do list of what
;; comes after (spelled with a K by tradition, since C is taken). Here all
;; three, plus a mode, are bundled into one vector:
;; s ↦ ν(C⊗control + E⊗env + K⊗continuation + M⊗mode).
;; Its 35 transition rules sit in a memory and are picked by similarity.
;; Rule keys are *sums* of features, so the most specific matching rule
;; wins by nearest neighbour, and shorter keys act as defaults. The host
;; loop is under 50 lines and knows no Lisp. Tail calls come for free.

;; ### One step
;;
;; The machine runs in steps. Each step goes round the loop below in three
;; phases, and each phase reads some vectors and produces others. The
;; orange lines follow one real step: the moment the machine meets
;; `(if (zero? x) :zero (inc x))`.

^:kindly/hide-code
(defn cek-diagram []
  (let [blue {:fill "#eef4fb" :stroke "#2c6fbb" :stroke-width 1.5 :rx 10}
        box (fn [x y title lines example]
              (into [:g [:rect (merge blue {:x x :y y :width 350 :height 150})]
                     [:text {:x (+ x 14) :y (+ y 26) :font-size 15 :font-weight 600
                             :fill "#1b3a5c"} title]
                     [:text {:x (+ x 14) :y (+ y 136) :font-size 12.5 :fill "#b05f00"
                             :font-family "monospace"} example]]
                    (map-indexed
                     (fn [i [a b]]
                       [:text {:x (+ x 14) :y (+ y 52 (* i 20)) :font-size 13 :fill "#222"}
                        [:tspan {:font-family "monospace" :font-weight 600 :fill "#2c6fbb"} a]
                        [:tspan {:x (+ x 50)} b]])
                     lines)))
        arrow (fn [x1 y1 x2 y2 label lx ly anchor]
                [:g [:line {:x1 x1 :y1 y1 :x2 x2 :y2 y2 :stroke "#555" :stroke-width 1.6
                            :marker-end "url(#arrowhead)"}]
                 [:text {:x lx :y ly :font-size 12.5 :font-style "italic" :fill "#555"
                         :text-anchor anchor} label]])]
    (kind/hiccup
     [:svg {:viewBox "0 0 800 440" :style {:width "100%" :max-width "800px"
                                           :font-family "system-ui, sans-serif"}}
      [:defs [:marker {:id "arrowhead" :markerWidth 10 :markerHeight 8 :refX 9 :refY 4
                       :orient "auto"} [:path {:d "M0,0 L10,4 L0,8 z" :fill "#555"}]]]
      (box 16 16 "the state s: one vector"
           [["C" "control: what to evaluate now"] ["E" "environment: variables → values"]
            ["K" "continuation: the frames still to do"] ["M" "mode: EVAL, RET, APPLY or HALT"]]
           "EVAL, C = (if (zero? x) …), K = [halt]")
      (box 434 16 "1 · fetch: read the state's shape"
           [["in " "s"] ["out" "C, E, K, M unpacked into registers,"]
            ["" "and KEY, a vector naming the shape:"] ["" "mode + kind of C + head of C + frame"]]
           "KEY = EVAL + list + if + halt")
      (box 434 274 "2 · select: which rule fits?"
           [["in " "KEY"] ["how" "nearest of the 35 rule keys"]
            ["" "stored in the rule memory R"] ["out" "that rule's microprogram"]]
           "rule if (its key: EVAL + list + if)")
      (box 16 274 "3 · run the microprogram"
           [["in " "the registers C, E, K, M"] ["how" "instructions on vectors: read a"]
            ["" "field, bind, clean up, build a record"] ["out" "the next state s′"]]
           "push an if frame; C = (zero? x)")
      (arrow 366 70 434 70 "unpack" 400 60 "middle")
      (arrow 609 166 609 274 "KEY" 620 224 "start")
      (arrow 434 340 366 340 "code" 400 330 "middle")
      (arrow 191 274 191 166 "s′ is the new s" 178 224 "end")
      [:text {:x 178 :y 240 :font-size 12.5 :font-style "italic" :fill "#555"
              :text-anchor "end"} "(until mode = HALT)"]
      [:rect {:x 280 :y 202 :width 240 :height 36 :rx 18 :fill "#fff4e0"
              :stroke "#d98a00" :stroke-width 1.5}]
      [:text {:x 400 :y 225 :font-size 13.5 :font-weight 600 :fill "#7a4b00"
              :text-anchor "middle"} "host loop · under 50 lines · no Lisp"]])))

^:kindly/hide-code
(cek-diagram)

;; The key is the whole trick. It describes the state by five features:
;; the **mode**; the **kind** of C (symbol, number, list, closure, …); the
;; **head** of C, which matters when C is a list starting with a special
;; form such as `if` or `let`; C's **identity**, which matters when C is
;; one of the two primitives that re-enter evaluation, `apply` and `eval`;
;; and the type of the **frame** on top of K. Every state's key has all
;; five; a rule names only those it cares about. Each feature is a random vector bound to its
;; role, and the key is their sum. The rule memory then does what a `case`
;; statement does in an ordinary interpreter, by similarity.
;;
;; Here is that step again, decoded live from the vectors. The rule `if`
;; saves the two branches and the environment in a new frame on K, and
;; makes the test the next thing to evaluate.

^:kindly/hide-code
(kind/hidden (machine/init!))

^:kindly/hide-code
(let [reg #'vsc.machine/reg
      steps (atom [])]
  (binding [machine/*trace* (fn [step i regs]
                              (swap! steps conj
                                     {:rule ((#'vsc.machine/v :rule-names) i)
                                      :key (#'vsc.machine/decode-key (reg regs 'KEY))
                                      :mode (str (#'vsc.machine/opm-name (reg regs 'M) 0.9))
                                      :C (pr-str (vsc/decode (reg regs 'C)))
                                      :K (pr-str (mapv #(symbol (str/replace (str %) "-fr" ""))
                                                       (#'vsc.machine/frames (reg regs 'K))))}))]
    (machine/run '((fn [x] (if (zero? x) :zero (inc x))) 1)))
  (let [[before after] (drop-while #(not= 'if (:rule %)) @steps)
        k (:key before)
        show (fn [s] (kind/code (str "mode " (:mode s) "\nC    " (:C s) "\nK    " (:K s))))]
    (kind/table
     {:column-names ["" "decoded from the vectors"]
      :row-vectors [["the state" (show before)]
                    ["1 · its key" (kind/code (str/join " + " (map (fn [f] (str (name f) " " (-> (str (get k f)) (str/replace #"^k-" "") (str/replace #"-fr$" ""))))
                                                                    (filter k [:mode :kind :head :ident :frame]))))]
                    ["2 · the rule chosen" (kind/code (str (:rule before)))]
                    ["3 · the next state" (show after)]]})))

;; ### The code
;;
;; The host loop is all the host code there is: fetch, select, run, until
;; the mode is HALT. `exec!` is the interpreter of the nine instructions.
;; Nothing in it knows about Lisp.

^:kindly/hide-code
(kind/code (str/join "\n\n" (map #(clojure.repl/source-fn %)
                                 '[vsc.machine/run-state! vsc.machine/run-micro! vsc.machine/exec!])))

;; Everything Lisp-specific is data, assembled into instruction vectors when
;; the machine starts. The instructions:
;;
;; | instruction | does |
;; |---|---|
;; | `FIELD d x r s` | d ← cleanup(r ⊘ x): read field r of record x |
;; | `CLEAN d x s` | d ← cleanup(x) in memory s, e.g. to classify x by kind |
;; | `BIND d x y`, `UNBIND d x y` | d ← x ⊗ y, d ← x ⊘ y |
;; | `REC d s r₁ x₁ …` | d ← a new record ν(Σ rᵢ⊗xᵢ), stored in memory s |
;; | `MOV d x` | d ← x |
;; | `JMPEQ x y l` | jump to l if x and y are the same vector |
;; | `PRIM d f x` | d ← the primitive f (`+`, `first`, …) applied to the list x |
;; | `DEF x y` | store the global x ↦ y |
;;
;; S, C, E, K, M are the state's registers; T, A, B, N, X, Y, Z are
;; scratch. `(PACK)` builds the next state from C, E, K and M, and `(JMP l)`
;; jumps unconditionally. The fetch phase is itself a microprogram of twelve
;; instructions:

^:kindly/hide-code
(kind/code (clojure.repl/source-fn 'vsc.machine/fetch-program))

;; And the two rules of `if`. The first fires on the step above. Read
;; with C = `(if test then else)`: X ← the rest of C, `(test then else)`;
;; C ← its first element, the test; A ← the rest, `(then else)`; push a
;; frame onto K that remembers the branches A, the environment E and the
;; old K as its next N; then evaluate C. The second rule fires when the
;; test's value comes back to that frame: it pops the frame, restores the
;; environment, and evaluates one branch. The frame is gone by then, which
;; is why tail calls come for free.

^:kindly/hide-code
(kind/code (with-out-str
             (doseq [r (filter #(#{'if 'if-branch} (:name %)) machine/rules)]
               (clojure.pprint/pprint r))))


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
;; The joint worlds are lists, not vectors. A vector is linear in its
;; elements, so [1 1] + [2 2] is exactly [1 2] + [2 1]: a sum of vectors
;; remembers each position's distribution, but not which values went
;; together.
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
;; - No macros, `loop`/`recur`, floats, ordering comparisons or string
;;   operations.
;; - Records hold about D/64 entries, vectors about D/48; superpositions
;;   about D/100 worlds.

;; ## What is new here
;;
;; To our knowledge, these are new:
;;
;; - **Clojure's data types as single vectors.** Earlier vector-symbolic
;;   Lisps encode the cons cells of Lisp 1.5, and integers. Here maps
;;   (with directed entries k⊗Π(v)), sets, positional vectors, closures
;;   and environments are each one vector too, and integers inside
;;   collections are boxed against chimeras.
;; - **A catalogue of what breaks.** Seven traps that the algebra hides,
;;   each measured and fixed. One rule covers several of them: decide by
;;   the best match or relative to chance, never by a fixed cutoff, and
;;   robustness grows with √D.
;; - **Measurements for a whole language**: noise, capacity and damage,
;;   with the split between robust stored data and fragile computation.
;; - **Why a Hopfield cleanup cannot help with exact recall**: an argument
;;   that holds before any experiment, and the experiments.
;; - **An evaluator in vector memory.** Rules held in vector memory are not
;;   new: production systems in HRR (Stewart & Eliasmith, 2008) and a
;;   Turing machine with its transition table in an item memory (Kleyko et
;;   al., 2022) do it. New is a complete language evaluator run this way: a
;;   CEK machine whose 35 rules are picked by nearest key, where sums of
;;   features make the most specific rule win, with tail calls.
;; - **Superposition programming as a language feature.** Weighted
;;   superpositions are ordinary values. Destructors, lookup, arithmetic and
;;   function bodies act on all worlds at once, `if` splits them, and
;;   run-time versus call-time choice is explicit. The
;;   semantics has precedents (Lineal; probabilistic functional
;;   programming), and so does probability in vectors (Furlong & Eliasmith;
;;   Hersche et al.). We know of no language that puts them together on a
;;   vector substrate, with a memory that keeps the worlds.
;;
;; ## Related work
;;
;; Most relevant first.
;;
;; 1. Tomkins-Flanagan, E., & Kelly, M. A. (2024). Hey Pentti, we did it!:
;;    A fully vector-symbolic Lisp. MathPsych/ICCM 2024.
;;    [arXiv:2510.17889](https://arxiv.org/abs/2510.17889),
;;    [code](https://github.com/eilene-ftf/holis). Lisp 1.5's elementary
;;    functions and lambda in HRR, with a lookup-table cleanup. The
;;    starting point of this work.
;; 2. Hanley, C., Tomkins-Flanagan, E., & Kelly, M. A. (2025). Hey Pentti,
;;    we did (more of) it!: A vector-symbolic Lisp with residue arithmetic.
;;    IJCNN 2025. [arXiv:2511.08767](https://arxiv.org/abs/2511.08767).
;;    Adds integers in a residue number system, the encoding whose
;;    chimeras trap 3 describes.
;; 3. Kleyko, D., et al. (2022). Vector symbolic architectures as a
;;    computing framework for emerging hardware. *Proceedings of the IEEE*
;;    110(10). [doi:10.1109/JPROC.2022.3209104](https://doi.org/10.1109/JPROC.2022.3209104).
;;    Names "computing in superposition", and runs a Turing machine whose
;;    transition table is an item memory: the closest precedent for rules
;;    in vector memory.
;; 4. Stewart, T. C., & Eliasmith, C. (2008). Building production systems
;;    with realistic spiking neurons. CogSci 2008.
;;    [pdf](https://compneuro.uwaterloo.ca/files/Stewart.ProductionSystemsNeurons.pdf).
;;    IF–THEN rules as HRR associations, chosen by an action selector over
;;    an HRR state, like the CEK machine's rule memory.
;; 5. Arrighi, P., & Dowek, G. (2017). Lineal: A linear-algebraic
;;    λ-calculus. *Logical Methods in Computer Science* 13(1).
;;    [doi:10.23638/LMCS-13(1:8)2017](https://doi.org/10.23638/LMCS-13(1:8)2017).
;;    Terms can be linear combinations α·t + β·u, and application
;;    distributes over sums: the formal model of lifting functions over
;;    superposed values.
;; 6. Erwig, M., & Kollmansberger, S. (2006). Probabilistic functional
;;    programming in Haskell. *Journal of Functional Programming* 16(1).
;;    [doi:10.1017/S0956796805005721](https://doi.org/10.1017/S0956796805005721).
;;    Values as weighted lists of outcomes, functions mapped over them:
;;    the same semantics with explicit lists instead of one vector.
;; 7. Kiselyov, O., & Shan, C.-c. (2009). Embedded probabilistic
;;    programming. DSL 2009, LNCS 5658.
;;    [doi:10.1007/978-3-642-03034-5_17](https://doi.org/10.1007/978-3-642-03034-5_17).
;;    Hansei: weighted worlds by lazy exact enumeration and importance
;;    sampling, inside OCaml.
;; 8. Hersche, M., Zeqiri, M., Benini, L., Sebastian, A., & Rahimi, A.
;;    (2023). A neuro-vector-symbolic architecture for solving Raven's
;;    progressive matrices. *Nature Machine Intelligence* 5(4).
;;    [doi:10.1038/s42256-023-00630-8](https://doi.org/10.1038/s42256-023-00630-8).
;;    Distributions over attribute values as VSA vectors, with rules
;;    applied to the whole distribution at once.
;; 9. Furlong, P. M., & Eliasmith, C. (2022). Fractional binding in vector
;;    symbolic architectures as quasi-probability statements. CogSci 2022.
;;    [escholarship](https://escholarship.org/uc/item/33j7c63v). And (2024)
;;    Modelling neural probabilistic computation using vector symbolic
;;    architectures. *Cognitive Neurodynamics* 18(6).
;;    [doi:10.1007/s11571-023-10031-7](https://doi.org/10.1007/s11571-023-10031-7).
;;    Distributions as vectors, inference by vector operations.
;; 10. Dewulf, P., De Baets, B., & Stock, M. (2023). The hyperdimensional
;;     transform for distributional modelling, regression and
;;     classification. [arXiv:2311.08150](https://arxiv.org/abs/2311.08150).
;;     Distributions and Bayesian inference with hypervectors.
;; 11. Vaux, L. (2009). The algebraic lambda calculus. *Mathematical
;;     Structures in Computer Science* 19(5).
;;     [doi:10.1017/S0960129509990089](https://doi.org/10.1017/S0960129509990089).
;;     Lambda terms with weighted sums over a semiring.
;; 12. Ramsey, N., & Pfeffer, A. (2002). Stochastic lambda calculus and
;;     monads of probability distributions. POPL 2002.
;;     [doi:10.1145/503272.503288](https://doi.org/10.1145/503272.503288).
;;     Semantics for a language whose values are distributions.
;; 13. Hussmann, H. (1993). *Nondeterminism in Algebraic Specifications and
;;     Algebraic Programs*. Birkhäuser.
;;     [doi:10.1007/978-1-4684-6834-2](https://doi.org/10.1007/978-1-4684-6834-2).
;;     Distinguishes call-time from run-time choice.
;; 14. McCarthy, J. (1963). A basis for a mathematical theory of
;;     computation. In *Computer Programming and Formal Systems*.
;;     [doi:10.1016/S0049-237X(08)72018-4](https://doi.org/10.1016/S0049-237X(08)72018-4).
;;     The ambiguous operator `amb`.
;; 15. Tomkins-Flanagan, E., Hanley, C., & Kelly, M. A. (2025). Hey Pentti,
;;     we did it again!: Differentiable vector-symbolic types that prove
;;     polynomial termination. MathPsych/ICCM 2025.
;;     [arXiv:2510.16533](https://arxiv.org/abs/2510.16533). Doug, a typed
;;     vector-symbolic language whose types guarantee polynomial-time
;;     termination.
;; 16. Kanerva, P. (2014). Computing with 10,000-bit words. Allerton 2014.
;;     [doi:10.1109/ALLERTON.2014.7028470](https://doi.org/10.1109/ALLERTON.2014.7028470).
;;     Discusses how a Lisp could be built from hypervector operations;
;;     Kanerva has also called for an HDC-Lisp in his talks.
;; 17. Meier, C. (2023). Vector symbolic architectures in Clojure.
;;     Clojure/Conj 2023. [vsa-clj](https://github.com/gigasquid/vsa-clj).
;;     VSA computing experiments in Clojure.
;; 18. Eliasmith, C., et al. (2012). A large-scale model of the functioning
;;     brain. *Science* 338(6111).
;;     [doi:10.1126/science.1225266](https://doi.org/10.1126/science.1225266).
;;     Spaun: action rules over semantic-pointer (HRR) states at scale.
;; 19. Cotteret, M., Greatorex, H., Ziegler, M., & Chicca, E. (2024).
;;     Vector symbolic finite state machines in attractor neural networks.
;;     *Neural Computation* 36(4).
;;     [doi:10.1162/neco_a_01638](https://doi.org/10.1162/neco_a_01638).
;;     Finite-state machines run as attractor dynamics over VSA states.
;; 20. Yerxa, T., Anderson, A. G., & Weiss, E. (2018). The hyperdimensional
;;     stack machine. Cognitive Computing 2018.
;;     [pdf](https://redwood.berkeley.edu/wp-content/uploads/2021/08/Yerxa_HSM.pdf).
;;     A stack machine from hypervector operations.
;; 21. Osipov, E., Kleyko, D., & Legalov, A. (2017). Associative synthesis
;;     of finite state automata model of a controlled object with
;;     hyperdimensional computing. IECON 2017.
;;     [doi:10.1109/IECON.2017.8216554](https://doi.org/10.1109/IECON.2017.8216554).
;;     Finite-state automata in hypervectors.
;; 22. Frady, E. P., Kent, S. J., Olshausen, B. A., & Sommer, F. T. (2020).
;;     Resonator networks, 1. *Neural Computation* 32(12).
;;     [doi:10.1162/neco_a_01331](https://doi.org/10.1162/neco_a_01331).
;;     Factoring bound and superposed vectors; what exponentially many
;;     worlds would need.
;; 23. Kymn, C. J., et al. (2025). Computing with residue numbers in
;;     high-dimensional representation. *Neural Computation* 37(1).
;;     [doi:10.1162/neco_a_01723](https://doi.org/10.1162/neco_a_01723).
;;     The residue integer encoding used here.
;; 24. Ramsauer, H., et al. (2021). Hopfield networks is all you need. ICLR
;;     2021. [arXiv:2008.02217](https://arxiv.org/abs/2008.02217). Modern
;;     Hopfield networks, whose update is attention.
;; 25. Millidge, B., Salvatori, T., Song, Y., Lukasiewicz, T., & Bogacz, R.
;;     (2022). Universal Hopfield networks. ICML 2022.
;;     [PMLR 162](https://proceedings.mlr.press/v162/millidge22a.html).
;;     Associative memories as similarity, separation and projection.
;; 26. Stewart, T. C., Tang, Y., & Eliasmith, C. (2011). A biologically
;;     realistic cleanup memory: Autoassociation in spiking neurons.
;;     *Cognitive Systems Research* 12(2).
;;     [doi:10.1016/j.cogsys.2010.06.006](https://doi.org/10.1016/j.cogsys.2010.06.006).
;;     Cleanup memory built with the Neural Engineering Framework.
;; 27. Plate, T. A. (1995). Holographic reduced representations. *IEEE
;;     Transactions on Neural Networks* 6(3).
;;     [doi:10.1109/72.377968](https://doi.org/10.1109/72.377968). HRR,
;;     the algebra used here.
;; 28. Smolensky, P. (1990). Tensor product variable binding and the
;;     representation of symbolic structures in connectionist systems.
;;     *Artificial Intelligence* 46.
;;     [doi:10.1016/0004-3702(90)90007-M](https://doi.org/10.1016/0004-3702(90)90007-M).
;;     The foundational scheme for structure in vectors.
;; 29. Selinger, P. (2004). Towards a quantum programming language.
;;     *Mathematical Structures in Computer Science* 14(4).
;;     [doi:10.1017/S0960129504004256](https://doi.org/10.1017/S0960129504004256).
;;     Quantum data with classical control; its types forbid copying,
;;     which weighted worlds allow.
;; 30. Bichsel, B., Baader, M., Gehr, T., & Vechev, M. (2020). Silq: A
;;     high-level quantum language with safe uncomputation and intuitive
;;     semantics. PLDI 2020.
;;     [doi:10.1145/3385412.3386007](https://doi.org/10.1145/3385412.3386007).
;;     A high-level quantum language with automatic uncomputation.
;;
;; -----
;;
;; *Generated by Claude on Benjamin Schwerdtner's behalf.*
