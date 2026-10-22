^:kindly/hide-code
(ns holis-reproduction
  (:require
   [clojure.data.json :as json]
   [clojure.string :as str]
   [scicloj.kindly.v4.kind :as kind]))

;; # Reproducing the Traps in the Original Code
;;
;; > Generated with Claude Code, and I am not ashamed. My own blog posts of
;; > this kind are part of the training data and have sculpted its style.
;;
;; *A companion to [A Vector-Symbolic Clojure](../). The overview found five
;; traps in a vector-symbolic Lisp. This page runs each one against the
;; original authors' own code. October 2026.*
;;
;; My Clojure started from Tomkins-Flanagan and Kelly's "Hey Pentti, We
;; Did It!" (ICCM 2024). That paper built Lisp 1.5 out of holographic
;; vectors and showed that it can be done. Their follow-up with Hanley,
;; "Hey Pentti, We Did (More of) It!" (IJCNN 2025), added integers in
;; residue arithmetic. Both have public code. My five traps came from my
;; code. The fair question is which of them exist in theirs. Here I
;; answer it by running their code: I fetch it at a pinned commit, add
;; noise the way my paper does, and count. The trap numbers 1–5 are those
;; of the overview.
;;
;; Their papers are proofs of concept, and their READMEs say so. Nothing
;; below changes that. A proof of concept is where the next person starts,
;; and this page lists what that next person (me) ran into.
;;
;; If you maintain or build on either codebase, go to
;; [Errata for the authors](#errata-for-the-authors): each defect with its
;; location, a minimal program and a suggested fix.

^:kindly/hide-code
(def dir "repro/holis/")

^:kindly/hide-code
(defn load-json [f]
  (json/read-str (slurp (str dir f))))

^:kindly/hide-code
(def hc (load-json "holis_control.json"))
^:kindly/hide-code
(def he (load-json "holis_errata.json"))
^:kindly/hide-code
(def hm (load-json "holis_commute.json"))
^:kindly/hide-code
(def ht (load-json "holis_thresholds.json"))
^:kindly/hide-code
(def hcell (load-json "holis_cells.json"))
^:kindly/hide-code
(def rc (load-json "residuelisp_control.json"))
^:kindly/hide-code
(def re (load-json "residuelisp_errata.json"))
^:kindly/hide-code
(def rt (load-json "residuelisp_thresholds.json"))
^:kindly/hide-code
(def hf (load-json "holis_fixes.json"))
^:kindly/hide-code
(def rf (load-json "residuelisp_fixes.json"))

^:kindly/hide-code
(defn table
  "A table with the given column names; each row is a vector."
  [cols rows]
  (kind/table {:column-names cols :row-vectors (vec rows)}))

^:kindly/hide-code
(defn frac [row ok-key n-key]
  (str (get row ok-key) " / " (get row n-key)))

^:kindly/hide-code
(defn fmt [x]
  (cond (nil? x) "–"
        (float? x) (format "%.3f" (double x))
        (map? x) (str/join ", " (for [[k v] x] (str k " ×" v)))
        :else (str x)))

^:kindly/hide-code
(defn mean [xs] (/ (reduce + xs) (count xs)))

^:kindly/hide-code
(defn pivot
  "Rows keyed by `row-key`; one column per distinct value of `col-fn`,
  cells from `cell-fn`."
  [rows row-key col-fn cell-fn row-label]
  (let [cols (distinct (map col-fn rows))
        by-row (group-by #(get % row-key) rows)]
    (table (cons row-label (map str cols))
           (for [[r rs] (sort-by key by-row)]
             (cons (fmt r)
                   (for [c cols]
                     (if-let [x (first (filter #(= c (col-fn %)) rs))]
                       (cell-fn x)
                       "–")))))))

;; ## The two codebases
;;
;; **holis** ([eilene-ftf/holis](https://github.com/eilene-ftf/holis),
;; commit `db208d7`, AGPL-3.0) is one Jupyter notebook. It adapts Norvig's
;; lispy to holographic reduced representations (HRR) at D = 512:
;;
;; - Vectors are random Gaussian, bound by circular convolution.
;; - A cons cell is the bundle lhs⊗a + rhs⊗b + φ, but it is stored behind
;;   a fresh random *pointer* in a second memory, because, in the
;;   notebook's words, "tuples turn out to degrade rather quickly".
;;   Equal bundles share a pointer.
;; - CAR and CDR follow the pointer, unbind a role and clean up against a
;;   lookup table (argmax).
;; - The pointer memory looks up a noisy pointer by its nearest key, but
;;   only if the cosine exceeds θ = 0.2. Otherwise it returns the probe
;;   itself.
;; - The evaluator tests with a fixed floor of 0.2, and COND is the
;;   paper's saturating ⊕ with cutoffs 0.8 and 0.2.
;; - Functions work by substitution, so there are no environments.
;;   Numbers are Peano lists: 0 is NIL and n+1 is (n).
;;
;; **residuelisp** ([hanleyc01/residuelisp](https://github.com/hanleyc01/residuelisp),
;; commit `09864c7`, MIT) is the code of the second paper, a typed Python
;; package. It uses complex FHRR vectors at D = 1000:
;;
;; - Cells sit behind pointers, as in holis.
;; - Function calls bind parameters in a small memory keyed by symbol.
;; - Integers are Kymn et al.'s residue code over the moduli 3, 5, 7, 11
;;   (range 0..1154), plus a tag: int(x) = ζ(x) + int.
;; - It keeps holis's numbers: θ = 0.2 in the pointer memory and a 0.2
;;   floor in its approximate equality. Its similarity is not normalised,
;;   though.
;;
;; ## What I ran
;;
;; Nothing of theirs is copied into my repository. A small loader
;; fetches each codebase at its pinned commit when a script runs. For
;; holis it executes the notebook's code cells as a module. The scripts in
;; [`repro/holis/`](https://github.com/benjamin-asdf/vector-symbolic-clj/tree/main/repro/holis)
;; then:
;;
;; - add **probe noise** exactly as in my paper: before every cleanup,
;;   the probe p becomes p + σ·|p|/√D · N(0, I). Cleanups are the item
;;   memory, the function memory and the pointer memory;
;; - change at most one thing as a **control**, and say which;
;; - count over seeds, and write every number on this page to a JSON file
;;   that the page reads.
;;
;; First, their code works. Noise-free, holis computes these at D = 512
;; and 2048 over 5 seeds each:

^:kindly/hide-code
(table ["program" "D" "correct"]
       (for [r (get hc "small")]
         [(kind/code (get r "program")) (get r "D") (frac r "correct" "runs")]))

;; One program misses a seed. I return to it under E1 below. The
;; authors' own fizzbuzz helpers, verbatim from the notebook, also give
;; the right answers. They are slow, which is why the noise sweeps below
;; use smaller programs:

^:kindly/hide-code
(table ["program" "result" "expected" "seconds"]
       (for [r (get hc "fizzbuzz")]
         [(get r "program") (kind/code (get r "result")) (kind/code (get r "expected"))
          (get r "seconds")]))

;; residuelisp, noise-free at D = 1000 over 5 seeds:

^:kindly/hide-code
(table ["program" "expected" "correct"]
       (for [r (get rc "rows")]
         [(kind/code (get r "program")) (kind/code (get r "expected")) (frac r "correct" "runs")]))

;; So both interpreters do what their papers show. What follows asks for
;; more than the papers claim.

;; ## Trap 1: Superposed cells, which the paper has and the code does not
;;
;; **Applies?** To the paper's equation (2), cons(a, b) = ν(L⊗a + R⊗b + φ),
;; yes. To the code, no: both codebases put each cell behind a random
;; pointer, as my fix does. The holis notebook gives the reason in a
;; comment.
;;
;; To see the paper's cell inside their code, I added a *paper mode*.
;; It is two attributes on their pointer memory: a cell is stored as its
;; own normalised bundle, and "following" it returns it unchanged. The
;; evaluator, CAR, CDR and the reader stay theirs. At D = 2048, Peano
;; numerals n and n+1 then look alike. Pointer cells do not:

^:kindly/hide-code
(table ["mode" "mean cos(n, n+1)" "mean cos(n, n+2)" "seeds"]
       (for [[mode rs] (group-by #(get % "mode") (get hcell "sibling"))]
         [mode (fmt (mean (map #(get % "cos(n,n+1)_mean") rs)))
          (fmt (mean (map #(get % "cos(n,n+2)_mean") rs)))
          (count rs)]))

;; **Reproduced?** Yes, noise-free, and in a way I did not expect. Their
;; evaluator decides that an expression "evaluates to itself", and stops,
;; when the result is similar to the input: `(ee == e) > 0.2`. With
;; pointers, a form and its result are unrelated. With superposed cells,
;; a form and its first evaluation step share terms, and their cosine is
;; well above 0.2:

^:kindly/hide-code
(table ["mode" "program" "correct" "results" "mean cos(form, first step)"]
       (for [r (get hcell "program")]
         [(get r "mode") (kind/code (get r "program")) (frac r "correct" "runs")
          (fmt (get r "results")) (fmt (get r "cos(form, ev(form))_mean"))]))

;; In paper mode the LAMBDA program stops after the substitution and
;; returns the body unevaluated. This is trap 1's lesson in its plainest
;; form: similarity of content is not identity.
;;
;; Under noise the picture is less clean. Here numerals 1 to 8 are built,
;; and then numeral 4 is read back with their reader:

^:kindly/hide-code
(pivot (get hcell "read") "sigma" #(get % "mode") #(frac % "correct" "runs") "σ")

;; With their θ = 0.2, pointer cells fail first, because of that
;; threshold (trap 3 below). With θ set relative to chance, 5/√D, the
;; comparison is between encodings alone. Then the curves cross: pointer
;; cells read more numerals right at σ = 5 and 6, superposed cells at
;; σ = 8. So this test does not show the loss of noise tolerance I
;; measured in my own code (trap 1 on the overview). One difference: a
;; superposed cell needs no pointer lookup, so each read makes fewer noisy
;; cleanups. Plain argmax (θ = −1) breaks the pointer reads even at σ = 0.
;; Their atom test needs a threshold, which I come back to in trap 3.
;;
;; **Verdict.** The paper's cell causes a noise-free failure in their
;; code. The code's own cell does not have that problem. The extra noise
;; sensitivity did not reproduce in my read test.

;; ## Trap 2: Binding commutes
;;
;; **Applies?** Mostly no. Neither codebase stores an unlabelled pair:
;; cells carry the roles lhs and rhs. holis has no environments, and
;; residuelisp's are a keyed memory, not a sum of var⊗value. As a
;; control, (x . y) and (y . x) read back correctly every time.
;;
;; There is one opening. In holis the roles are also ordinary atoms:
;; `vsa_env` enters `lhs`, `rhs` and `φ` in the lexicon. A program that
;; mentions the symbol `lhs` therefore gets the role vector itself. The
;; cell (x . lhs) then contains lhs⊗x + rhs⊗lhs. Unbinding lhs finds x
;; and rhs at equal strength, because rhs⊗lhs = lhs⊗rhs.
;;
;; **Reproduced?** Yes, through their parser, evaluator and reader, 40
;; seeds per row:

^:kindly/hide-code
(table ["program" "D" "expected" "correct" "results"]
       (for [r (get hm "rows")]
         [(kind/code (get r "program")) (get r "D") (get r "expected")
          (frac r "correct" "runs") (fmt (get r "results"))]))

;; About half the reads give the wrong atom at every D. It is a tie, not
;; noise, so a bigger D does not help. residuelisp names its roles
;; `__lhs` and `__rhs`. Its lexer starts identifiers with a letter, so a
;; program cannot name them (I read this in their lexer and did not run
;; it).

;; ## Trap 3: Fixed thresholds cap robustness
;;
;; **Applies?** To holis, yes. Its pointer memory accepts the nearest key
;; only above cosine θ = 0.2. A probe with noise σ has cosine
;; 1/√(1 + σ²) to its key. That falls below 0.2 at σ = √24 ≈ 4.9,
;; whatever D is.
;;
;; **Reproduced?** Yes. The control is the same code with θ = 5/√D,
;; a threshold relative to chance, as my paper prescribes. It is one
;; attribute set on their object. Plain argmax (θ = −1) would not do:
;; their atom test needs the threshold to tell an atom, which matches no
;; pointer, from a cell. With θ = −1 even noise-free reads fail. For
;; D = 512, 5/√D = 0.22 is close to their 0.2. First, single CAR lookups
;; on 25 stored 4-lists, 200 lookups per cell. The cells give correct
;; lookups, and in parentheses how often θ made the memory return the
;; probe itself:

^:kindly/hide-code
(pivot (get ht "lookup") "sigma"
       #(str "D=" (get % "D") ", " (get % "mode"))
       #(str (get % "correct") " (" (get % "theta_fallbacks") ")")
       "σ")

;; With their θ, the lookups collapse between σ = 5 and 5.5 at D = 2048
;; and at D = 8192 alike, just past the predicted 4.9, and the fallback
;; count shows why. With the chance floor, D = 8192 still gets 200 of 200
;; at σ = 8. At D = 512 the two rows agree, because there the item memory
;; gives out first, and 5/√512 is close to 0.2 anyway.
;;
;; The same holds one level up. Here is a 4-list read back with their
;; reader, 20 seeds per cell:

^:kindly/hide-code
(pivot (get ht "read") "sigma"
       #(str "D=" (get % "D") ", " (get % "mode"))
       #(frac % "correct" "runs")
       "σ")

;; And a whole program, `((LAMBDA (x) (CONS x (b c))) a)`, 10 seeds per
;; cell. Its noise-free row already misses some seeds (E1 below):

^:kindly/hide-code
(pivot (get ht "program") "sigma"
       #(str "D=" (get % "D") ", " (get % "mode"))
       #(frac % "correct" "runs")
       "σ")

;; Their θ stops the program at σ = 5 for D = 2048 and 8192 alike. With
;; the chance floor, D = 8192 still gets 8 of 10 at σ = 5 and 6. That is
;; as many as at σ = 4, and more than the 7 of 10 at σ = 0, which E1
;; costs.
;;
;; **residuelisp** keeps the 0.2 but compares it with an unnormalised
;; similarity. A noisy pointer keeps similarity about 1 to its own key at
;; any σ. What grows is the spread of chance matches, about
;; √((1 + σ²)/(2D)). So the cutoff should give way at a σ that grows like
;; √D. Here is the two-lambda program, 10 seeds per cell:

^:kindly/hide-code
(pivot (get rt "rows") "sigma" #(str "D=" (get % "D")) #(frac % "correct" "runs") "σ")

;; The last σ with all 10 runs right is 1 at D = 500, 3 at D = 2000 and 6
;; at D = 8000. It roughly doubles for each 4× in D, as √D predicts. So in
;; residuelisp the fixed numbers do not set a cap independent of D.
;; Its failures are loud: every failing run raised an exception (mostly
;; `InterpreterError`) or hit my cleanup budget, and none returned a
;; wrong value (raw table in the appendix). A fixed floor has a
;; different cost there, in trap 4.

;; ## Trap 4: A shared field (the integer tag)
;;
;; **Applies?** Not to holis, which has no records. It does apply to
;; residuelisp, in a form I did not foresee. Every integer carries the
;; same tag, int(x) = ζ(x) + int, which is the paper's equation (11).
;; Atoms count as `eq?` when their unnormalised similarity exceeds 0.2,
;; and the tag alone contributes 1. So any two integers are equal.
;;
;; **Reproduced?** Yes, noise-free at D = 1000, in 5 of 5 seeds (rows R1
;; below). The similarities behind it:

^:kindly/hide-code
(let [rows (get re "tag_similarity")
      ks ["sim(3,3)" "sim(3,5)" "sim(3,1000)" "sim(a,b)" "cos(3,5)" "cos(3,1000)"]]
  (table (cons "seeds" ks)
         [(cons (count rows) (for [k ks] (fmt (mean (map #(get % k) rows)))))]))

;; Their own tests compare an arithmetic result with the expected integer
;; through this equality. So those assertions pass whatever integer the
;; arithmetic returns. In their test's pattern, the result of `(+ 5 5)` "equals" 10,
;; 7, 0 and 999:

^:kindly/hide-code
(table ["seed" "program" "equals(result, n) for n = 10, 7, 0, 999"]
       (for [r (get re "test_oracle")]
         [(get r "seed") (kind/code (get r "program"))
          (fmt (get r "equals(result, encode(n))"))]))

;; ## Trap 5: A sum of tuples
;;
;; **Applies?** No. Neither codebase builds superpositions of values, and
;; neither has positional vectors, so there are no joint worlds to lose.

;; ## Further errata
;;
;; Running the code turned up problems that my list does not cover. All
;; are noise-free.
;;
;; **E1 (holis). CAR of an atom.** To decide "is this head a lambda
;; expression?", their evaluator compares car(head) with LAMBDA. It does
;; this even when head is an atom, and an atom has no CAR. car() then
;; cleans up a meaningless probe and returns some stored item. When that
;; item happens to match, the evaluator takes the wrong branch. Here is
;; `((LAMBDA (x) (CONS x (b c))) a)` over 100 seeds. The control makes
;; car() answer NIL for anything that is not a stored pointer, and changes
;; nothing else:

^:kindly/hide-code
(table ["D" "variant" "wrong"]
       (for [r (get he "car_of_atom")]
         [(get r "D") (get r "variant") (frac r "wrong" "runs")]))

;; Some of the wrong results:

^:kindly/hide-code
(table ["D" "seed" "result"]
       (for [r (get he "car_of_atom_examples")]
         [(get r "D") (get r "seed") (kind/code (get r "result"))]))

;; What car(a) returns for the atom `a`, after the run, over the 100
;; seeds. It is spread over the whole vocabulary:

^:kindly/hide-code
(table ["D" "car(a) → item ×seeds"]
       (for [[d m] (get he "car_of_a_spread")]
         [d (str/join ", " (for [[k v] (sort-by (comp - val) m)] (str k " ×" v)))]))

;; **E2, E3 (holis).** CAR does not evaluate its argument (nor does CDR;
;; CONS does not either, but their evaluator hides that, see the errata),
;; and LABEL is
;; dispatched but raises. Their paper says it leaves LABEL out, so E3 is a
;; leftover, not a broken claim:

^:kindly/hide-code
(table ["program" "D" "results over seeds"]
       (for [r (get he "other")]
         [(kind/code (get r "program")) (get r "D") (fmt (get r "results"))]))

;; **E4 (holis). The item memory grows quadratically.**
;; `SimpleCleanUp.memorize` appends 100 empty rows once it is full, but
;; it never raises its capacity. So past 100 items, every new item appends
;; another 100 rows. An empty row scores 0, so it can win a lookup only
;; when every stored row scores below 0. Mostly it costs memory and time.
;; It is why the fizzbuzz calls above take minutes, and why my
;; noise sweep at D = 8192 needed smaller memories to stay under 8 GB.
;; residuelisp's `CleanupMemory` does raise its capacity.

^:kindly/hide-code
(table ["items stored" "matrix rows" "MB at D = 512"]
       (for [r (get he "memory_growth")]
         [(get r "items stored") (get r "matrix rows") (get r "MB at D=512")]))

^:kindly/hide-code
(let [m (get he "memory_after_fizzbuzz_defines")]
  (table ["after their fizzbuzz DEFINEs, D = 512" "items stored" "matrix rows"]
         [["seed 0" (get m "items stored") (get m "matrix rows")]]))

;; **R2–R5 (residuelisp).**
;;
;; - R2: `and` tests its left argument twice and never the right one.
;;   Structural `eq?` on lists joins car and cdr with `and`, so it
;;   compares only the cars.
;; - R3: `if` does not evaluate its condition. The condition is the
;;   unevaluated form, a pointer, which is neither `nil` nor `#f`.
;; - R4: the decoder searches 0..199, while the moduli cover 0..1154.
;; - R5: see the lexer table below.
;;
;; D = 1000, 5 seeds:

^:kindly/hide-code
(table ["erratum" "program" "expected" "results"]
       (for [r (get re "programs")]
         [(get r "erratum") (kind/code (get r "program")) (get r "expected")
          (fmt (get r "results"))]))

;; R5: the lexer drops a `)` that directly follows an integer, so nested
;; arithmetic parses wrongly. My residuelisp programs put a space there.

^:kindly/hide-code
(table ["source" "tokens" "result (seed 0)"]
       (for [r (get re "lexer")]
         [(kind/code (get r "source")) (kind/code (get r "tokens"))
          (or (get r "result_seed0") "–")]))

;; ## Errata for the authors
;;
;; This section is for Tomkins-Flanagan, Kelly and Hanley, and for anyone
;; who builds on their code. Each item says where the defect is, shows a
;; minimal program, gives the cause in one sentence and suggests a small
;; fix. Line numbers are for residuelisp at `09864c7`. holis is one
;; notebook, so for it I name the cell and the function, at `db208d7`.
;; Most items are slips that a proof of concept is allowed to have. They
;; are listed so they are quick to fix.
;;
;; Every fix below is **tested**. My harness applies it as a runtime
;; patch to their code and runs the erratum's minimal programs. Every fix
;; except trap 3's then also runs the noise-free control programs from the
;; start of this page again (`holis_fixes.py`, `residuelisp_fixes.py`;
;; E1's own control is in `holis_errata.py`). The trap 3 fix is tested
;; only in `holis_thresholds.py`: lookups, reads and the LAMBDA program. Where a suggested fix
;; turned out wrong or incomplete, I say so. The fix shown is always the
;; one I ran.
;;
;; The order is by harm: wrong answers without any noise first, then
;; robustness under noise, then memory and leftovers. The counts below
;; come from the same JSON files as the rest of the page:

^:kindly/hide-code
(defn by
  "The first row of `rows` whose value at `k` is `v`."
  [rows k v]
  (first (filter #(= v (get % k)) rows)))

^:kindly/hide-code
(defn wrong-of
  "\"wrong / runs\" for a result row with \"correct\" and \"runs\"."
  [r]
  (str (- (get r "runs") (get r "correct")) " / " (get r "runs")))

^:kindly/hide-code
(defn cell
  "\"right / runs: results\" for a result row, or – if there is none."
  [r]
  (if r (str (frac r "correct" "runs") " right: " (fmt (get r "results"))) "–"))

^:kindly/hide-code
(defn before-after
  "The rows of one erratum side by side: one line per program (and D),
  one column per variant. `vkey` names the variant field of a row."
  [rows vkey variants]
  (let [k (juxt #(get % "program") #(get % "D"))
        has-d (some #(get % "D") rows)
        pick (fn [kk v] (first (filter #(and (= kk (k %)) (= v (get % vkey))) rows)))]
    (table (concat ["program"] (when has-d ["D"]) ["should give"] variants)
           (for [kk (distinct (map k rows))]
             (concat [(kind/code (first kk))] (when has-d [(second kk)])
                     [(get (some #(pick kk %) variants) "expected")]
                     (for [v variants] (cell (pick kk v))))))))

^:kindly/hide-code
(defn rf-errata
  "residuelisp_fixes rows for one erratum."
  [e]
  (filter #(= e (get % "erratum")) (get rf "errata")))

^:kindly/hide-code
(defn label
  "A variant name from the JSON files as shown on this page: the data
  calls the trap 2 fix \"4.2\"."
  [v]
  (str/replace v "4.2" "trap 2"))

^:kindly/hide-code
(defn regression-table
  "The control programs with each fix applied."
  [regs vkey]
  (table ["fix" "programs right in every run" "runs right"]
         (for [r regs]
           [(label (get r vkey))
            (if-let [n (get r "all_seeds_correct")]
              (str n " / " (get r "programs"))
              (str (count (filter #(= (get % "correct") (get % "runs")) (get r "rows")))
                   " / " (count (get r "rows")) " (program × D)"))
            (frac r "correct" "runs")])))

^:kindly/hide-code
(let [rp (fn [p] (by (get re "programs") "program" p))
      fx (fn [e p] (by (filter #(not= "theirs" (get % "fix")) (rf-errata e)) "program" p))
      hv (fn [k v p] (first (filter #(and (= v (get % "variant")) (= p (get % "program"))) (get hf k))))
      at-d (fn [rs] (str/join ", " (for [r rs] (str (wrong-of r) " at D = " (get r "D")))))
      row (fn [id code r f]
            [id code (kind/code (get r "program")) (get r "expected")
             (fmt (get r "results")) (wrong-of r) (wrong-of f)])
      e1 (filter #(= "theirs" (get % "variant")) (get he "car_of_atom"))
      e1f (filter #(not= "theirs" (get % "variant")) (get he "car_of_atom"))
      e1x (by (get he "car_of_atom_examples") "D" 2048)
      r5 (by (get re "lexer") "source" "(+ (+ 1 2) 3)")
      r5ok (by (get re "lexer") "source" "(+ (+ 1 2 ) 3 )")
      c42 (filter #(= "(CAR (x . lhs))" (get % "program")) (get hm "rows"))
      c42f (filter #(and (= "4.2" (get % "variant")) (= "(CAR (x . lhs))" (get % "program")))
                   (get hf "roles"))
      e2 (by (get he "other") "note" "Lisp 1.5 gives b")
      e2f (filter #(and (= "E2 + unwrap QUOTE" (get % "variant"))
                        (= "(CAR (CDR (a b c)))" (get % "program")))
                  (get hf "e2"))
      e3 (by (get he "other") "note" "LABEL")
      e3f (hv "e3" "E3 removed" "(LABEL f (LAMBDA (x) x))")
      lk (fn [mode] (first (filter #(and (= 8192 (get % "D")) (= 5.5 (get % "sigma"))
                                         (= mode (get % "mode")))
                                   (get ht "lookup"))))
      th (lk "theirs (theta=0.2)")
      ch (lk "chance floor (theta=5/sqrt(D))")
      g800 (by (get he "memory_growth") "items stored" 800)
      g800f (first (filter #(and (= "E4" (get % "variant")) (= 800 (get % "items stored")))
                           (get hf "e4_growth")))]
  (table ["item" "code" "program" "should give" "gives (×runs)" "wrong" "wrong with the fix"]
         [(row "R1 (trap 4)" "residuelisp" (rp "(eq? 3 5 )") (fx "R1" "(eq? 3 5 )"))
          (row "R2" "residuelisp" (rp "(and #t #f)") (fx "R2" "(and #t #f)"))
          (row "R2" "residuelisp" (rp "(eq? (cons a (cons b nil)) (cons a (cons c nil)))")
               (fx "R2" "(eq? (cons a (cons b nil)) (cons a (cons c nil)))"))
          (row "R3" "residuelisp" (rp "(if (eq? a b) yes no)") (fx "R3" "(if (eq? a b) yes no)"))
          ["E1" "holis" (kind/code (get-in he ["meta" "settings" "program"])) "(a b c)"
           (str "varies, e.g. " (get e1x "result") " (D = 2048, seed " (get e1x "seed") ")")
           (str/join ", " (for [r e1] (str (get r "wrong") " / " (get r "runs")
                                           " at D = " (get r "D"))))
           (str/join ", " (for [r e1f] (str (get r "wrong") " / " (get r "runs")
                                            " at D = " (get r "D"))))]
          (row "R4" "residuelisp" (rp "(+ 150 100 )") (fx "R4" "(+ 150 100 )"))
          ["R5" "residuelisp" (kind/code (get r5 "source")) (get r5ok "result_seed0")
           (str (get r5 "result_seed0")) "seed 0 (lexer, no randomness)"
           (wrong-of (fx "R5" "(+ (+ 1 2) 3)"))]
          ["trap 2" "holis" (kind/code "(CAR (x . lhs))") "x"
           (str/join "; " (for [r c42] (str "D = " (get r "D") ": " (fmt (get r "results")))))
           (at-d c42) (at-d c42f)]
          ["E2" "holis" (kind/code (get e2 "program")) "b" (fmt (get e2 "results"))
           (str (- (get e2 "runs") (get-in e2 ["results" "[\"b\"]"] 0)) " / " (get e2 "runs"))
           (at-d e2f)]
          ["trap 3" "holis" "CAR lookup on a stored 4-list, σ = 5.5, D = 8192" "the stored head"
           (str (get th "correct") " right with θ = 0.2; " (get ch "correct")
                " right with θ = 5/√D")
           (str (- (get th "trials") (get th "correct")) " / " (get th "trials"))
           (str (- (get ch "trials") (get ch "correct")) " / " (get ch "trials"))]
          ["E4" "holis" "store 800 items in SimpleCleanUp" "about 800 rows"
           (str (get g800 "matrix rows") " rows, " (get g800 "MB at D=512") " MB at D = 512") "–"
           (str (get g800f "matrix rows") " rows")]
          ["E3" "holis" (kind/code (get e3 "program")) "an error that names LABEL, or no LABEL"
           (fmt (get e3 "results")) (str (get e3 "runs") " / " (get e3 "runs"))
           (str (reduce + (for [[k n] (get e3f "results") :when (str/includes? k "Error")] n))
                " / " (get e3f "runs") " (LABEL removed)")]]))

;; With each fix applied, the control programs from the start of the page
;; still give the right answers. holis, 10 programs at D = 512 and 2048,
;; 5 seeds each. The one program that misses a seed in most rows is the
;; LAMBDA program of E1; only the rows with an E1 fix get it right:

^:kindly/hide-code
(regression-table (get hf "regression") "variant")

;; residuelisp, its 15 control programs over 5 seeds:

^:kindly/hide-code
(regression-table (get rf "regression") "fix")

;; The holis control programs have no DEFINE, so I also ran their
;; fizzbuzz helpers: their DEFINEs, then the five calls in one
;; environment, D = 512, 5 seeds, each call stopped after 60 s. All these
;; runs use the E4 fix, which changes no answer and turns minutes into
;; seconds (E4 below). Runs right per call:

^:kindly/hide-code
(let [rows (get hf "fizzbuzz")
      vs (distinct (map #(get % "variant") rows))]
  (table (cons "call" vs)
         (for [p (distinct (map #(get % "program") rows))]
           (cons p (for [v vs :let [rs (filter #(and (= p (get % "program")) (= v (get % "variant")))
                                               rows)]]
                     (str (count (filter #(get % "correct") rs)) " / " (count rs)))))))

;; What the wrong runs gave:

^:kindly/hide-code
(let [short (fn [x] (if (> (count x) 70) (str (subs x 0 70) " …") x))]
  (table ["fix" "call" "results of the wrong runs"]
         (for [[[v p] rs] (group-by (juxt #(get % "variant") #(get % "program"))
                                    (remove #(get % "correct") (get hf "fizzbuzz")))]
           [v p (str/join "; " (for [[x n] (frequencies (map #(short (get % "result")) rs))]
                                 (str x " ×" n)))])))

;; ### Wrong answers without noise
;;
;; **R1 (residuelisp; my trap 4). Any two integers are `eq?`.**
;;
;; - *Where.* `src/language/interpreter.py`, `equals`, lines 320–362; the
;;   atom comparison is line 331. The tag is added in
;;   `src/language/encoding.py`, `encode_rhc_integer`, lines 322–348.
;; - *Symptom.* `(eq? 3 5 )` and `(eq? 3 1000 )` give `#t` (table above).
;;   The trap 4 table shows the same for their test pattern: the result of
;;   `(+ 5 5)` "equals" 10, 7, 0 and 999. So `test_rhc_add` and
;;   `test_rhc_sub` in `src/tests/test_interpret.py` (lines 302–335) pass
;;   whatever the arithmetic returns.
;; - *Cause.* Every integer is ζ(x) + `__int`, and the shared tag alone
;;   gives a similarity of about 1, far above the 0.2 floor.
;; - *Fix.* Compare integers without their tag, as `rhc_add` already
;;   does. Normalising the similarity is not enough: the cosine of 3 and
;;   5 is still about 0.5 (trap 4 table). In the "both atomic" branch of
;;   `equals`, before line 331:
;;
;; ```python
;; tag = enc_env.codebook["__int"]
;; if is_true(check_int(lhs, enc_env), enc_env) and is_true(check_int(rhs, enc_env), enc_env):
;;     lhs, rhs = lhs - tag, rhs - tag
;; ```
;;
;;   In the tests, compare the decoded value instead. Then a wrong sum
;;   fails:
;;
;; ```python
;; assert decode(value, enc_env, eval_env) == "3"
;; ```
;;
;; - *Status.* **Tested**, as shown. Before and after, D = 1000, 5 seeds:

^:kindly/hide-code
(before-after (rf-errata "R1") "fix" ["theirs" "R1"])

;;   Their test pattern now tells a right sum from a wrong one. The result
;;   of `(+ 5 5 )` compared with 10, 7, 0 and 999, through `equals` and
;;   through the decoded value:

^:kindly/hide-code
(table ["fix" "seed" "equals(result, n) for n = 10, 7, 0, 999" "decode(result)"]
       (for [r (get rf "test_oracle")]
         [(get r "fix") (get r "seed") (fmt (get r "equals(result, encode(n))"))
          (get r "decode(result)")]))

;;   And their own `test_rhc_add` and `test_rhc_sub` sources, checked
;;   against the right answer 3 and a wrong answer 4, counted over the
;;   seeds:

^:kindly/hide-code
(table ["source" "fix" "assert passes for 3" "assert passes for 4" "decode gives 3"]
       (for [[[src fix] rs] (group-by (juxt #(get % "source") #(get % "fix")) (get rf "their_tests"))
             :let [n (count rs) c (fn [pred] (str (count (filter pred rs)) " / " n))]]
         [(kind/code src) fix (c #(get % "equals 3")) (c #(get % "equals 4"))
          (c #(= "3" (get % "decode")))]))

;;   With the fix their assertion fails for 4, as it should. The decoded
;;   value was right all along. The control programs still pass (the
;;   regression table above).
;;
;; **R2 (residuelisp). `and` never looks at its second argument.**
;;
;; - *Where.* `src/language/interpreter.py`, `_and_`, lines 286–289.
;;   `equals` uses it at line 360 to join the car and cdr comparisons.
;; - *Symptom.* `(and #t #f)` gives `#t`. Structural `eq?` compares only
;;   the cars, so `(a b)` and `(a c)` are `eq?` (table above).
;; - *Cause.* The second clause tests `lhs` again instead of `rhs`.
;; - *Fix.* Line 288:
;;
;; ```python
;;         is_nil(rhs, enc_env) or is_false(rhs, enc_env)
;; ```
;;
;;   Their `test_and` and `test_and_comp` only use true arguments, so a
;;   test of `(and #t #f)` would have caught it.
;; - *Status.* **Tested**, as shown. Structural `eq?` now compares the
;;   cdrs too, and the control programs still pass:

^:kindly/hide-code
(before-after (rf-errata "R2") "fix" ["theirs" "R2"])

;;
;; **R3 (residuelisp). `if` does not evaluate its condition.**
;;
;; - *Where.* `src/language/interpreter.py`, `if_`, lines 427–461; the
;;   condition is taken at line 447 and tested at line 458.
;; - *Symptom.* `(if (eq? a b) yes no)` gives `yes`. Any compound
;;   condition takes the consequent. A literal `#f` works, which is why
;;   `test_if_consequent` and `test_if_alternate` pass.
;; - *Cause.* The condition is the unevaluated form, a pointer, and a
;;   pointer is neither `nil` nor `#f`.
;; - *Fix.* Line 447:
;;
;; ```python
;;     condition = evaluate(car(rand, enc_env, eval_env), enc_env, eval_env)
;; ```
;;
;; - *Status.* **Tested**, as shown. The literal conditions give the same
;;   answers as before, and the control programs still pass:

^:kindly/hide-code
(before-after (rf-errata "R3") "fix" ["theirs" "R3"])

;;
;; **E1 (holis). CAR of an atom returns a random item.**
;;
;; - *Where.* Code cell 2 ("Lisp environment definitions"), `vsa_env`,
;;   the inner `car`. `ev` calls it on atoms in two tests: "is head a
;;   lambda expression?" (`car(head)`) and "is head a defined function?"
;;   (`car(echo)`).
;; - *Symptom.* `((LAMBDA (x) (CONS x (b c))) a)` gives a wrong list in
;;   some seeds (table above, and the E1 tables under "Further errata").
;; - *Cause.* For a probe that matches no pointer, `SimpleAssoc.recall`
;;   returns the probe itself, so `car` unbinds `lhs` from an atom and
;;   cleans up noise, which lands on an arbitrary stored item.
;; - *Fix.* Let `car` answer NIL for anything that is not a stored
;;   pointer. residuelisp's `car` raises in this case (line 226), which
;;   is also fine.
;;
;; ```python
;; def car(c, ind=''):
;;     if c not in assoc.A: return env['NIL']
;;     return mem.recall(assoc.recall(c) * ~env["lhs"])
;; ```
;;
;; - *Status.* **Tested, and incomplete.** On the LAMBDA program, wrong
;;   results go to 0 of 100 at D = 512 and 2048 (E1 table), and the
;;   control programs all pass (regression table above). But it breaks
;;   every DEFINE'd function (the fizzbuzz table above, column `E1 + E4`):
;;   their calls come back unevaluated. Their function memory holds cell
;;   bundles, not pointers, and `ev` finds a function by `car` of such a
;;   bundle, which the guard now answers with NIL. Admitting stored cells
;;   too (`c in assoc.A or c in assoc.A.reverse`) brings DEFINE back,
;;   but in one of the 5 seeds four of the five calls then run past 60 s
;;   (column
;;   `E1 admitting cells + E4`). So I have no fix for E1 that I can
;;   recommend. The cause is that their code uses one `car` both on
;;   pointers and on bundles. A fix would give the function memory its
;;   own lookup, and let `SimpleAssoc.recall` report a miss instead of
;;   returning the probe.
;;
;; **R4 (residuelisp). The decoder stops at 199.**
;;
;; - *Where.* `src/language/interpreter.py`, `decode_rhc`, lines
;;   1253–1271; the range is line 1265.
;; - *Symptom.* `(+ 150 100 )` gives a different wrong number in each
;;   seed (table above). So do `(+ 199 1 )` and `(- 3 5 )`.
;; - *Cause.* The argmax runs over the codes of 0..199 (plus literals
;;   seen), while the moduli 3, 5, 7, 11 cover 0..1154.
;; - *Fix.* Lines 1265–1267. `RHC.moduli` is the moduli the encoder
;;   used; `enc_env.moduli` is empty by default. `RHC.z_ms` are the roots
;;   the encoder used, so each code is built as `RHC.encode` builds it:
;;
;; ```python
;;     for i in range(int(np.prod(RHC.moduli))):   # 0..1154
;;         if i not in codebook:
;;             codebook[i] = np.prod([z_m ** i for z_m in RHC.z_ms], axis=0)
;; ```
;;
;;   Results then wrap modulo 1155, so `(- 3 5 )` decodes as 1153. Map
;;   the upper half to negatives if that is wanted.
;; - *Status.* **Tested**, as shown. My first version called
;;   `RHC.encode(enc_env.dim, i, RHC.moduli).data` in the loop. It gave
;;   the same answers, but each call builds an `RHC` object whose
;;   constructor recomputes the four roots, so one decode took about 20 s
;;   at D = 1000. The version above takes well under a second:

^:kindly/hide-code
(before-after (rf-errata "R4") "fix" ["theirs" "R4"])

;;
;; **R5 (residuelisp). The lexer drops a `)` right after a number.**
;;
;; - *Where.* `src/syntax/parser.py`, `_digit`, lines 121–127.
;; - *Symptom.* `(+ (+ 1 2) 3)` lexes as `( + ( + 1 2 3` and fails. With
;;   a space before each `)` it gives 6 (table above, and the lexer table
;;   under "Further errata").
;; - *Cause.* The loop calls `_next` to look at the character after the
;;   digits, so that character is consumed and lost.
;; - *Fix.* Peek first, as `_key_or_ident` does:
;;
;; ```python
;; def _digit(d, conts):
;;     ds = [d]
;;     while (curr := _peek(conts)) is not None and curr.isdigit():
;;         ds.append(_next(conts))
;;     return Token("".join(ds), TokenKind.Int)
;; ```
;;
;; - *Status.* **Tested**, as shown. The tokens:

^:kindly/hide-code
(table ["fix" "source" "tokens"]
       (for [r (get rf "lexer")]
         [(get r "fix") (kind/code (get r "source")) (kind/code (get r "tokens"))]))

^:kindly/hide-code
(before-after (rf-errata "R5") "fix" ["theirs" "R5"])

^:kindly/hide-code
(table ["fix" "control programs without the space before ), right in every run"]
       (for [r (get rf "regression_unspaced")]
         [(get r "fix") (str (get r "all_seeds_correct") " / " (get r "programs"))]))

;;   The control programs pass with the fix, with and without the space
;;   before each `)`. They pass without the space even unfixed: a `)`
;;   lost at the very end is forgiven by the parser. Only a lost `)`
;;   inside the form breaks a program. That is also why their tests,
;;   which write `(+ 1 2)`, never saw it.
;;
;; With all five fixes applied together, the programs of R1 to R5 give:

^:kindly/hide-code
(let [rs (rf-errata "all fixes")]
  (table ["programs" "right in every run" "runs right"]
         [[(count rs) (count (filter #(= (get % "correct") (get % "runs")) rs))
           (str (reduce + (map #(get % "correct") rs)) " / " (reduce + (map #(get % "runs") rs)))]]))

;; And programs that need two fixes at once:

^:kindly/hide-code
(before-after (rf-errata "combined") "fix" ["theirs" "all"])

;;
;; **Trap 2 (holis). Programs can name the role vectors.**
;;
;; - *Where.* Code cell 2, `vsa_env`, the `reserved` list. It puts `lhs`,
;;   `rhs` and `φ` into `env['lex']` and the item memory, and `make_atom`
;;   returns the lexicon entry for any name a program uses.
;; - *Symptom.* `(CAR (x . lhs))` gives `rhs` in about half the runs, at
;;   every D (table above). `read` and `car` can also return `lhs`, `rhs`
;;   or `φ` as atoms (the car(a) spread under E1).
;; - *Cause.* The symbol `lhs` is the role vector itself, and binding
;;   commutes, so unbinding `lhs` from rhs⊗lhs gives `rhs` as strongly as
;;   unbinding it from lhs⊗x gives `x`.
;; - *Fix.* Keep the roles out of the lexicon and the item memory:
;;
;; ```python
;; reserved = ['T', 'F', "NIL", 1]      # atoms a program may name
;; for role in ['lhs', 'rhs', 'φ']:
;;     env[role] = HRR(N=dim)           # roles only, not in env['lex'] or mem
;; ```
;;
;; - *Status.* **Tested.** My harness lets `vsa_env` draw the roles as
;;   before and then deletes them from `env['lex']` and their rows from
;;   the item memory. That is the state the change above gives, with the
;;   same random draws. `lhs` in a program is then a fresh atom. The
;;   trap 2 programs, 40 seeds per row:

^:kindly/hide-code
(before-after (map #(update % "variant" label) (get hf "roles"))
              "variant" ["theirs" "trap 2"])

;;   Nothing else reads the roles from `env['lex']`: the control programs
;;   give the same counts as without the fix (regression table above).
;;
;; **E2 (holis). CAR does not evaluate its argument.**
;;
;; - *Where.* Code cell 2, `vsa_env`, `caar`, `cdar` and `cons`.
;; - *Symptom.* `(CAR (CDR (a b c)))` gives `CDR`. Lisp 1.5 gives `b`.
;;   `(CDR (CDR (a b c)))` gives `((a b c))` instead of `(c)` (table
;;   below).
;; - *Cause.* `caar(c)` is `car(car(c))`: it takes the first element of
;;   the argument form instead of its value.
;; - *Fix,* if Lisp 1.5 semantics are intended. Evaluate first, and read
;;   their QUOTE form as its argument:
;;
;; ```python
;; def val(x, ind=''):
;;     v = ev_until_done(x, ind)
;;     if (car(v) == env['lex']['QUOTE']) > floor:   # (QUOTE (x)) -> x
;;         v = car(car(cdr(v)))
;;     return v
;; def caar(c, ind=''):  return car(val(car(c), ind))
;; def cdar(c, ind=''):  return cdr(val(car(c), ind))
;; def cons(ab, ind=''): return ccons(val(car(ab), ind), val(car(cdr(ab)), ind))
;; ```
;;
;; - *Status.* **Tested.** I first suggested only the two `ev_until_done`
;;   lines, and that `quote` return its argument. Both were incomplete.
;;   Without the unwrapping, a quoted list comes back with its wrapper,
;;   `(QUOTE ((a b c)))`. With `quote` returning its argument,
;;   `ev_until_done` evaluates the argument again, so `(QUOTE (CDR a))`
;;   runs `CDR` on an atom (E1) and gives a random item. The version
;;   above leaves `quote` alone. Unquoted literal lists such as `(a b c)`
;;   keep working, because their evaluator returns a list with an unknown
;;   head unchanged. CONS evaluates by accident in their code: the loop
;;   in `ev_until_done` evaluates the new cell again. Over 5 seeds:

^:kindly/hide-code
(before-after (get hf "e2") "variant"
              ["theirs" "E2" "E2 + QUOTE returns its argument" "E2 + unwrap QUOTE"])

;;   Here `E2` is the two lines alone (and CONS), and `E2 + unwrap QUOTE`
;;   is the fix shown. The control programs keep their counts with it
;;   (regression table above). Their fizzbuzz helpers do not all survive
;;   it (the fizzbuzz table above, column `E2 + unwrap QUOTE + E4`): in
;;   one of the 5 seeds, every call after the first runs into unbounded
;;   recursion. Raising Python's recursion limit to 200 000 did not let it
;;   finish within 15 minutes (a side run, not in the JSON files). I
;;   suspect E1: evaluating
;;   CAR's argument runs `ev` on more atoms, and `ev` takes CAR of an
;;   atom. I did not confirm this, and with no working E1 fix (above) I
;;   could not test the pair.
;;
;; ### Robustness under noise
;;
;; **Trap 3 (holis). A fixed θ = 0.2 in the pointer memory.**
;;
;; - *Where.* Code cell 2, `SimpleAssoc.__init__` (`theta=0.2`); `vsa_env`
;;   creates it with the default.
;; - *Symptom.* Under probe noise, lookups fail past σ ≈ 4.9 at any D
;;   (table above, and the trap 3 tables).
;; - *Cause.* A noisy probe has cosine 1/√(1 + σ²) to its key, which
;;   falls below 0.2 at σ = √24 whatever D is, while chance cosines
;;   shrink like 1/√D.
;; - *Fix.* Set θ relative to chance:
;;
;; ```python
;; assoc = SimpleAssoc(lexicon=env['lex'], theta=5 / np.sqrt(dim))
;; ```
;;
;; - *Status.* **Tested**: at D = 8192 lookups stay at 200 of 200 up to
;;   σ = 8, and the LAMBDA program keeps 8 of 10 at σ = 5 and 6
;;   (trap 3 tables). At D = 512 it changes nothing, since 5/√512 ≈ 0.22. I did
;;   not change the evaluator's other 0.2 floors, and I did not run the
;;   fix on the other control programs. Without noise the LAMBDA program
;;   gives the same counts as with θ = 0.2 (9 of 10 at D = 512 and 2048,
;;   7 of 10 at D = 8192).
;;
;; ### Memory and leftovers
;;
;; **E4 (holis). The item memory grows quadratically.**
;;
;; - *Where.* Code cell 2, `SimpleCleanUp.memorize`.
;; - *Symptom.* 800 stored items take 70 100 rows (table above, and the
;;   E4 tables). Their fizzbuzz runs take minutes because of this.
;; - *Cause.* When the matrix is full it appends `k` empty rows but never
;;   raises `m`, so every later item appends another `k` rows.
;; - *Fix.* One line, as residuelisp's `CleanupMemory.memorize` does
;;   (`src/language/encoding.py`, line 90):
;;
;; ```python
;;         if self.i >= self.m:
;;             self.M = np.concatenate([self.M, np.zeros((self.k, self.n))], axis=0)
;;             self.m += self.k
;; ```
;;
;; - *Status.* **Tested**, as shown. Rows of the matrix after storing n
;;   items:

^:kindly/hide-code
(table ["items stored" "rows, theirs" "rows, with the fix" "MB at D = 512, theirs" "MB, with the fix"]
       (for [[n rs] (group-by #(get % "items stored") (get hf "e4_growth"))
             :let [t (by rs "variant" "theirs") f (by rs "variant" "E4")]]
         [n (get t "matrix rows") (get f "matrix rows") (get t "MB at D=512") (get f "MB at D=512")]))

;;   The control programs keep their counts (regression table above). The
;;   fizzbuzz helpers give the right answers in all 5 seeds (fizzbuzz
;;   table above, column `E4`), and the time per call drops from minutes
;;   to seconds. Seed 0, against the control at the top of the page:

^:kindly/hide-code
(let [theirs (into {} (for [r (get hc "fizzbuzz")] [(get r "program") (get r "seconds")]))]
  (table ["call" "seconds, theirs" "seconds, with the fix"]
         (for [r (get hf "fizzbuzz")
               :when (and (= "E4" (get r "variant")) (= 0 (get r "seed")))]
           [(get r "program") (get theirs (get r "program")) (get r "seconds")])))

;;   So here the empty rows changed no result. They only cost time.
;;
;; **E3 (holis). LABEL is a leftover.** Their paper says it leaves LABEL
;; out, so this breaks no claim.
;;
;; - *Where.* Code cell 2, `vsa_env`: `'LABEL'` in `basic_fs` and in
;;   `env.update`, and the inner `label`.
;; - *Symptom.* `(LABEL f (LAMBDA (x) x))` raises a `TypeError`.
;; - *Cause.* `ev` calls every basic function with `ind=`, and `label`
;;   has no such parameter. Inside, `mem.memorize(lbl, exp)` passes two
;;   arguments to a one-argument method, so it would fail next.
;; - *Fix.* Remove `'LABEL'` from `basic_fs` and from `env.update`. To
;;   keep it, let it enter the name the way DEFINE does, and return the
;;   lambda:
;;
;; ```python
;; def label(lbl_exp, ind=''):
;;     define(lbl_exp, ind)
;;     return car(cdr(lbl_exp))
;; ```
;;
;;   Unlike Lisp 1.5, the name is then global.
;; - *Status.* **Tested.** I first suggested keeping their `label` with an
;;   `ind=''` parameter and `mem.memorize(lbl)`. That stops the error, but
;;   recursion still fails: their helper `ll` replaces the label only at
;;   the top level of the body, never inside it. Removing LABEL turns the
;;   form into plain data. D = 512, 5 seeds:

^:kindly/hide-code
(before-after (get hf "e3") "variant" ["theirs" "E3 removed" "E3 kept" "E3 as DEFINE"])

;;   `E3 kept` is the first suggestion, `E3 as DEFINE` the version above.
;;   Every variant keeps the control programs' counts (regression table
;;   above).

;; ## What did not reproduce
;;
;; - Trap 1 does not apply to either codebase as written: both use pointer
;;   cells. In my paper mode of holis it shows up as a noise-free
;;   failure, but the loss of noise tolerance did not show in my read
;;   test.
;; - Trap 2 shows up only through role names leaking into holis's symbol
;;   table. The (x . y) / (y . x) controls are 40 of 40 correct.
;; - Trap 3 does not reproduce in residuelisp in the form "a cap independent
;;   of D", because its similarity is unnormalised (see the table above).
;; - Trap 5 does not apply.
;;
;; ## Summary
;;
;; | trap | holis (2024) | residuelisp (2025) |
;; |---|---|---|
;; | 1 superposed cells | code: n/a (pointers), paper mode: noise-free failure reproduced, noise margin not | n/a (pointers) |
;; | 2 commuting bind | reproduced via role symbols | n/a |
;; | 3 fixed thresholds | reproduced | not reproduced (the cliff moves with √D) |
;; | 4 shared field | n/a | reproduced (integer tag) |
;; | 5 sum of tuples | n/a | n/a |
;; | new | E1 car of an atom, E2, E3, E4 memory growth | R2 `and`, R3 `if`, R4 decoder range, R5 lexer |

;; ## Appendix
;;
;; ### A. Commands
;;
;; ```sh
;; cd repro/holis
;; python3 -m venv .venv          # Python 3.14.7
;; .venv/bin/pip install numpy more-itertools
;; export OPENBLAS_NUM_THREADS=1
;; ulimit -v 8000000
;; .venv/bin/python holis_control.py
;; .venv/bin/python holis_errata.py
;; .venv/bin/python holis_commute.py
;; .venv/bin/python holis_thresholds.py
;; .venv/bin/python holis_cells.py
;; .venv/bin/python residuelisp_control.py
;; .venv/bin/python residuelisp_errata.py
;; .venv/bin/python residuelisp_thresholds.py
;; .venv/bin/python holis_fixes.py
;; .venv/bin/python residuelisp_fixes.py
;; ```
;;
;; Each script writes the JSON file of the same name, which this page
;; reads. The loader `upstream.py` fetches holis at
;; `db208d782863febfb50ec46b25a3700369f5cfea` and residuelisp at
;; `09864c74f1a35680a0008e59d7ab2571269a986f` into a gitignored cache.
;;
;; ### B. Settings, per file

^:kindly/hide-code
(table ["file" "upstream" "python / numpy" "settings"]
       (for [[f d] [["holis_control.json" hc] ["holis_errata.json" he]
                    ["holis_commute.json" hm] ["holis_thresholds.json" ht]
                    ["holis_cells.json" hcell] ["residuelisp_control.json" rc]
                    ["residuelisp_errata.json" re] ["residuelisp_thresholds.json" rt] ["holis_fixes.json" hf]
                    ["residuelisp_fixes.json" rf]]]
         (let [m (get d "meta")]
           [f (str (get-in m ["upstream" "repo"]) " @ "
                   (subs (get-in m ["upstream" "commit"]) 0 7))
            (str (get m "python") " / " (get m "numpy"))
            (kind/code (json/write-str (get m "settings")))])))

;; The holis notebook I ran has SHA-256
;; `80eee8cdeed896fe1987bd9c391d3001c3e46f736227f8cac8f9784b2dd09175`.
;;
;; ### C. Harness excerpts (my code)
;;
;; Probe noise, wrapped around their two cleanup methods
;; (`holis_common.py`):
;;
;; ```python
;; def _noisy(hrr):
;;     s = STATE["sigma"]
;;     if s <= 0:
;;         return hrr
;;     v = hrr.v
;;     n = STATE["rng"].normal(size=v.size) * (s * np.linalg.norm(v) / math.sqrt(v.size))
;;     return H.HRR(data=v + n)
;;
;; H.SimpleCleanUp.recall = lambda self, hrr: _ORIG_MEM_RECALL(self, _noisy(hrr))
;; H.SimpleAssoc.recall   = ...  # the same, and counts theta fallbacks
;; ```
;;
;; The trap 3 control is one attribute, `env["assoc"].theta = 5 / sqrt(D)`. The E1
;; control rebinds their inner `car` in the closures of `vsa_env`
;; (`holis_errata.py`):
;;
;; ```python
;; def car(c, ind=""):
;;     return orig(c) if c in assoc.A else nil
;; ```
;;
;; Paper mode, the trap 1 experiment, is two attributes on their pointer
;; memory (`holis_cells.py`):
;;
;; ```python
;; a.memorize = lambda probe, trace: (trace.unit(), trace.unit())
;; a.recall = lambda probe: probe
;; ```
;;
;; For residuelisp I also silence its debug output, stub the `decode` it
;; calls inside log messages while a program runs, reset its class-level
;; RHC state per seed, and stop a run after a budget of cleanups. Under
;; noise its `tuple_to_list` loops until `car` raises, which can be never.
;;
;; ### D. Raw results
;;
;; holis, trap 3 lookups (all rows):

^:kindly/hide-code
(table ["D" "mode" "σ" "correct" "θ fallbacks"]
       (for [r (get ht "lookup")]
         [(get r "D") (get r "mode") (get r "sigma") (frac r "correct" "trials")
          (get r "theta_fallbacks")]))

;; holis, trap 3 program runs, with what went wrong:

^:kindly/hide-code
(table ["D" "mode" "σ" "correct" "errors" "wrong values"]
       (for [r (get ht "program")]
         [(get r "D") (get r "mode") (get r "sigma") (frac r "correct" "runs")
          (fmt (get r "errors")) (fmt (get r "wrong_values"))]))

;; residuelisp, trap 3 runs, with what went wrong:

^:kindly/hide-code
(table ["D" "σ" "correct" "errors" "wrong values"]
       (for [r (get rt "rows")]
         [(get r "D") (get r "sigma") (frac r "correct" "runs")
          (fmt (get r "errors")) (fmt (get r "wrong_values"))]))

;; -----
;;
;; *Generated by Claude on Benjamin Schwerdtner's behalf.*
