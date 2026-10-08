;; Every failure mode of paper §4, rebuilt broken next to the current code.
;; The broken variants are swapped in with with-redefs; nothing in src/ changes.
;;
;;   clojure -M:jvm scripts/failure_modes.clj
;;
;; Takes ~15 min. Each section prints the broken and the fixed behaviour;
;; paper §4 and docs/ISSUE-FINDINGS.md quote these outputs.
(require '[vsc.core :as c] '[vsc.hdc :as h] '[vsc.worlds :as w]
         '[libpython-clj2.python :as py] '[clojure.string :as str])

(defn p [& xs] (apply println xs) (flush))
(defn try-run [f]
  (try (f)
       (catch Throwable e
         (let [msg (str/replace (str "ERR " (ex-message e)) #"\s+" " ")]
           (subs msg 0 (min 70 (count msg)))))))
(defn priv [s] @(ns-resolve 'vsc.core s))

(def nu (priv 'nu)) (def bundle (priv 'bundle)) (def bind (priv 'bind))
(def unbind (priv 'unbind)) (def role (priv 'role)) (def kind->label (priv 'kind->label))
(def label->kind (priv 'label->kind)) (def m* (priv 'm)) (def permute (priv 'permute))
(def num-vec (priv 'num-vec)) (def tag (priv 'tag)) (def check-capacity (priv 'check-capacity))
(def vec-slot (priv 'vec-slot))

(defn budgeted
  "Runs f under a time and row budget: some broken variants run away."
  [f]
  (h/budget! (c/space) {:seconds 30 :max-rows 20000})
  (try (try-run f) (finally (h/budget! (c/space) {}))))

;; ---------------------------------------------------------------------------
(p "\n== 4.1 superposed cells (the cell is the trace L⊗a + R⊗b, no pointer)")
(def broken-41
  {#'vsc.core/cell (fn [kind a b]
                     (let [M (c/item-memory)]
                       (h/mem-get M (h/mem-add! M (nu (bundle (bind (role :L) a) (bind (role :R) b)))
                                                (kind->label kind) 0.999))))
   #'vsc.core/car (fn [x] (c/cleanup (unbind (role :L) x)))
   #'vsc.core/cdr (fn [x] (c/cleanup (unbind (role :R) x)))})
(doseq [dim [1024 2048] sigma [0.0 2.0]
        :let [forms '[(fact 5) (reduce + 0 [1 2 3 4])]
              run (fn [seed]
                    (c/init! {:dim dim :seed seed})
                    (c/run '(defn fact [n] (if (zero? n) 1 (* n (fact (dec n))))))
                    (c/set-knobs! {:probe-noise sigma :noise-seed seed})
                    (mapv (fn [form] (budgeted #(c/run form))) forms))]]
  (let [fixed (mapv run (range 3))
        broken (with-redefs-fn broken-41 #(mapv run (range 3)))]
    (p "D" dim "σ" sigma "| fixed:" fixed "| superposed cells:" broken)))
(c/set-knobs! {})

;; ---------------------------------------------------------------------------
(p "\n== 4.2 binding commutes (map values boxed like keys)")
(c/init!)
(p "fixed: (get {1 2 2 3} 2) =" (c/run '(get {1 2 2 3} 2)))
(with-redefs-fn {#'vsc.core/value-slot (priv 'slot)}
  #(do (c/init!) (p "values boxed like keys:" (try-run (fn [] (c/run '(get {1 2 2 3} 2)))))))

;; ---------------------------------------------------------------------------
(p "\n== 4.3 residue chimeras (integers unboxed in sets)")
(p "Not reproducible end to end: the readout that failed is gone.")
(p "The mechanism: python3 scripts/paper_f3.py")
(c/init! {:prelude? false})
(p "fixed: (contains? #{11 12} 13102) =" (c/run '(contains? #{11 12} 13102))
   " (contains? #{11 12} 12) =" (c/run '(contains? #{11 12} 12)))
(with-redefs-fn {#'vsc.core/slot (fn [x] x)}
  #(do (c/init! {:prelude? false})
       (p "unboxed: (contains? #{11 12} 13102) =" (try-run (fn [] (c/run '(contains? #{11 12} 13102)))))))

;; ---------------------------------------------------------------------------
(p "\n== 4.4 zero")
(c/init!)
(let [s (h/sim (c/space) (c/eval-form 0) (c/eval-form 5005))]
  (p "sim(0, 5005) =" (format "%.3f" s) "-> a zero? thresholded at 0.5 says" (> s 0.5)
     "| fixed: (zero? 5005) =" (c/run '(zero? 5005)) " (zero? 0) =" (c/run '(zero? 0))))
;; dimension 0 dead: the old 0 was B^0, a delta at index 0
(let [sp (c/space)
      np (py/import-module "numpy")
      mask (py/call-attr np "ones" (py/get-attr sp "dim"))]
  (py/set-item! mask 0 0.0)
  (py/set-attr! sp "mask" mask)
  (p "fixed, dimension 0 dead: (+ 17 25) =" (try-run #(c/run '(+ 17 25)))
     " (* 6 7) =" (try-run #(c/run '(* 6 7)))
     "| the old 0 (B^0) keeps"
     (format "%.3f" (py/->jvm (py/call-attr np "sum"
                                            (py/call-attr np "abs"
                                                          (py/call-attr np "multiply" (h/identity-vec sp) mask)))))
     "of its weight")
  (py/set-attr! sp "mask" nil))

;; ---------------------------------------------------------------------------
(p "\n== 4.5 fixed thresholds (kind recognised at cos > 0.5)")
(def broken-45
  {#'vsc.core/kind (fn [v]
                     (let [[label i s] (h/recognize (m* :C) v)
                           k (label->kind label)]
                       (if (> s 0.5)
                         [k (and (not= k :num) (contains? (m* :empty-idx) i))]
                         [:unknown false])))})
(defn roundtrip [sigma seed]
  (c/init! {:prelude? false :seed seed})
  (c/run '(def x (quote (a b c d))))
  (c/set-knobs! {:probe-noise sigma :noise-seed seed})
  (let [r (try (= '(a b c d) (c/run 'x)) (catch Throwable _ false))]
    (c/set-knobs! {}) r))
(doseq [sigma [1.0 2.0 3.0 5.0]]
  (p "σ" sigma "| fixed:" (count (filter true? (map #(roundtrip sigma %) (range 10)))) "/10"
     "| threshold 0.5:" (with-redefs-fn broken-45 #(count (filter true? (map (fn [s] (roundtrip sigma s)) (range 10))))) "/10"))

;; ---------------------------------------------------------------------------
(p "\n== 4.6 a shared field (the count inside the vector)")
(defn header-variant [role-k weight]
  {#'vsc.core/mk-vec-plain
   (fn [xs]
     (let [xs (vec xs) n (count xs) f (if ((priv 'hook) :construct) nu identity)]
       (if (zero? n)
         (tag :vec)
         (do (check-capacity "vector" 1.5 n)
             (let [elems (nu (apply bundle (map-indexed (fn [i x] (permute (bind (role :V) (f (vec-slot x))) i)) xs)))
                   trace (nu (h/lincomb (c/space) [elems (bind (role role-k) (num-vec n))] [1.0 weight]))]
               (h/mem-get (c/item-memory) (h/mem-add! (c/item-memory) trace (kind->label :vec) 0.999)))))))
   #'vsc.core/vec-size
   (fn ^long [v] (let [[_ n] (h/recognize (m* :C) (unbind (role role-k) v))]
                   (if (<= 1 n 64) n (throw (ex-info (str "unreadable vector size " n) {})))))})
(doseq [[label variant] [["count C⊗n in the trace" (header-variant :C 1.0)]
                         ["count ½N⊗n in the trace" (header-variant :N 0.5)]
                         ["fixed (count behind the row)" {}]]]
  (with-redefs-fn variant
    #(do (c/init!)
         (p label "| #{[1 2] [1 3]} ->" (budgeted (fn [] (c/run '#{[1 2] [1 3]})))
            "| similarity [:a :b] [:c :d] =" (try-run (fn [] (c/run '(similarity [:a :b] [:c :d])))))
         (w/init! {:dim 2048})
         (p "   worlds of [[:a [:x]] [:b [:y :z]]] ->"
            (budgeted (fn [] (w/run '(quote [[:a [:x]] [:b [:y :z]]]))))))))

;; ---------------------------------------------------------------------------
(p "\n== 4.7 a sum of tuples forgets the joint (TV to the exact joint)")
(def die [1 2 3 4])
(def dice-exact (into {} (for [a die b die] [[a (+ a b)] (/ 1.0 16)])))
(defn sprinkler-exact []
  (let [p (fn [q x] (if x q (- 1.0 q)))
        sprk {true 0.1 false 0.4}
        wet {[true true] 0.99 [true false] 0.8 [false true] 0.9 [false false] 0.0}]
    (into {} (for [r [true false] s [true false] wt [true false]
                   :let [pr (* (p 0.2 r) (p (sprk r) s) (p (wet [r s]) wt))]
                   :when (pos? pr)]
               [[r s wt] pr]))))
(def sprinkler-src (slurp "examples/worlds/sprinkler.clj"))
(doseq [seed [0 1 2]]
  (let [tv (fn [want form] (try (format "%.3f" (w/tv-distance want (w/worlds (w/eval-form form))))
                                (catch Throwable e (str "ERR " (ex-message e)))))]
    (w/init! {:dim 4096 :seed seed})
    (let [dl (tv dice-exact '(for-worlds [a (amb 1 2 3 4) b (amb 1 2 3 4)] (list a (+ a b))))
          dv (tv dice-exact '(for-worlds [a (amb 1 2 3 4) b (amb 1 2 3 4)] [a (+ a b)]))]
      (w/run-string sprinkler-src)
      (let [sl (tv (sprinkler-exact) 'joint)]
        (w/run-string (str/replace sprinkler-src "(list rain sprinkler wet)" "[rain sprinkler wet]"))
        (p "seed" seed "| dice: lists" dl "vectors" dv
           "| sprinkler: lists" sl "vectors" (tv (sprinkler-exact) 'joint))))))

(p "\ndone")
(shutdown-agents)
(System/exit 0)
