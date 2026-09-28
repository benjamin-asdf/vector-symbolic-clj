;; S1 tower overhead: the same forms evaluated by the host `veval` (one level)
;; and by `vsc-eval` from resources/vsc/eval.clj running on the host (two
;; levels). Host Clojure, not the dialect. Run with
;;
;;   VSC_PYTHON=.venv/bin/python clojure -M:jvm -i bench/s1_overhead.clj
;;
;; Every measurement starts from a fresh machine (init!, eval.clj, the prelude
;; loaded through vsc-eval, fact/fib defined at both levels), so both levels
;; see the same memory size |M|: cleanup cost is linear in |M|.

(require '[vsc.core :as vsc] '[vsc.hdc :as h] '[clojure.java.io :as io])

(def counts (atom {}))

(doseq [v [#'h/part #'h/cleanup #'h/recognize #'h/intern! #'vsc.core/veval]]
  (let [f @v k (keyword (:name (meta v)))]
    (alter-var-root v (fn [_] (fn [& args] (swap! counts update k (fnil inc 0)) (apply f args))))))

;; JVM stack depth, sampled at every application while `sample?` is on (it
;; costs a stack walk per call, so it is only on for the countdown forms)
(def max-depth (atom 0))
(def sample? (atom false))
(let [f @#'vsc.core/apply-fn]
  (alter-var-root #'vsc.core/apply-fn
                  (fn [_] (fn [g args]
                            (when @sample?
                              (swap! max-depth max (count (.getStackTrace (Thread/currentThread)))))
                            (f g args)))))

(def defs
  '[(defn fact [n] (if (zero? n) 1 (* n (fact (dec n)))))
    (defn fib [n] (cond (zero? n) 0 (zero? (dec n)) 1
                        :else (+ (fib (dec n)) (fib (dec (dec n))))))
    (defn countdown [n] (if (zero? n) 0 (countdown (dec n))))
    (defn m-eval [e env]
      (cond (number? e) e
            (symbol? e) (get env e)
            (= (first e) (quote lambda)) (list (quote closure) (nth e 1) (nth e 2) env)
            (= (first e) (quote if0)) (if (zero? (m-eval (nth e 1) env))
                                        (m-eval (nth e 2) env)
                                        (m-eval (nth e 3) env))
            (= (first e) (quote +)) (+ (m-eval (nth e 1) env) (m-eval (nth e 2) env))
            :else (m-apply (m-eval (first e) env) (m-eval (nth e 1) env))))
    (defn m-apply [c arg]
      (m-eval (nth c 2) (assoc (nth c 3) (nth c 1) arg)))])

(defn s1-form [form] (list 'vsc-eval (list 'quote form) ()))

(defn fresh! []
  (vsc/init!)
  (vsc/run-string (slurp (io/resource "vsc/eval.clj")))
  (vsc/run (list 'vsc-run-all (list 'quote (vsc/read-forms (slurp (io/resource "vsc/prelude.clj"))))))
  (doseq [d defs] (vsc/run d) (vsc/run (s1-form d))))

(defn measure [level form]
  (fresh!)
  (reset! counts {})
  (reset! max-depth 0)
  (reset! sample? (= 'countdown (first form)))
  (let [m0 (:traces (vsc/stats))
        base-depth (count (.getStackTrace (Thread/currentThread)))
        t0 (System/nanoTime)
        r (vsc/run (if (= level :s1) (s1-form form) form))
        dt (/ (- (System/nanoTime) t0) 1e9)]
    (merge {:form form :level level :result r :wall-s dt :M0 m0
            :traces (- (:traces (vsc/stats)) m0)
            :jvm-frames (- @max-depth base-depth)}
           @counts)))

(def forms
  (or (some-> (System/getenv "S1_FORMS") read-string)
      '[(fact 2) (fact 3) (fact 4) (fact 5)
        (fib 3) (fib 4) (fib 5) (fib 6)
        (countdown 1) (countdown 2) (countdown 4) (countdown 8)
        (m-eval (quote (((lambda x (lambda y (+ x y))) 3) 4)) {})]))

(println "| form | level | result | wall s | veval | part | cleanup+recognize | new traces | max JVM frames |")
(println "|---|---|---|---|---|---|---|---|---|")
(doseq [form forms level [:host :s1]]
  (let [{:keys [result wall-s veval part cleanup recognize traces jvm-frames]} (measure level form)]
    (println (format "| `%s` | %s | %s | %.2f | %d | %d | %d | %d | %d |"
                     (pr-str form) (name level) (pr-str result) wall-s
                     (or veval 0) (or part 0) (+ (or cleanup 0) (or recognize 0)) traces jvm-frames))
    (flush)))

(shutdown-agents)
