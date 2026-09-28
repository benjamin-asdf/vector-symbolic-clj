(ns vsc.machine-report
  "Numbers and figures for docs/results-S2.md:

    clojure -M:jvm -m vsc.machine-report [out-dir]

  writes the rule table, a state trace of a tiny program, and a benchmark
  of the host evaluator against the vector machine."
  (:require
   [clojure.java.io :as io]
   [clojure.pprint :as pp]
   [libpython-clj2.python :as py]
   [vsc.core :as vsc]
   [vsc.hdc :as h]
   [vsc.machine :as machine]))

(def programs
  '[(defn fact [n] (if (zero? n) 1 (* n (fact (dec n)))))
    (defn fib [n] (cond (zero? n) 0 (zero? (dec n)) 1
                        :else (+ (fib (dec n)) (fib (dec (dec n))))))])

(defn- M [] (get @@#'vsc.core/machine :M))

(defn- counting
  "Run f while counting substrate calls by kind. A full scan of the item
  memory M is: a nearest/intern/deref on M, a recognize or cleanup (one
  scan each), a part (two). Returns [result counts]."
  [f]
  (let [c (atom {})
        tick! (fn [k n] (swap! c update k (fnil + 0) n))
        wrap (fn [k scans g]
               (fn [& args]
                 (tick! k 1)
                 (tick! :m-scans (scans args))
                 (apply g args)))
        on-M (fn [n] (fn [[mem]] (if (identical? mem (M)) n 0)))
        always (fn [n] (constantly n))
        none (constantly 0)
        r (with-redefs [h/bind (wrap :bind none h/bind)
                        h/unbind (wrap :unbind none h/unbind)
                        h/bundle (wrap :bundle none h/bundle)
                        h/nearest (wrap :nearest (on-M 1) h/nearest)
                        h/intern! (wrap :intern (on-M 1) h/intern!)
                        h/deref-ptr (wrap :deref (on-M 1) h/deref-ptr)
                        h/recall (wrap :recall (on-M 1) h/recall)
                        h/recognize (wrap :recognize (always 1) h/recognize)
                        h/cleanup (wrap :cleanup (always 1) h/cleanup)
                        h/part (wrap :part (always 2) h/part)]
            (f))]
    [r @c]))

(defn- secs [t0] (/ (- (System/nanoTime) t0) 1e9))

(defn- on-big-stack [f]
  (let [p (promise)
        t (Thread. nil #(deliver p (try (f) (catch Throwable e e))) "big" (* 512 1024 1024))]
    (.start t)
    (let [r @p] (if (instance? Throwable r) (throw r) r))))

(defn- bench-host
  "Host veval: wall time, veval calls (eval steps), substrate calls, M growth."
  [form]
  (machine/init!)
  (doseq [p programs] (machine/run p))
  (let [before (h/mem-size (M))
        steps (atom 0)
        veval @#'vsc.core/veval
        t0 (System/nanoTime)
        [r c] (with-redefs [vsc.core/veval (fn [e env] (swap! steps inc) (veval e env))]
                (on-big-stack #(counting (fn [] (vsc/run form)))))]
    {:evaluator :host :form form :value r :secs (secs t0) :steps @steps
     :m-scans (:m-scans c) :calls (dissoc c :m-scans)
     :traces+ (- (h/mem-size (M)) before)}))

(defn- bench-machine
  [form {:keys [icache?] :or {icache? true}}]
  (machine/init! {:icache? icache?})
  (doseq [p programs] (machine/run p))
  (machine/reset-stats!)
  (let [before (h/mem-size (M))
        t0 (System/nanoTime)
        [r c] (counting #(machine/run form))
        s (machine/stats)]
    {:evaluator (if icache? :machine :machine-no-icache) :form form :value r
     :secs (secs t0) :steps (:steps s) :instrs (:instrs s)
     ;; substrate: M scans in the machine's own datapath + in primitives
     :m-scans (+ (:scan s) (:m-scans c 0)) :calls (dissoc c :m-scans)
     :work (dissoc s :steps :instrs)
     :traces+ (- (h/mem-size (M)) before)}))

(defn- margins
  "For every rule selection while running `form`: the similarity of the
  winning key, the runner-up, and the softmax weight of the winner at
  several inverse temperatures β."
  [form]
  (machine/init!)
  (doseq [p programs] (machine/run p))
  (let [rows (atom [])
        W (get @@#'vsc.machine/vm :work)]
    (binding [machine/*trace*
              (fn [_ i regs]
                (let [key (aget ^objects regs (.indexOf ^java.util.List machine/registers 'KEY))
                      sims (vec (py/->jvm (py/call-attr W "rule_sims" key)))
                      [a b] (take 2 (sort > sims))]
                  (swap! rows conj {:rule i :top a :second b :sims sims})))]
      (machine/run form))
    (let [rs @rows
          weight (fn [beta {:keys [sims rule]}]
                   (let [e (map #(Math/exp (* beta %)) sims)]
                     (/ (nth e rule) (reduce + e))))]
      {:selections (count rs)
       :min-top (apply min (map :top rs))
       :max-second (apply max (map :second rs))
       :min-margin (apply min (map #(- (:top %) (:second %)) rs))
       :mean-margin (/ (reduce + (map #(- (:top %) (:second %)) rs)) (count rs))
       :winner-weight (into (sorted-map)
                            (for [beta [1 10 30 100]]
                              [beta {:min (apply min (map #(weight beta %) rs))
                                     :mean (/ (reduce + (map #(weight beta %) rs)) (count rs))}]))})))

(defn -main [& [dir]]
  (let [dir (io/file (or dir "out"))
        spit* (fn [name s] (spit (io/file dir name) s) (println "wrote" (str (io/file dir name))))]
    (.mkdirs dir)
    (machine/init!)
    (spit* "s2-rules.txt" (with-out-str (machine/print-rule-table)))
    (spit* "s2-trace.txt"
           (with-out-str
             (machine/print-trace '((fn [x] (if (zero? x) :zero (inc x))) 1))))
    (let [rows (vec (for [form '[(fact 5) (fib 8)]
                          r [(bench-host form) (bench-machine form {})
                             (when (= form '(fact 5)) (bench-machine form {:icache? false}))]
                          :when r]
                      (do (pp/pprint r) r)))]
      (spit* "s2-bench.edn" (with-out-str (pp/pprint rows))))
    (let [m (margins '(fib 8))]
      (pp/pprint m)
      (spit* "s2-margins.edn" (with-out-str (pp/pprint m)))))
  (shutdown-agents))
