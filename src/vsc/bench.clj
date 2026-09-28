(ns vsc.bench
  "The P3 task suite: the programs in bench/*.clj, with their expected values
  (derived in real Clojure, see bench/reference/) in bench/expected.edn.

    clojure -M:jvm:bench              ;; every task, a timing table
    clojure -M:jvm:bench fib bst      ;; just these
    clojure -M:jvm:bench --dim 4096   ;; at another dimension"
  (:require
   [clojure.edn :as edn]
   [clojure.pprint :as pp]
   [clojure.string :as str]
   [vsc.core :as vsc]
   [vsc.hdc :as h]))

(def expected-file "bench/expected.edn")

(defn tasks
  "task name (a symbol) -> {:file :expected :notes}, sorted by name."
  []
  (into (sorted-map) (edn/read-string (slurp expected-file))))

;; Substrate calls are counted by wrapping the bridge functions in vsc.hdc:
;; each one is one call into Python (bind, cleanup, a memory scan, ...).
(def ^:private substrate-fns
  [#'h/unitary #'h/bind #'h/unbind #'h/inverse #'h/bundle #'h/normalize #'h/sim
   #'h/degrade #'h/num-vec #'h/read-num #'h/mem-size #'h/mem-add! #'h/mem-put!
   #'h/mem-get #'h/mem-label #'h/nearest #'h/recall #'h/peel #'h/intern!
   #'h/deref-ptr #'h/recognize #'h/cleanup #'h/part])

(defn- counting
  "Run `thunk` with every substrate call counted and timed;
  [result {fn-name count} {fn-name nanoseconds}]."
  [thunk]
  (let [counts (atom {})
        nanos (atom {})
        timed (fn [k t0 x]
                (swap! nanos update k (fnil + 0) (- (System/nanoTime) t0))
                (swap! counts update k (fnil inc 0))
                x)
        ;; callers of primitive-hinted fns invoke them as IFn$OOOD / IFn$OL,
        ;; so their wrappers must keep the same primitive signature
        wrap (fn [v]
               (let [f @v k (:name (meta v))]
                 (condp = v
                   #'h/sim (fn ^double [s a b]
                             (let [t0 (System/nanoTime)]
                               (double (timed k t0 (.invokePrim ^clojure.lang.IFn$OOOD f s a b)))))
                   #'h/mem-size (fn ^long [mem]
                                  (let [t0 (System/nanoTime)]
                                    (long (timed k t0 (.invokePrim ^clojure.lang.IFn$OL f mem)))))
                   (fn [& args] (let [t0 (System/nanoTime)] (timed k t0 (apply f args)))))))]
    (with-redefs-fn (into {} (map (juxt identity wrap)) substrate-fns)
      (fn [] (let [x (thunk)] [x @counts @nanos])))))

(defn run-task
  "Run one task on a fresh machine (`init-opts` as for vsc.core/init!).
  Returns the decoded value, whether it matches, wall time of the program
  (not of init!), trace growth of M, and the substrate calls it made with
  the share of the wall time spent inside them."
  ([task] (run-task task {}))
  ([task init-opts]
   (let [{:keys [file expected]} (get (tasks) task)
         src (slurp file)]
     (vsc/init! init-opts)
     (let [before (vsc/stats)
           t0 (System/nanoTime)
           [value calls nanos] (counting #(try (vsc/run-string src)
                                         (catch Exception e {::error (ex-message e)})))
           ms (/ (- (System/nanoTime) t0) 1e6)
           after (vsc/stats)]
       {:task task
        :value value
        :expected expected
        :ok? (= expected value)
        :ms ms
        :traces (- (:traces after) (:traces before))
        :calls (reduce + (vals calls))
        :substrate-ms (/ (reduce + (vals nanos)) 1e6)
        :substrate-ms-by-fn (update-vals nanos #(/ % 1e6))
        :calls-by-fn calls}))))

(defn- top-calls
  "The three call kinds that take the most substrate time: name, count, ms."
  [{:keys [calls-by-fn substrate-ms-by-fn]}]
  (->> substrate-ms-by-fn (sort-by val >) (take 3)
       (map (fn [[k ms]] (format "%s %d/%.0fms" k (calls-by-fn k) ms)))
       (str/join ", ")))

(defn -main [& args]
  (let [[opts names] (loop [opts {} names [] [a & more :as args] args]
                       (cond (empty? args) [opts names]
                             (= a "--dim") (recur (assoc opts :dim (parse-long (first more)))
                                                  names (rest more))
                             :else (recur opts (conj names (symbol a)) more)))
        names (if (seq names) names (keys (tasks)))
        rows (doall
              (for [t names]
                (let [r (run-task t opts)]
                  (println (format "%-20s %s %8.0f ms" t (if (:ok? r) "ok  " "FAIL") (:ms r)))
                  (flush)
                  r)))]
    (pp/print-table
     [:task :ok? :ms :substrate% :traces :calls :top-calls]
     (for [r rows]
       (assoc r
              :ms (Math/round (double (:ms r)))
              :substrate% (Math/round (/ (* 100.0 (:substrate-ms r)) (:ms r)))
              :top-calls (top-calls r))))
    (doseq [r rows :when (not (:ok? r))]
      (println "FAIL" (:task r) "expected" (pr-str (:expected r)) "got" (pr-str (:value r))))
    (println (format "%d/%d tasks ok, %.1f s total"
                     (count (filter :ok? rows)) (count rows)
                     (/ (reduce + (map :ms rows)) 1000.0)))
    (shutdown-agents)
    (System/exit (if (every? :ok? rows) 0 1))))
