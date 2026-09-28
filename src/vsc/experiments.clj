(ns vsc.experiments
  "Experiment runner: an EDN spec in, one CSV row per run out.

    clojure -M:jvm:exp specs/e1.edn [--workers 2] [--resume]

  A spec is a grid plus a task:

    {:name \"e2\"                     ;; out/e2.csv unless :out is given
     :workers 2                        ;; process pool size (capped, see below)
     :init {:prelude? false}           ;; extra init! options for every run
     :budget {:max-ops 2e6 :max-rows 60000 :seconds 120}
     :reference? false                 ;; also run noise-free, report the first
                                       ;; cleanup whose winner differs
     :grid [[:dim [1024 2048]]         ;; ordered; the last axis varies fastest
            [:probe-noise {:linspace [0 3 13]}]
            [:seed {:range [0 20]}]
            [:task [:list :map]]]
     :tasks {:list {:setup [(def x (quote (a b c)))] :form x :expect (a b c)}
             :map  {:gen vsc.experiments.tasks/structure :struct :map :n 8}}
     :exclude [{:dim 1024 :task :map}]}  ;; grid points to skip (submatch)

  Grid keys :dim :seed :memory :memory-opts :op-noise :probe-noise :lesion
  :memory-damage configure the machine; every other key is a task parameter.
  A task is {:form f :expect v} (or {:file path :expect v}, value = last
  form), with optional :setup forms (and a :prepare thunk) run before the
  knobs are switched on, or
  {:gen sym ...}: a fn of the run's params (task keys merged over grid keys)
  that returns such a map, or one with :run (a thunk after setup, returning
  {:successes k :trials t}) or :native (a thunk that builds its own
  substrate; no init!).

  Each run: init! noise-free, run :setup, switch the knobs on, damage!,
  reset the instruments, then time the task. Knobs act on everything after
  setup, the reader and printer included."
  (:require
   [clojure.edn :as edn]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [vsc.core :as vsc]
   [vsc.hdc :as h])
  (:import
   (java.io File)))

;; ---------------------------------------------------------------------------
;; the grid

(def machine-keys [:dim :seed :memory :memory-opts :op-noise :probe-noise :lesion :memory-damage])

(defn- axis-values [v]
  (cond
    (map? v) (let [[k args] (first v)]
               (case k
                 :range (vec (apply range args))
                 :linspace (let [[a b n] args]
                             (if (= 1 n)
                               [(double a)]
                               (mapv #(+ a (* % (/ (- b a) (dec n) 1.0))) (range n))))
                 :geom (let [[a b n] args
                             r (Math/pow (/ b a) (/ 1.0 (dec n)))]
                         (mapv #(Math/round (* a (Math/pow r %))) (range n)))
                 (throw (ex-info (str "unknown axis form " k) {:axis v}))))
    (sequential? v) (vec v)
    :else [v]))

(defn expand
  "Every run of `spec` as a param map with a stable :run index."
  [spec]
  (let [axes (mapv (fn [[k v]] [k (axis-values v)]) (:grid spec))
        excluded? (fn [m] (some #(= % (select-keys m (keys %))) (:exclude spec)))
        combos (remove excluded?
                       (reduce (fn [acc [k vs]] (for [m acc v vs] (assoc m k v))) [{}] axes))]
    (vec (map-indexed (fn [i m] (assoc (merge (:fixed spec) m) :run i)) combos))))

(defn- block-size
  "Runs that share everything but the innermost axis go to one worker (so a
  task can cache its setup across them)."
  [spec]
  (max 1 (count (axis-values (second (last (:grid spec)))))))

(defn shard [spec runs i n]
  (let [b (block-size spec)]
    (->> (partition-all b runs)
         (keep-indexed (fn [j blk] (when (= i (mod j n)) blk)))
         (apply concat))))

;; ---------------------------------------------------------------------------
;; tasks

(defn- resolve-task [spec params]
  (let [t (if-let [k (:task params)]
            (or (get-in spec [:tasks k]) (throw (ex-info (str "no task " k) {:task k})))
            (:task spec))
        t (if-let [g (:gen t)]
            ((requiring-resolve g) (merge params (dissoc t :gen)))
            t)]
    (if-let [f (:file t)]
      (let [forms (vsc/read-forms (slurp f))]
        (assoc t :setup (vec (concat (:setup t) (butlast forms))) :form (last forms)))
      t)))

(defn- knobs [params] (select-keys params [:op-noise :probe-noise :lesion :noise-seed]))

(defn- init-opts [spec params]
  (merge {:prelude? true} (:init spec) (select-keys params [:dim :seed :memory :memory-opts])))

(defn- run-task
  "Run task `t` on the current machine: setup, knobs, damage, measure. The
  clock covers the task only, not the setup."
  [spec params t {:keys [log?]}]
  (let [t0 (volatile! (System/nanoTime))
        ms #(/ (- (System/nanoTime) @t0) 1e6)]
    (try
      (try
        (doseq [f (:setup t)] (vsc/run f))
        (when-let [p (:prepare t)] (p))
        (catch Throwable e
          (throw (ex-info (str "setup: " (or (ex-message e) (.getName (class e)))) {} e))))
      (vsc/set-knobs! (knobs params))
      (vsc/damage! (or (:memory-damage params) 0.0))
      (vsc/reset-instruments!)
      (vsc/instrument! {:count-ops? true :log-margins? (boolean log?)})
      (h/budget! (vsc/space) (:budget spec))
      (vreset! t0 (System/nanoTime))
      (let [r (if-let [f (:run t)]
                (f)
                (let [v (vsc/run (:form t))]
                  {:value v :successes (if (= v (:expect t)) 1 0) :trials 1}))]
        (assoc r :wall-ms (ms)))
      (catch Throwable e
        {:successes 0 :trials (or (:trials t) 1) :wall-ms (ms)
         :error (or (ex-message e) (.getName (class e)))})
      (finally
        (h/budget! (vsc/space) {})))))

(defn- first-divergence
  "Index of the first cleanup whose [kind winner] differs, or -1."
  [ref xs]
  (or (first (keep-indexed (fn [i [a b]] (when (not= a b) i)) (map vector ref xs)))
      (if (= (count ref) (count xs)) -1 (min (count ref) (count xs)))))

(defn- init! [spec params]
  (let [t0 (System/nanoTime)]
    (vsc/init! (init-opts spec params))
    (/ (- (System/nanoTime) t0) 1e6)))

(defn run-one
  "One row: params, outcome, op counts, margins, timings."
  [spec params]
  (let [t (try (resolve-task spec params)
               (catch Throwable e {:native #(throw e)}))]
    (if-let [f (:native t)]
      (let [t0 (System/nanoTime)
            r (try (f) (catch Throwable e
                         {:successes 0 :trials (or (:trials t) 1)
                          :error (or (ex-message e) (.getName (class e)))}))]
        (merge {:wall-ms (/ (- (System/nanoTime) t0) 1e6)} r))
      (let [ref (when (:reference? spec)
                  (init! spec params)
                  (run-task spec (assoc params :op-noise 0 :probe-noise 0 :lesion 0 :memory-damage 0)
                            t {:log? true})
                  (h/winner-log (vsc/space)))
            init-ms (init! spec params)
            r (run-task spec params t {:log? true})
            {:keys [counts margins]} (vsc/instruments)
            winners (when ref (h/winner-log (vsc/space)))]
        (merge r
               {:init-ms init-ms
                :expected (:expect t)
                :counts counts
                :margins margins
                :m-size (:traces (vsc/stats))}
               (when ref {:first-divergence (first-divergence ref winners)
                          :ref-cleanups (count ref)}))))))

;; ---------------------------------------------------------------------------
;; output

(def op-kinds
  ["bind" "unbind" "bundle" "normalize" "sim" "unitary" "inverse" "num_vec" "num_read"
   "nearest" "recall" "recognize" "cleanup" "clean" "deref" "add" "intern" "put" "peel" "rows"])

(defn- row [spec params r]
  (let [{:keys [counts margins]} r
        task-keys (sort (remove (set (conj machine-keys :run :task :seed)) (keys params)))]
    (merge
     {:exp (:name spec) :run (:run params) :task (some-> (:task params) name)}
     (select-keys params machine-keys)
     (into {} (for [k task-keys] [k (get params k)]))
     {:correct (and (pos? (:trials r 1)) (= (:successes r) (:trials r 1)))
      :successes (:successes r) :trials (:trials r 1)
      :value (some-> (:value r) pr-str) :expected (some-> (:expected r) pr-str)
      :error (:error r)
      :wall-ms (:wall-ms r) :init-ms (:init-ms r) :m-size (:m-size r)
      :first-divergence (:first-divergence r) :ref-cleanups (:ref-cleanups r)
      :cleanups (:n margins) :margin-min (:min margins) :margin-p05 (:p05 margins)
      :margin-median (:median margins) :s1-min (:s1-min margins)
      :ops-total (when counts (reduce + (vals (dissoc counts :rows))))}
     (into {} (for [k op-kinds] [(keyword (str "ops." k)) (get counts (keyword (str/replace k "_" "-")))]))
     (into {} (for [[k v] (:metrics r)] [(keyword (str "m." (name k))) v])))))

(defn- csv-cell [x]
  (let [s (cond (nil? x) "" (keyword? x) (name x) (map? x) (pr-str x) :else (str x))]
    (if (re-find #"[\",\n\r]" s) (str "\"" (str/replace s "\"" "\"\"") "\"") s)))

(defn- write-csv [path header-lines rows]
  (let [cols (vec (distinct (mapcat keys rows)))]
    (io/make-parents path)
    (with-open [w (io/writer path)]
      (doseq [l header-lines] (.write w (str "# " l "\n")))
      (.write w (str (str/join "," (map #(csv-cell (name %)) cols)) "\n"))
      (doseq [r rows] (.write w (str (str/join "," (map #(csv-cell (get r %)) cols)) "\n"))))))

(defn- out-path [spec] (or (:out spec) (str "out/" (:name spec) ".csv")))
(defn- part-path [spec i] (str/replace (out-path spec) #"\.csv$" (str ".part" i ".edn")))

(defn- read-part [path]
  (if (.exists (io/file path))
    (with-open [r (io/reader path)]
      (vec (keep #(try (edn/read-string %) (catch Exception _ nil)) (line-seq r))))
    []))

(defn- rss-mb
  "Resident set size of this process (JVM + embedded Python), in MB."
  []
  (try
    (let [l (first (filter #(str/starts-with? % "VmRSS")
                           (java.nio.file.Files/readAllLines (.toPath (io/file "/proc/self/status")))))]
      (quot (parse-long (re-find #"\d+" l)) 1024))
    (catch Exception _ -1)))

(defn run-shard
  "Run shard `i` of `n`, appending one EDN row per line to its part file."
  [spec i n {:keys [resume?]}]
  (let [runs (shard spec (expand spec) i n)
        path (part-path spec i)
        done (if resume? (set (map :run (read-part path))) #{})
        todo (remove #(done (:run %)) runs)]
    (io/make-parents path)
    (when-not resume? (spit path ""))
    (with-open [w (io/writer path :append true)]
      (doseq [[j params] (map-indexed vector todo)]
        (let [r (row spec params (run-one spec params))]
          (.write w (str (pr-str r) "\n"))
          (.flush w)
          (binding [*out* *err*]
            (println (format "[%s %d/%d shard %d] run %d %s -> %s/%s %s%.0f ms, rss %d MB"
                             (:name spec) (inc j) (count todo) i (:run params)
                             (pr-str (dissoc params :run)) (:successes r) (:trials r)
                             (if (:error r) (str "error: " (:error r) " ") "")
                             (double (:wall-ms r)) (rss-mb))))
          (System/gc)
          (h/collect!))))))

;; ---------------------------------------------------------------------------
;; process pool

(defn- mem-available-gb []
  (try
    (let [lines (java.nio.file.Files/readAllLines (.toPath (io/file "/proc/meminfo")))
          l (first (filter #(str/starts-with? % "MemAvailable") lines))]
      (/ (parse-long (re-find #"\d+" l)) 1048576.0))
    (catch Exception _ 16.0)))

(def ^:private gb-per-worker 2.0)
(def ^:private reserve-gb 10.0)
(def ^:private max-workers 2)

(defn- slice-headroom-gb
  "Headroom of the cgroup slice this process's scope sits in (a mem-scope
  wrapper puts JVMs under one aggregate cap), or nil outside one."
  []
  (try
    (let [cg (-> (java.nio.file.Files/readAllLines (.toPath (io/file "/proc/self/cgroup")))
                 first (str/split #":" 3) last)
          dir (.getParentFile (io/file (str "/sys/fs/cgroup" cg)))
          rd #(str/trim (slurp (io/file dir %)))
          mx (rd "memory.max")]
      (when-not (= "max" mx)
        (/ (- (parse-long mx) (parse-long (rd "memory.current"))) 1073741824.0)))
    (catch Exception _ nil)))

(defn- pool-size
  "Workers the machine can afford: each JVM + Python ≈ 2 GB, 10 GB of RAM
  stay free, and the aggregate cgroup cap keeps 2 GB of headroom."
  [wanted]
  (let [avail (mem-available-gb)
        fit (long (Math/floor (/ (- avail reserve-gb) gb-per-worker)))
        fit (if-let [h (slice-headroom-gb)] (min fit (long (Math/floor (/ (- h 2.0) gb-per-worker)))) fit)]
    (max 1 (min wanted max-workers fit))))

(defn- spawn [spec-path i n resume?]
  (let [args (cond-> ["clojure" "-M:jvm:exp" spec-path "--shard" (str i) "--of" (str n)]
               resume? (conj "--resume"))
        log (io/file (str/replace (out-path (edn/read-string (slurp spec-path))) #"\.csv$" (str ".part" i ".log")))]
    (io/make-parents log)
    (-> (ProcessBuilder. ^java.util.List args)
        (.redirectErrorStream true)
        (.redirectOutput log)
        (.start))))

(defn- header [spec spec-path n]
  (let [{:keys [numpy blas]} (h/versions)]
    [(str "exp " (:name spec) "  spec " spec-path "  workers " n)
     (str "numpy " numpy)
     (str "blas " blas)
     (str "date " (java.time.Instant/now))]))

(defn run-spec
  "Run every shard (this process runs shard 0), then merge into one CSV."
  [spec-path {:keys [workers resume?]}]
  (let [spec (edn/read-string (slurp spec-path))
        n (pool-size (or workers (:workers spec) 1))
        runs (expand spec)
        _ (binding [*out* *err*]
            (println (format "%s: %d runs on %d worker(s), %.1f GB available, slice headroom %s GB"
                             (:name spec) (count runs) n (mem-available-gb)
                             (some->> (slice-headroom-gb) (format "%.1f")))))
        children (doall (for [i (range 1 n)] (spawn spec-path i n resume?)))]
    (run-shard spec 0 n {:resume? resume?})
    (doseq [^Process p children] (.waitFor p))
    (let [rows (sort-by :run (mapcat #(read-part (part-path spec %)) (range n)))
          rows (vals (into (sorted-map) (map (juxt :run identity) rows)))
          numpy (:numpy (h/versions))]
      (write-csv (out-path spec) (header spec spec-path n) (map #(assoc % :numpy numpy) rows))
      (doseq [i (range n)] (.delete (File. ^String (part-path spec i))))
      (binding [*out* *err*]
        (println (format "%s: wrote %d rows (%d expected) to %s"
                         (:name spec) (count rows) (count runs) (out-path spec)))))))

(defn -main [& args]
  (let [[spec-path & flags] args
        opt (fn [f] (some->> (drop-while #(not= f %) flags) second))
        resume? (some #{"--resume"} flags)]
    (if-let [i (opt "--shard")]
      (run-shard (edn/read-string (slurp spec-path)) (parse-long i) (parse-long (opt "--of"))
                 {:resume? resume?})
      (run-spec spec-path {:workers (some-> (opt "--workers") parse-long) :resume? resume?}))
    (shutdown-agents)
    (System/exit 0)))
