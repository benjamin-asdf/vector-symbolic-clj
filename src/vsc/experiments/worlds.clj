(ns vsc.experiments.worlds
  "W3 task generators for vsc.experiments (specs/w-*.edn): superposition
  programs run on vsc.worlds, scored against exact answers computed on the
  host. Every task is :native, since it builds its machine with
  vsc.worlds/init! rather than vsc.core/init!."
  (:require
   [clojure.string :as str]
   [vsc.core :as vsc]
   [vsc.hdc :as h]
   [vsc.worlds :as w]))

(defn- dist [pairs]
  (reduce (fn [m [v p]] (update m v (fnil + 0.0) p)) {} pairs))

(defn- scored
  "Row fields for a readout `got` against exact `want`."
  [want got]
  (let [tv (w/tv-distance want got)
        same (= (set (keys want)) (set (keys got)))]
    {:successes (if (and same (< tv 0.05)) 1 0) :trials 1
     :value got :expected want
     :metrics {:tv tv :same-worlds (if same 1 0) :worlds (count got)}}))

(def ^:private budget
  "The runner budgets only non-native tasks; these set their own. Op noise
  can keep a split recursion from ever reaching its base case."
  {:max-ops 2000000 :max-rows 60000 :seconds 60})

(defn- counted
  "Run thunk f with op counting on and the budget set; its result merged
  with the op counts."
  [f]
  (vsc/reset-instruments!)
  (vsc/instrument! {:count-ops? true})
  (h/budget! (vsc/space) budget)
  (try
    (let [r (f)
          c (:counts (vsc/instruments))]
      (-> r
          (assoc :counts c)
          (update :metrics merge {:rows (get c :rows 0) :readouts (get c :readout 0)
                                  :superposed (get c :superposed 0)})))
    (finally
      (h/budget! (vsc/space) {}))))

(defn- attempt [trials f]
  (try (f)
       (catch Throwable e
         ;; a Python traceback shrinks to its last line
         (let [msg (or (ex-message e) (str (class e)))]
           {:successes 0 :trials trials
            :error (last (remove #(re-matches #"\s*" %) (str/split-lines msg)))}))))

;; ---------------------------------------------------------------------------
;; exactness of linear lifting vs op noise

(def ^:private lift-tasks
  {:dbl {:setup '[(defn dbl [n] (if (zero? n) 0 (+ 2 (dbl (dec n)))))]
         :form '(dbl (superpose {2 5 3 3 5 2}))
         :want {4 0.5 6 0.3 10 0.2}}
   :len {:setup '[(defn len [xs] (if (empty? xs) 0 (inc (len (rest xs)))))]
         :form '(len (superpose {(quote (a b)) 5 (quote (c d e)) 3 (quote (f g h i j)) 2}))
         :want {2 0.5 3 0.3 5 0.2}}
   :nth3 {:setup []
          :form '(first (rest (rest (rest (superpose {(quote (a b c d e)) 5
                                                      (quote (f g h i j)) 3
                                                      (quote (k l m n o)) 2})))))
          :want '{d 0.5 i 0.3 n 0.2}}
   :inc20 {:setup []
           :form (nth (iterate (fn [f] (list 'inc f)) '(superpose {1 5 2 3 3 2})) 20)
           :want {21 0.5 22 0.3 23 0.2}}
   :let3 {:setup []
          :form '(let [x (superpose {:a 5 :b 3 :c 2})]
                   (let [y x] (let [z y] z)))
          :want {:a 0.5 :b 0.3 :c 0.2}}})

(defn lift-noise
  "One superposition program under op noise σ (switched on after setup)."
  [{:keys [dim seed lift op-noise]}]
  (let [{:keys [setup form want]} (lift-tasks lift)]
    {:native
     (fn []
       (attempt 1
                (fn []
                  (w/init! {:dim dim :seed seed})
                  (doseq [f setup] (w/run f))
                  (vsc/set-knobs! {:op-noise op-noise})
                  (counted #(scored want (w/run-worlds form))))))}))

;; ---------------------------------------------------------------------------
;; BFS vs graph size and D

(defn random-graph
  "n keyword nodes, each with `deg` distinct random successors (seeded)."
  [n deg seed]
  (let [rnd (java.util.Random. (long seed))
        node #(keyword (str "n" %))]
    (into {} (for [i (range n)]
               [(node i) (->> (repeatedly #(.nextInt rnd n)) (remove #{i}) distinct
                              (take deg) (mapv node))]))))

(defn bfs-levels [graph start]
  (loop [frontier #{start} seen #{start} levels []]
    (if (empty? frontier)
      levels
      (let [nxt (set (remove seen (mapcat graph frontier)))]
        (recur nxt (into seen nxt) (conj levels frontier))))))

(def bfs-program
  '(defn bfs [graph frontier visited levels]
     (let [levels (cons (keys (worlds frontier)) levels)
           next (without (support (follow graph frontier)) visited)]
       (if (nil? next)
         (reverse levels)
         (bfs graph next (union visited next) levels)))))

(defn bfs
  "Level-synchronous BFS on a random graph; exact means every level right."
  [{:keys [dim seed n deg] :or {deg 2}}]
  {:native
   (fn []
     (let [g (random-graph n deg (+ (* 7919 n) seed))
           want (bfs-levels g :n0)
           reach (reduce + (map count want))]
       (attempt 1
                (fn []
                  (w/init! {:dim dim :seed seed})
                  (w/run bfs-program)
                  (w/run (list 'def 'g (list 'relation (vec (for [[k vs] g] [k vs])))))
                  (counted
                   (fn []
                     (let [got (mapv set (w/run '(bfs g :n0 :n0 ())))
                           level-of (fn [ls] (into {} (for [[d l] (map-indexed vector ls) x l] [x d])))
                           lw (level-of want) lg (level-of got)
                           right (count (filter #(= (lw %) (lg %)) (keys lw)))
                           wrong (count (remove lw (keys lg)))]
                       {:successes (if (= want got) 1 0) :trials 1
                        :metrics {:node-acc (/ (double right) (max 1 (+ (count lw) wrong)))
                                  :reachable reach :levels (count want)
                                  :peak-frontier (apply max (map count want))}})))))))})

;; ---------------------------------------------------------------------------
;; demos vs D

(defn- exact-dice [faces]
  (let [die (range 1 (inc faces))]
    (dist (for [a die b die] [(+ a b) (/ 1.0 (* faces faces))]))))

(defn- exact-sprinkler []
  (let [p (fn [q x] (if x q (- 1.0 q)))
        pw {[true true] 0.99 [true false] 0.8 [false true] 0.9 [false false] 0.0}
        joint (for [r [true false] s [true false] wt [true false]
                    :let [pr (* (p 0.2 r) (p ({true 0.1 false 0.4} r) s) (p (pw [r s]) wt))]
                    :when (pos? pr)]
                [[r s wt] pr])
        wet (filter (fn [[[_ _ wt] _]] wt) joint)
        z (reduce + (map second wet))]
    (dist (for [[[r] q] wet] [r (/ q z)]))))

(defn demo
  "A W2 demo at dimension D: dice sums (d4, d6), the sprinkler query, the
  call-time parallel map."
  [{:keys [dim seed demo]}]
  {:native
   (fn []
     (attempt 1
              (fn []
                (w/init! {:dim dim :seed seed})
                (counted
                 (fn []
                   (case demo
                     :dice4 (scored (exact-dice 4) (w/run-worlds '(+ (amb 1 2 3 4) (amb 1 2 3 4))))
                     :dice6 (scored (exact-dice 6) (w/run-worlds '(+ (amb 1 2 3 4 5 6) (amb 1 2 3 4 5 6))))
                     :sprinkler (do (w/run-string (slurp "examples/worlds/sprinkler.clj"))
                                    (scored (exact-sprinkler) (w/run-worlds 'rain-given-wet)))
                     :pmap (do (w/run-string (slurp "examples/worlds/parallel-map.clj"))
                               (scored {'(2 3 4) 0.5 '(11 21 31) 0.5} (w/run-worlds 'paired)))
                     :bfs12 (let [levels (w/run-string (slurp "examples/worlds/bfs.clj"))
                                  ok (= [#{:n0} #{:n1 :n5} #{:n2 :n6 :n9} #{:n3 :n10}] (mapv set levels))]
                              {:successes (if ok 1 0) :trials 1})))))))})
