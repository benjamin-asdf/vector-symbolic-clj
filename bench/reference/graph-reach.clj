;; Real-Clojure equivalent of bench/graph-reach-<n>.clj: breadth-first search
;; with loop/recur (a different traversal from the dialect's DFS; the
;; reachable set is the same). `graph` comes from each variant's bench file.
(fn [{:keys [graph]}]
  (loop [frontier [0] seen #{}]
    (if (empty? frontier)
      seen
      (let [fresh (remove seen frontier)]
        (recur (distinct (mapcat graph fresh)) (into seen fresh))))))
