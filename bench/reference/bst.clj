;; Real-Clojure equivalent of bench/bst.clj, with the host's own < and
;; [left key right] vectors rather than nested maps. `ks` and `probes` come
;; from the bench file.
(fn [{:keys [ks probes]}]
  (let [insert (fn insert [t k]
                 (cond (nil? t) [nil k nil]
                       (= k (t 1)) t
                       (< k (t 1)) (assoc t 0 (insert (t 0) k))
                       :else (assoc t 2 (insert (t 2) k))))
        depth (fn depth [t] (if (nil? t) 0 (inc (max (depth (t 0)) (depth (t 2))))))
        t (reduce insert nil ks)]
    [(sort (distinct ks)) (depth t) (map #(contains? (set ks) %) probes)]))
