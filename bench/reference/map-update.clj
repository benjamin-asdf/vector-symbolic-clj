;; Real-Clojure equivalent of bench/map-update-<n>.clj; `n` comes from each
;; variant's bench file.
(fn [{:keys [n]}]
  (let [m (reduce (fn [m i] (assoc m i (* 2 (inc i)))) {} (reverse (range n)))]
    (reduce (fn [m k] (update m k inc)) m (range n))))
