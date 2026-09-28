;; Real-Clojure equivalent of bench/fact.clj; `n` comes from the bench file.
(fn [{:keys [n]}]
  (reduce * 1 (range 1 (inc n))))
