;; Real-Clojure equivalent of bench/fib.clj (iterative, not the same
;; algorithm); `n` comes from the bench file.
(fn [{:keys [n]}]
  (first (nth (iterate (fn [[a b]] [b (+ a b)]) [0 1]) n)))
