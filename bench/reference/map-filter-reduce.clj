;; Real-Clojure equivalent of bench/map-filter-reduce.clj; `n` comes from the
;; bench file.
(fn [{:keys [n]}]
  (let [doubled (map #(* 2 %) (range n))
        kept (filter #(< % n) doubled)]
    [doubled kept (reduce + 0 (map inc kept))]))
