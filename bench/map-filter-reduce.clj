;; map / filter / reduce from the prelude over (range n): list walking and
;; closures and a set passed as functions (a set is callable, so it serves as
;; the filter predicate). Keeps the doubled values that are still below n.

(def n 10)

(let [xs (range n)
      doubled (map (fn [x] (+ x x)) xs)
      kept (filter (set xs) doubled)]
  [doubled kept (reduce + 0 (map inc kept))])
