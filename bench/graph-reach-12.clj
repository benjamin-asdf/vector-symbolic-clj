;; Reachability by depth-first search on a graph stored as a map of sets
;; (node -> set of successors). Stresses map lookup, set membership and set
;; enumeration (explaining away), and conj onto a growing set. The graph-reach-*
;; variants differ only in the graph literal below: the nodes are 0..n-1, and
;; part of each graph is unreachable from node 0.

(def graph
  {0 #{1 5}
   1 #{2}
   2 #{3 6}
   3 #{1}
   4 #{8}
   5 #{6 9}
   6 #{}
   7 #{4 11}
   8 #{7}
   9 #{10 2}
   10 #{5}
   11 #{0}})

(defn visit [g seen x]
  (if (contains? seen x)
    seen
    (reduce (fn [s y] (visit g s y)) (conj seen x) (seq (get g x)))))

(visit graph #{} 0)
