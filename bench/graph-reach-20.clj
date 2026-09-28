;; Reachability by depth-first search on a graph stored as a map of sets
;; (node -> set of successors). Stresses map lookup, set membership and set
;; enumeration (explaining away), and conj onto a growing set. The graph-reach-*
;; variants differ only in the graph literal below: the nodes are 0..n-1, and
;; part of each graph is unreachable from node 0. At 20 nodes the graph map is
;; one entry short of the D=2048 map capacity (21).

(def graph
  {0 #{1 2}
   1 #{3 4}
   2 #{5}
   3 #{6 0}
   4 #{7}
   5 #{8 9}
   6 #{}
   7 #{10 4}
   8 #{11}
   9 #{12 2}
   10 #{13}
   11 #{14 15}
   12 #{}
   13 #{7}
   14 #{16}
   15 #{14}
   16 #{11}
   17 #{18 0}
   18 #{19}
   19 #{17}})

(defn visit [g seen x]
  (if (contains? seen x)
    seen
    (reduce (fn [s y] (visit g s y)) (conj seen x) (seq (get g x)))))

(visit graph #{} 0)
