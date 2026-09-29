;; W2.1: level-synchronous breadth-first search, one vector per level.
;;
;; Run with vsc.worlds (docs/results-W.md): the frontier is ONE vector, the
;; superposition of the nodes at the current distance, and
;;
;;   (follow graph frontier)   the successors of every frontier node, read in
;;                             one lookup of the item memory (graph is a
;;                             relation: one row per node, R⊗node ↦ its
;;                             successors)
;;   (support ...)             re-binarise: the uniform superposition of what
;;                             was found (set semantics; path counts dropped)
;;   (without ... visited)     drop the visited nodes: one coefficient test per
;;                             candidate, visited itself is never read out
;;   (union visited next)      grow the visited set, again without a readout
;;
;; The graph is bench/graph-reach-12.clj with keyword nodes. That task walks
;; it depth-first, one node, one map lookup and one set membership at a time;
;; here each level costs one lookup whatever the frontier's size.

(def graph
  (relation [[:n0 [:n1 :n5]]
             [:n1 [:n2]]
             [:n2 [:n3 :n6]]
             [:n3 [:n1]]
             [:n4 [:n8]]
             [:n5 [:n6 :n9]]
             [:n6 []]
             [:n7 [:n4 :n11]]
             [:n8 [:n7]]
             [:n9 [:n10 :n2]]
             [:n10 [:n5]]
             [:n11 [:n0]]]))

;; the BFS levels from `frontier`: a list of lists of nodes
(defn bfs [frontier visited levels]
  (let [levels (cons (keys (worlds frontier)) levels)
        next (without (support (follow graph frontier)) visited)]
    (if (nil? next)
      (reverse levels)
      (bfs next (union visited next) levels))))

(bfs :n0 :n0 ())
