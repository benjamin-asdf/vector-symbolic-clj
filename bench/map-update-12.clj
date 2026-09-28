;; An assoc/update loop on one map: n assocs build {i 2i+2}, then a second
;; pass updates every entry with inc. Every assoc/update rebuilds the map
;; trace by reading back all its entries, so this stresses map rebuilds and
;; map capacity. Values collide with keys (2 is both), which binding's
;; commutativity used to confuse. The map-update-* variants differ only in n.
;; n = 12 is the largest that is exact at D=2048 across seeds: at 16 and 20
;; some of the ~2n² lookups fail even at noise 0 (see bench/README.md), below
;; the enforced single-lookup capacity of 21.

(def n 12)

(defn fill [m i]
  (if (zero? i) m (fill (assoc m (dec i) (+ i i)) (dec i))))

(defn bump [m ks]
  (if (empty? ks) m (bump (update m (first ks) inc) (rest ks))))

(bump (fill {} n) (range n))
