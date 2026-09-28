;; A binary search tree as nested maps {:k key :l left :r right}; a missing
;; child is nil. Inserts the keys in order, then reads the tree back by an
;; in-order walk, its depth, and lookups of present and absent keys. Stresses
;; deep nesting: every insert rebuilds the maps along one root-to-leaf path,
;; and the tree below is 4 maps deep. lt? is the Peano comparison from
;; insertion-sort (linear in the values).

(def ks '(0 3 1 2 4))
(def probes '(2 5))

(defn lt? [a b]
  (cond (zero? b) false
        (zero? a) true
        :else (lt? (dec a) (dec b))))

(defn insert [t k]
  (cond (nil? t) {:k k}
        (= k (:k t)) t
        (lt? k (:k t)) (assoc t :l (insert (:l t) k))
        :else (assoc t :r (insert (:r t) k))))

(defn has? [t k]
  (cond (nil? t) false
        (= k (:k t)) true
        (lt? k (:k t)) (has? (:l t) k)
        :else (has? (:r t) k)))

(defn walk [t acc]
  (if (nil? t) acc (walk (:l t) (cons (:k t) (walk (:r t) acc)))))

(defn depth [t]
  (if (nil? t)
    0
    (let [a (depth (:l t)) b (depth (:r t))]
      (inc (if (lt? a b) b a)))))

(let [t (reduce insert nil ks)]
  [(walk t ()) (depth t) (map (fn [k] (has? t k)) probes)])
