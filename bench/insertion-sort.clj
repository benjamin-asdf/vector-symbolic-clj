;; Insertion sort of a list of naturals. There is no `<` primitive, so lt? is
;; written in the dialect by decrementing both sides at once (Peano style):
;; (lt? a b) costs min(a, b) + 1 recursive calls, i.e. it is linear in the
;; values, not constant. Stresses long runs and many traces.

(def xs '(4 1 6 0 5 2 3))

(defn lt? [a b]
  (cond (zero? b) false
        (zero? a) true
        :else (lt? (dec a) (dec b))))

(defn insert [x ys]
  (cond (empty? ys) (list x)
        (lt? (first ys) x) (cons (first ys) (insert x (rest ys)))
        :else (cons x ys)))

(defn isort [ys] (reduce (fn [acc y] (insert y acc)) () ys))

(isort xs)
