;; W2.3a: weights as probabilities. Dice, sums and conditioning.
;;
;; Run with vsc.worlds at D ≥ 4096 (docs/results-W.md): the seven integer
;; worlds of a two-dice sum are past the integer capacity of D = 2048.
;;
;; Run-time choice is exactly right for independent draws: every call of
;; (die) is its own superposition, and + binds them, so the sum is the
;; convolution of the two distributions, in one bind. Conditioning needs to
;; know which die gave which sum, so it enumerates the joint worlds with
;; for-worlds and post-selects with assert.

(defn die [] (amb 1 2 3 4))

;; {2 1/16, 3 2/16, 4 3/16, 5 4/16, 6 3/16, 7 2/16, 8 1/16}
(def two (+ (die) (die)))

;; the joint worlds [first-die sum], 16 of them
(def rolls (for-worlds [a (die) b (die)] [a (+ a b)]))

;; P(first die | sum ≥ 6) = {2 1/6, 3 2/6, 4 3/6}
(def first-given-high
  (first (assert (fn [r] (contains? #{6 7 8} (nth r 1))) rolls)))

two
