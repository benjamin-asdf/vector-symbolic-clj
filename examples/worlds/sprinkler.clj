;; W2.3b: a small Bayesian network, queried by post-selection.
;;
;; Run with vsc.worlds (docs/results-W.md). Rain makes the sprinkler less
;; likely, either makes the grass wet. P(rain | wet)?
;;
;; The weights inside the dialect are integers (there are no floats), so a
;; probability is written in percent: (flip 20) is true with weight 20.

(defn flip [p] (superpose {true p false (- 100 p)}))

;; The joint distribution, by call-time choice: for-worlds enumerates each
;; variable, so the later ones see one world of the earlier ones.
(def joint
  (for-worlds [rain (flip 20)
               sprinkler (if rain (flip 10) (flip 40))
               wet (cond (and rain sprinkler) (flip 99)
                         rain (flip 80)
                         sprinkler (flip 90)
                         :else false)]
              [rain sprinkler wet]))

;; P(rain | wet): keep the wet worlds, look at rain. first lifts linearly.
(def rain-given-wet (first (assert (fn [w] (nth w 2)) joint)))

;; The same network with let instead of for-worlds is wrong, and this is
;; the run-time choice semantics at work: the split if gives the wet branch
;; only the wet worlds of `wet`, but `rain` is a separate superposition
;; whose correlation with `wet` is gone, so the answer is the prior P(rain).
(def rain-given-wet-naive
  (let [rain (flip 20)
        sprinkler (if rain (flip 10) (flip 40))
        wet (cond (and rain sprinkler) (flip 99)
                  rain (flip 80)
                  sprinkler (flip 90)
                  :else false)]
    (assert (fn [x] (not (= x :dry))) (if wet rain :dry))))

rain-given-wet
