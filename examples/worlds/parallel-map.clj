;; W2.2: (map f (amb xs ys)), map over a superposition of lists.
;;
;; Run with vsc.worlds (docs/results-W.md). The prelude's map is evaluated
;; ONCE for both worlds: (empty? xs) is false in both, first and rest lift
;; linearly through the memory, inc lifts through binding, so the recursion
;; runs three levels, not twice three. What it cannot keep is which element
;; came from which list: every cons chooses its elements on its own
;; (run-time choice), so the result is the product of the per-position
;; distributions, 2³ = 8 worlds, each with weight 1/8. Every per-position
;; marginal is exact.
;;
;; for-worlds is the explicit call-time choice: it enumerates the worlds and
;; runs map once per world, so the lists stay whole.

(def xss (amb [1 2 3] [10 20 30]))

;; one run of map for both worlds; 8 worlds of weight 1/8
(def mapped (map inc xss))

;; per-position marginals are exact: {3 1/2, 21 1/2}
(def second-elements (first (rest mapped)))

;; lists of different lengths leave the recursion at different depths: the
;; superposed (empty? xs) splits them between the two branches, and the
;; lengths come out right, {2 1/2, 4 1/2}
(def lengths (count (map inc (amb [1 2] [10 20 30 40]))))

;; call-time choice: {(2 3 4) 1/2, (11 21 31) 1/2}
(def paired (for-worlds [xs xss] (map inc xs)))

paired
