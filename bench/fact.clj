;; Factorial: recursion depth and environment chains. `*` is the prelude's
;; repeated addition, so (fact n) also runs ~n²/2 nested additions.

(def n 5)

(defn fact [n] (if (zero? n) 1 (* n (fact (dec n)))))

(fact n)
