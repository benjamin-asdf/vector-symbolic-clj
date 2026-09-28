;; Cartesian closedness, test (4): an interpreter for a language, written in
;; the vector-symbolic dialect and run in vector space. Its programs, its
;; environments (VSA maps: Σ k⊗v) and its closures are all single vectors.

(defn m-eval [e env]
  (cond (number? e) e
        (symbol? e) (get env e)
        (= (first e) (quote lambda)) (list (quote closure) (nth e 1) (nth e 2) env)
        (= (first e) (quote if0)) (if (zero? (m-eval (nth e 1) env))
                                    (m-eval (nth e 2) env)
                                    (m-eval (nth e 3) env))
        (= (first e) (quote +)) (+ (m-eval (nth e 1) env) (m-eval (nth e 2) env))
        :else (m-apply (m-eval (first e) env) (m-eval (nth e 1) env))))

(defn m-apply [c arg]
  (m-eval (nth c 2) (assoc (nth c 3) (nth c 1) arg)))

;; currying
(m-eval (quote (((lambda x (lambda y (+ x y))) 3) 4)) {})

;; recursion by self-application: 4 + 3 + 2 + 1
(m-eval (quote (((lambda f (lambda n ((f f) n)))
                 (lambda self (lambda n (if0 n 0 (+ n ((self self) (+ n -1)))))))
                4))
        {})
