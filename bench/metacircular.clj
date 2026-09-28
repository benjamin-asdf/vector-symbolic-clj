;; A λ-calculus interpreter written in the dialect (adapted from
;; examples/metacircular.clj): the program under interpretation is data, its
;; environments are VSA maps and its closures are lists holding an env, so
;; this stresses program-as-data and nested closures, one interpreter level
;; deep. `k` is the argument of the self-applied recursive sum.

(def k 2)

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

[;; currying
 (m-eval (quote (((lambda x (lambda y (+ x y))) 3) 4)) {})
 ;; recursion by self-application: k + (k-1) + ... + 1
 (m-eval (list (quote ((lambda f (lambda n ((f f) n)))
                       (lambda self (lambda n (if0 n 0 (+ n ((self self) (+ n -1))))))))
               k)
         {})]
