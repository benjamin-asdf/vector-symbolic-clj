;; Doubly recursive Fibonacci: many short-lived environment cells, about
;; 2·fib(n+1) calls in total.

(def n 8)

(defn fib [n]
  (cond (zero? n) 0
        (zero? (dec n)) 1
        :else (+ (fib (dec n)) (fib (dec (dec n))))))

(fib n)
