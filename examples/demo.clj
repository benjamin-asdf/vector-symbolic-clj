;; Lisp 1.5's five elementary functions, as vector operations
(cons 'a '(b c))
(first '(a b c))
(rest '(a b c))
(= 'a 'a)
(symbol? 'a)

;; Clojure data: vectors, maps, sets, keywords
[1 2 (+ 1 2)]
(:name {:name "Ada" :lang :clojure})
(get {:a 1 :b 2} :b)
(assoc {:a 1} :b 2)
(contains? #{:x :y :z} :y)
(= {:a 1 :b 2} {:b 2 :a 1})

;; arithmetic is binding: n = B^n
(+ 40 2)
(- 10 3)

;; closures, recursion, higher-order functions
(defn fact [n] (if (zero? n) 1 (* n (fact (dec n)))))
(fact 5)
(defn fib [n] (cond (zero? n) 0 (zero? (dec n)) 1 :else (+ (fib (dec n)) (fib (dec (dec n))))))
(fib 10)
(map (fn [x] (* x x)) (range 6))
(let [add (fn [a] (fn [b] (+ a b)))] ((add 3) 4))
(reduce + 0 [1 2 3 4])
(map :k [{:k 1} {:k 2}])

;; code is data is a vector
(eval (list '+ 1 2))

;; robustness: add noise as large as the signal, then read the structure back
(first (degrade '(a b c) 100))
(get (degrade {:a 1 :b 2} 60) :b)
