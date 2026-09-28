(ns vsc.core-test
  (:require
   [clojure.test :refer [deftest is testing use-fixtures]]
   [vsc.core :as vsc]))

(use-fixtures :once (fn [t] (vsc/init!) (t)))

(defn- same-as-clojure
  "Evaluate `form` in real Clojure and in vector space; both must agree."
  [form]
  (is (= (eval form) (vsc/run form)) (pr-str form)))

(deftest lisp-1-5-elementary-functions
  (doseq [form '[(cons (quote a) (quote (b c)))
                 (first (quote (a b c)))
                 (rest (quote (a b c)))
                 (= (quote a) (quote a))
                 (= (quote a) (quote b))
                 (symbol? (quote a))
                 (coll? (quote (a)))]]
    (same-as-clojure form)))

(deftest clojure-data
  (doseq [form '[[1 2 (+ 1 2)]
                 (quote (1 (2 (3 (4)))))
                 {:a 1 :b [1 2] :c {:d "e"}}
                 #{:x :y :z}
                 (:name {:name "Ada" :lang :clojure})
                 (get {:a 1 :b 2} :b)
                 (get {:a 1} :z)
                 (get {:a 1} :z 7)
                 ({:a 1} :a)
                 (#{1 2} 2)
                 (assoc {:a 1} :b 2)
                 (assoc {:a 1} :a 2)
                 (dissoc {:a 1 :b 2} :a)
                 (contains? #{:x :y :z} :y)
                 (contains? #{:x :y :z} :w)
                 (contains? {:a 1} :a)
                 (= {:a 1 :b 2} {:b 2 :a 1})
                 (= #{1 2 3} #{3 2 1})
                 (= [1 2] (quote (1 2)))
                 (count [1 2 3 4])
                 (count {:a 1 :b 2})
                 (nth [:a :b :c] 2)
                 (conj [1 2] 3)
                 (conj (quote (2 3)) 1)
                 (conj #{1} 2)
                 (conj {:a 1} [:b 2])
                 (vec (quote (1 2)))
                 (empty? [])
                 (empty? ())
                 (empty? [1])
                 (nil? nil)
                 (seq [])
                 (first [])
                 (first nil)
                 (next [1])
                 (keyword? :k)
                 (string? "s")
                 (number? 3)
                 (map? {})
                 (set? #{})
                 (vector? [])
                 (list? ())]]
    (same-as-clojure form))
  (testing "maps and sets enumerate by explaining away"
    (is (= #{:a :b :c} (set (vsc/run '(keys {:a 1 :b 2 :c 3})))))
    (is (= #{1 2 3} (set (vsc/run '(vals {:a 1 :b 2 :c 3})))))))

(deftest map-values-that-are-also-keys
  ;; binding commutes: without value pointers, the entry 1⊗2 answered the
  ;; probe 2 with 1, found by the bench task map-update
  (doseq [form '[(get {1 2 2 3} 2)
                 (get {1 2 2 3} 1)
                 (get {1 1 2 3} 2)
                 (get {:a :b :b :c} :b)
                 ({:x :y :y :z :z :x} :y)
                 (update (update {0 0 1 1 2 4} 0 inc) 1 inc)]]
    (same-as-clojure form))
  (is (= #{2 3} (set (vsc/run '(vals {1 2 2 3}))))))

(deftest arithmetic-is-binding
  (doseq [form '[(+ 40 2) (+ 1 2 3 4) (- 10 3) (- 3 10) (- 5) (inc 41) (dec 0)
                 (zero? 0) (zero? 1) (+ 400000 400000)
                 ;; 5005 = 5·7·11·13 shares four residue bands with 0
                 (zero? 5005) (= 0 5005) (= 5 5005) (+ 5000 5)]]
    (same-as-clojure form))
  (is (thrown? clojure.lang.ExceptionInfo (vsc/run 900000))))

(deftest special-forms
  (doseq [form '[(if true 1 2) (if false 1 2) (if nil 1) (do 1 2 3)
                 (let [a 1 b (+ a 1)] [a b])
                 (let [x 1] (let [x 2] x))
                 (cond false 1 nil 2 :else 3)
                 (and 1 2) (and 1 nil 2) (or nil false 3) (or nil false)
                 ((fn [x y] [y x]) 1 2)
                 ((fn [& xs] xs) 1 2 3)
                 ((fn [a & xs] [a xs]) 1 2 3)
                 (((fn [a] (fn [b] (+ a b))) 3) 4)
                 ((fn self [n] (if (zero? n) :done (self (dec n)))) 5)]]
    (same-as-clojure form)))

(deftest programs
  (vsc/run '(defn fact [n] (if (zero? n) 1 (* n (fact (dec n))))))
  (is (= 120 (vsc/run '(fact 5))))
  (vsc/run '(defn fib [n] (cond (zero? n) 0 (zero? (dec n)) 1
                                :else (+ (fib (dec n)) (fib (dec (dec n)))))))
  (is (= 21 (vsc/run '(fib 8))))
  (vsc/run '(defn even?* [n] (if (zero? n) true (odd?* (dec n)))))
  (vsc/run '(defn odd?* [n] (if (zero? n) false (even?* (dec n)))))
  (is (= [true false] (vsc/run '[(even?* 6) (even?* 7)])))
  (testing "the prelude, itself written in the dialect"
    (doseq [form '[(map inc [1 2 3])
                   (filter (fn [x] (contains? #{2 4} x)) (range 6))
                   (reduce + 0 [1 2 3 4])
                   (reverse (quote (1 2 3)))
                   (concat [1 2] [3])
                   (range 5)
                   (map :k [{:k 1} {:k 2}])
                   ((comp inc inc) 1)
                   ((partial + 10) 5)
                   (update {:n 1} :n inc)
                   (apply + 1 2 [3 4])]]
      (same-as-clojure form)))
  (testing "redefinition replaces the global binding"
    (vsc/run '(def x 1))
    (vsc/run '(def x 2))
    (is (= 2 (vsc/run 'x)))))

(deftest code-is-data-is-a-vector
  (same-as-clojure '(eval (list (quote +) 1 2)))
  (same-as-clojure '(eval (quote ((fn [x] [x x]) :y)))))

(deftest metacircular-interpreter
  (is (= 10 (vsc/run-string (slurp "examples/metacircular.clj")))))

(deftest robustness
  (testing "structures survive noise as large as the signal, via cleanup"
    (is (= 'a (vsc/run '(first (degrade (quote (a b c)) 100)))))
    (is (= 2 (vsc/run '(get (degrade {:a 1 :b 2} 60) :b))))
    (is (= [1 2 3] (vsc/run '(degrade [1 2 3] 100))))))
