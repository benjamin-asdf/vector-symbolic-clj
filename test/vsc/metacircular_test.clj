(ns vsc.metacircular-test
  "Differential tests for S1: every form is evaluated by the host `veval` and
  by `vsc-eval` (resources/vsc/eval.clj, written in the dialect) running on
  top of it; both must agree. The prelude is loaded *through* vsc-eval, so
  map, filter, reduce, ... used at the S1 level are S1 closures.

  S1 is two orders of magnitude slower than the host (docs/results-S1.md), so
  the forms are small, and each deftest starts from a fresh machine: cleanup
  cost grows with the size of M, and M only grows.

  `s1-smoke` runs with the default suite (about 1.5 minutes for the whole default suite). The full
  differential set is tagged ^:slow (20-30 minutes) and is excluded from
  `clojure -M:jvm:test`; run it with `clojure -M:jvm:test-s1`. Set
  VSC_S1_TIMING=1 to print the S1 wall time and |M| per form."
  (:require
   [clojure.java.io :as io]
   [clojure.test :refer [deftest is testing]]
   [vsc.core :as vsc]))

(def ^:private programs
  '[(defn fact [n] (if (zero? n) 1 (* n (fact (dec n)))))
    (defn fib [n] (cond (zero? n) 0 (zero? (dec n)) 1
                        :else (+ (fib (dec n)) (fib (dec (dec n))))))
    (defn even?* [n] (if (zero? n) true (odd?* (dec n))))
    (defn odd?* [n] (if (zero? n) false (even?* (dec n))))])

(defn- s1-form [form] (list 'vsc-eval (list 'quote form) ()))

(defn- s1
  "Evaluate host form `form` one level up, in vsc-eval, and decode."
  [form]
  (vsc/run (s1-form form)))

(defn- fresh!
  "A new machine with the S1 evaluator loaded at the host level, the prelude
  loaded through it, and `defs` defined at both levels."
  [defs]
  (vsc/init!)
  (vsc/run-string (slurp (io/resource "vsc/eval.clj")))
  (vsc/run (list 'vsc-run-all
                 (list 'quote (vsc/read-forms (slurp (io/resource "vsc/prelude.clj"))))))
  (doseq [d defs]
    (vsc/run d)
    (s1 d)))

(def ^:private timing? (some? (System/getenv "VSC_S1_TIMING")))

(defn- same-as-host [form]
  (let [host (vsc/run form)
        t0 (System/nanoTime)
        up (s1 form)]
    (when timing?
      (println (format "%8.1fs  |M|=%5d  %s" (/ (- (System/nanoTime) t0) 1e9)
                       (:traces (vsc/stats)) (pr-str form))))
    (is (= host up) (pr-str form))))

(deftest s1-smoke
  (fresh! (take 1 programs))
  (doseq [form '[(first (quote (a b c)))
                 {:a (+ 1 1) :b [1 2]}
                 (#{1 2} 2)
                 (let [a 1 b (+ a 1)] [a b])
                 (or nil false)
                 ((fn [a & xs] [a xs]) 1 2 3)
                 (fact 2)
                 (map inc [1])]]
    (same-as-host form))
  (is (= :vsc/closure (s1 '(first map))) "map is an S1 closure"))

(deftest ^:slow s1-lisp-and-clojure-data
  (fresh! [])
  (doseq [form '[(cons (quote a) (quote (b c)))
                 (first (quote (a b c)))
                 (rest (quote (a b c)))
                 (= (quote a) (quote b))
                 (symbol? (quote a))
                 (coll? (quote (a)))
                 [1 2 (+ 1 2)]
                 (quote (1 (2 (3 (4)))))
                 {:a 1 :b [1 2] :c {:d "e"}}
                 #{:x :y :z}
                 (:name {:name "Ada" :lang :clojure})
                 (get {:a 1} :z 7)
                 ({:a 1} :a)
                 (#{1 2} 2)
                 (assoc {:a 1} :b 2)
                 (dissoc {:a 1 :b 2} :a)
                 (contains? #{:x :y :z} :w)
                 (= {:a 1 :b 2} {:b 2 :a 1})
                 (= [1 2] (quote (1 2)))
                 (count {:a 1 :b 2})
                 (nth [:a :b :c] 2)
                 (conj [1 2] 3)
                 (conj {:a 1} [:b 2])
                 (empty? [])
                 (seq [])
                 (next [1])
                 (keyword? :k)
                 (string? "s")
                 (+ 1 2 3 4) (- 3 10) (- 5) (dec 0) (zero? 5005)]]
    (same-as-host form))
  (testing "maps and sets enumerate by explaining away"
    (is (= #{:a :b :c} (set (s1 '(keys {:a 1 :b 2 :c 3})))))))

(deftest ^:slow s1-special-forms
  (fresh! [])
  (doseq [form '[(if true 1 2) (if false 1 2) (if nil 1) (do 1 2 3)
                 (let [a 1 b (+ a 1)] [a b])
                 (let [x 1] (let [x 2] x))
                 (cond false 1 nil 2 :else 3)
                 (and 1 2) (and 1 nil 2) (and) (or nil false 3) (or nil false) (or)
                 ((fn [x y] [y x]) 1 2)
                 ((fn [& xs] xs) 1 2 3)
                 ((fn [a & xs] [a xs]) 1 2 3)
                 (((fn [a] (fn [b] (+ a b))) 3) 4)
                 ((fn self [n] (if (zero? n) :done (self (dec n)))) 2)
                 (eval (list (quote +) 1 2))
                 (eval (quote ((fn [x] [x x]) :y)))
                 (fn? inc) (fn? (fn [x] x)) (fn? :k)]]
    (same-as-host form))
  (testing "def stays inside the level"
    (is (= 'x (s1 '(def x 1))))
    (s1 '(def x 2))
    (is (= 2 (s1 'x)))
    (is (thrown? clojure.lang.ExceptionInfo (vsc/run 'x)) "the host never saw x")))

(deftest ^:slow s1-programs
  (fresh! programs)
  (doseq [form '[(fact 4) (fib 5) [(even?* 3) (odd?* 3)]]]
    (same-as-host form)))

(deftest ^:slow s1-prelude
  (fresh! [])
  (testing "the prelude, defined inside the level"
    (is (= :vsc/closure (s1 '(first map))) "map is an S1 closure")
    (doseq [form '[(map inc [1 2])
                   (filter (fn [x] (contains? #{2} x)) (range 3))
                   (reduce + 0 [1 2 3])
                   (reverse (quote (1 2)))
                   (concat [1] [2])
                   (map :k [{:k 1}])
                   ((comp inc inc) 1)
                   ((partial + 10) 5)
                   (update {:n 1} :n inc)
                   (apply + 1 2 [3 4])
                   (* 3 2)]]
      (same-as-host form))))

(deftest ^:slow s1-three-levels
  (testing "the λ-calculus interpreter, defined in and run by vsc-eval"
    (let [[m-eval m-apply] (vsc/read-forms (slurp "examples/metacircular.clj"))]
      (fresh! [m-eval m-apply])
      (same-as-host '(m-eval (quote (((lambda x (lambda y (+ x y))) 3) 4)) {})))))
