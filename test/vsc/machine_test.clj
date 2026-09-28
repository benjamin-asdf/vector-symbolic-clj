(ns vsc.machine-test
  "Differential tests: every form of core_test, run on the vector CEK machine
  (vsc.machine/run), must give what the host evaluator (vsc.core/run) gives.
  Both run on the same core (same memories, same globals).

  The host evaluator recurses on the JVM stack, so its side of the
  comparison runs on a thread with a 512 MB stack; the machine runs on the
  test thread, or on a deliberately small stack. This namespace therefore
  passes without the :jvm alias's -Xss512m."
  (:require
   [clojure.test :refer [deftest is testing use-fixtures]]
   [vsc.core :as vsc]
   [vsc.machine :as machine]))

(use-fixtures :once (fn [t] (machine/init!) (t)))

(defn- on-stack
  "Run f on a fresh thread with a stack of `bytes`; [:ok value] or [:threw class]."
  [bytes f]
  (let [p (promise)
        t (Thread. nil #(deliver p (try [:ok (f)] (catch Throwable e [:threw (class e)])))
                   "vsc-stack" (long bytes))]
    (.start t)
    @p))

(defn- host [form]
  (let [[tag x] (on-stack (* 512 1024 1024) #(vsc/run form))]
    (if (= tag :ok) x (throw (ex-info (str "host threw " x) {:form form})))))

(defn- same
  "The machine agrees with the host evaluator on `form`."
  [form]
  (is (= (host form) (machine/run form)) (pr-str form)))

(deftest lisp-1-5-elementary-functions
  (doseq [form '[(cons (quote a) (quote (b c)))
                 (first (quote (a b c)))
                 (rest (quote (a b c)))
                 (= (quote a) (quote a))
                 (= (quote a) (quote b))
                 (symbol? (quote a))
                 (coll? (quote (a)))]]
    (same form)))

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
    (same form))
  (testing "maps and sets enumerate by explaining away"
    (is (= #{:a :b :c} (set (machine/run '(keys {:a 1 :b 2 :c 3})))))
    (is (= #{1 2 3} (set (machine/run '(vals {:a 1 :b 2 :c 3})))))))

(deftest arithmetic-is-binding
  (doseq [form '[(+ 40 2) (+ 1 2 3 4) (- 10 3) (- 3 10) (- 5) (inc 41) (dec 0)
                 (zero? 0) (zero? 1) (+ 400000 400000)
                 (zero? 5005) (= 0 5005) (= 5 5005) (+ 5000 5)]]
    (same form))
  (is (thrown? clojure.lang.ExceptionInfo (machine/run 900000))))

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
                 ((fn self [n] (if (zero? n) :done (self (dec n)))) 5)
                 ;; beyond core_test
                 (do) (and) (or) (let [] 7) (cond) (if false 1)]]
    (same form)))

(deftest programs
  (machine/run '(defn fact [n] (if (zero? n) 1 (* n (fact (dec n))))))
  (is (= 120 (machine/run '(fact 5))))
  (same '(fact 5))
  (machine/run '(defn fib [n] (cond (zero? n) 0 (zero? (dec n)) 1
                                    :else (+ (fib (dec n)) (fib (dec (dec n)))))))
  (is (= 21 (machine/run '(fib 8))))
  (machine/run '(defn even?* [n] (if (zero? n) true (odd?* (dec n)))))
  (machine/run '(defn odd?* [n] (if (zero? n) false (even?* (dec n)))))
  (same '[(even?* 6) (even?* 7)])
  (testing "the prelude, loaded by the machine itself"
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
      (same form)))
  (testing "redefinition replaces the global binding"
    (machine/run '(def x 1))
    (machine/run '(def x 2))
    (is (= 2 (machine/run 'x)))
    (same 'x)))

(deftest code-is-data-is-a-vector
  (same '(eval (list (quote +) 1 2)))
  (same '(eval (quote ((fn [x] [x x]) :y)))))

(deftest metacircular-interpreter
  (is (= 10 (machine/run-string (slurp "examples/metacircular.clj")))))

(deftest robustness
  (testing "structures survive noise as large as the signal, via cleanup"
    (is (= 'a (machine/run '(first (degrade (quote (a b c)) 100)))))
    (is (= 2 (machine/run '(get (degrade {:a 1 :b 2} 60) :b))))
    (is (= [1 2 3] (machine/run '(degrade [1 2 3] 100))))))

(deftest errors
  (is (thrown? clojure.lang.ExceptionInfo (machine/run 'no-such-symbol)))
  (is (thrown? clojure.lang.ExceptionInfo (machine/run '([1 2] 0)))))

;; ---------------------------------------------------------------------------
;; deep recursion: the host evaluator needs a stack frame chain per call, the
;; machine keeps its continuation in W and needs none

(def ^:private small-stack (* 256 1024))     ;; a quarter of the JVM default
(def ^:private default-stack (* 1024 1024))  ;; the JVM default on x86-64 Linux

(deftest tail-calls-run-in-constant-space
  (machine/run '(defn countdown [n] (if (zero? n) :done (countdown (dec n)))))
  (testing "the host evaluator overflows a default-size stack"
    (is (= [:threw StackOverflowError]
           (on-stack default-stack #(vsc/run '(countdown 1000))))))
  (testing "the machine runs the same program on a 256 KB stack"
    (is (= [:ok :done] (on-stack small-stack #(machine/run '(countdown 1000))))))
  (testing "mutual tail recursion"
    (machine/run '(defn ping [n] (if (zero? n) :ping (pong (dec n)))))
    (machine/run '(defn pong [n] (if (zero? n) :pong (ping (dec n)))))
    (is (= [:ok :pong] (on-stack small-stack #(machine/run '(ping 501)))))))

(deftest deep-non-tail-recursion-lives-in-vector-memory
  (machine/run '(defn sum-to [n] (if (zero? n) 0 (+ n (sum-to (dec n))))))
  (is (= [:threw StackOverflowError] (on-stack default-stack #(vsc/run '(sum-to 600)))))
  (is (= [:ok 180300] (on-stack small-stack #(machine/run '(sum-to 600))))))

(deftest ^:deep countdown-10000
  ;; about 10 minutes; run with VSC_DEEP=1
  (when (System/getenv "VSC_DEEP")
    (machine/run '(defn countdown [n] (if (zero? n) :done (countdown (dec n)))))
    (let [t0 (System/nanoTime)
          r (on-stack small-stack #(machine/run '(countdown 10000)))]
      (println "countdown 10000 on the machine:" r
               (format "%.0f s" (/ (- (System/nanoTime) t0) 1e9)) (vsc/stats))
      (is (= [:ok :done] r)))))
