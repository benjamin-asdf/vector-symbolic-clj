(ns vsc.worlds-test
  "W1: superposition programming (vsc.worlds on the :proj memory). The
  acceptance tests of docs/W0-superposition-design.md, section 6, plus the
  rest of the API."
  (:require
   [clojure.test :refer [deftest is testing use-fixtures]]
   [vsc.core :as vsc]
   [vsc.hdc :as h]
   [vsc.worlds :as w]))

(use-fixtures :once (fn [t] (w/init!) (t)))

(defn- close?
  "Host distribution `got` is within TV `tol` of `want`, with the same worlds."
  [want got tol]
  (and (= (set (keys want)) (set (keys got)))
       (<= (w/tv-distance want got) tol)))

(defn- worlds [form] (w/run-worlds form))

(deftest proj-is-the-codebook-on-single-worlds
  ;; bit for bit: the same programs leave the same rows in every memory
  (let [forms '[(defn fact [n] (if (zero? n) 1 (* n (fact (dec n)))))
                (fact 4)
                {:a 1 :b [1 2] :c {:d "e"}}
                (keys {:a 1 :b 2 :c 3})
                #{1 2 3}
                (let [x 5 y (quote (a b c))] (cons x y))
                (first (degrade (quote (a b c)) 100))]
        fingerprint (fn [opts]
                      (vsc/init! opts)
                      [(mapv vsc/run forms) (vsc/digest)])]
    (is (= (fingerprint {:memory :codebook}) (fingerprint {:memory :proj}))))
  (w/init!))

(deftest arithmetic-lifts-exactly
  (is (close? {11 0.5 12 0.5} (worlds '(+ (amb 1 2) 10)) 1e-9))
  (is (close? {11 0.3 12 0.7} (worlds '(+ (superpose {1 3 2 7}) 10)) 1e-9))
  (is (close? {0 0.5 1 0.5} (worlds '(dec (amb 1 2))) 1e-9))
  (testing "the printer reads integer worlds in a bounded range, no chimeras"
    (is (= (tagged-literal 'worlds {11 0.5 12 0.5}) (w/run '(+ (amb 1 2) 10))))))

(deftest plain-printer-has-no-chimeras
  ;; without vsc.worlds the codebook dialect used to print 646657
  (vsc/init! {:prelude? false})
  (is (= (tagged-literal 'worlds {11 0.5 12 0.5}) (vsc/run '(+ (bundle 1 2) 10))))
  (is (= 5 (vsc/run '(degrade 5 30))))
  (is (= 800000 (vsc/run '(+ 400000 400000))))
  (w/init!))

(deftest destructors-lift-and-keep-weights
  (is (close? {'a 0.5 'c 0.5} (worlds '(first (amb (quote (a b)) (quote (c d))))) 0.02))
  (is (close? {'(b) 0.5 '(d) 0.5} (worlds '(rest (amb (quote (a b)) (quote (c d))))) 0.02))
  (is (close? {'b 0.25 'd 0.75}
              (worlds '(first (rest (superpose {(quote (a b)) 1 (quote (c d)) 3})))) 0.02))
  (testing "worlds of different kinds, or empty ones, are enumerated"
    (is (close? {'a 0.5 nil 0.5} (worlds '(first (amb (quote (a)) ()))) 0.02)))
  (testing "get with a superposed key enumerates (exact weights)"
    (is (close? {1 0.5 2 0.5} (worlds '(get {:a 1 :b 2 :c 3} (amb :a :b))) 1e-6)))
  (testing "a relation reads every world of the key in one lookup"
    (is (close? {:x 0.5 :y 0.25 :z 0.25}
                (worlds '(follow (relation {:a #{:x} :b #{:y :z}}) (amb :a :b))) 0.02))
    (is (close? {:x 0.5 :y 0.25 :z 0.25}
                (worlds '(follow (relation [[:a [:x]] [:b [:y :z]]]) (amb :a :b))) 0.02))
    (is (nil? (w/run '(follow (relation {:a #{:x}}) :q))))))

(deftest environment-does-not-snap
  (is (close? {:a 0.5 :b 0.5} (worlds '(let [x (amb :a :b)] x)) 0.02))
  (is (close? {1 0.5 2 0.5} (worlds '(let [x (amb 1 2)] x)) 0.02))
  (is (close? {2 0.5 3 0.3 5 0.2} (worlds '((fn [n] n) (superpose {2 5 3 3 5 2}))) 0.02))
  (is (close? {:a 0.5 :b 0.5} (worlds '(((fn [x] (fn [] x)) (amb :a :b)))) 0.02)))

(deftest constructors-enumerate
  (is (close? {'(1) 0.5 '(2) 0.5} (worlds '(cons (amb 1 2) ())) 1e-6))
  (is (close? {[1 :x] 0.5 [2 :x] 0.5} (worlds '[(amb 1 2) :x]) 1e-6))
  (is (close? {{:k 1} 0.5 {:k 2} 0.5} (worlds '{:k (amb 1 2)}) 1e-6))
  (testing "equal worlds merge and their weights add"
    (is (close? {true 1.0} (worlds '(some? (amb 1 2))) 1e-6))))

(deftest run-time-choice-and-for-worlds
  ;; every occurrence of a superposed variable chooses on its own
  (is (close? {2 0.25 3 0.5 4 0.25} (worlds '(let [x (amb 1 2)] (+ x x))) 0.02))
  (is (close? {true 0.5 false 0.5} (worlds '(let [x (amb 1 2)] (= x x))) 0.02))
  ;; for-worlds shares one world per binding: call-time choice
  (is (close? {2 0.5 4 0.5} (worlds '(for-worlds [x (amb 1 2)] (+ x x))) 1e-6))
  (is (close? {true 1.0} (worlds '(for-worlds [x (amb 1 2)] (= x x))) 1e-6))
  (is (close? {11 0.25 12 0.25 21 0.25 22 0.25}
              (worlds '(for-worlds [x (amb 1 2) y (amb 10 20)] (+ x y))) 1e-6)))

(deftest superposed-if
  (is (close? {:then 0.5 :else 0.5} (worlds '(if (amb true false) :then :else)) 0.02))
  (is (close? {:then 0.75 :else 0.25}
              (worlds '(if (superpose {true 3 nil 1}) :then :else)) 0.02))
  (testing "the split: each branch sees only its own worlds of the test variable"
    (is (close? {:zero 0.5 0 0.5}
                (worlds '(let [n (amb 0 1)] (if (zero? n) :zero (dec n)))) 0.02)))
  (testing "recursion whose depth differs between worlds terminates"
    (w/run '(defn dbl [n] (if (zero? n) 0 (+ 2 (dbl (dec n))))))
    (is (close? {4 0.5 6 0.3 10 0.2} (worlds '(dbl (superpose {2 5 3 3 5 2}))) 0.02))
    (w/run '(defn len [xs] (if (empty? xs) 0 (inc (len (rest xs))))))
    (is (close? {2 0.5 3 0.3 5 0.2}
                (worlds '(len (superpose {(quote (a b)) 5 (quote (a b c)) 3
                                          (quote (a b c d e)) 2})))
                0.03)))
  (testing "cond, and, or"
    (is (close? {:z 0.5 :nz 0.5} (worlds '(let [n (amb 0 1)] (cond (zero? n) :z :else :nz))) 0.02))
    (is (close? {false 0.5 :y 0.5} (worlds '(and (amb true false) :y)) 0.02))
    (is (close? {:y 0.5 true 0.5} (worlds '(or (amb true false) :y)) 0.02))))

(deftest api
  (is (close? {:a 0.75 :b 0.25} (worlds '(superpose {:a 3 :b 1})) 1e-6))
  (is (= {:a 750 :b 250} (w/run '(worlds (superpose {:a 3 :b 1})))))
  (is (= 250 (w/run '(weight (superpose {:a 3 :b 1}) :b))))
  (is (= 0 (w/run '(weight (superpose {:a 3 :b 1}) :c))))
  (is (= :a (w/run '(collapse (superpose {:a 3 :b 1})))))
  (testing "partial collapse sharpens (and so distorts) the weights"
    (let [p (worlds '(collapse (superpose {:a 3 :b 1}) 4))]
      (is (> (get p :a) 0.75))))
  (testing "sample draws in proportion to weight"
    (let [xs (repeatedly 200 #(w/run '(sample (superpose {:a 3 :b 1}))))]
      (is (<= 120 (count (filter #{:a} xs)) 180))))
  (testing "assert is post-selection"
    (is (close? {2 0.5 3 0.5} (worlds '(assert (fn [x] (not (= x 1))) (amb 1 2 3))) 1e-6))
    (is (thrown? Exception (w/run '(assert (fn [x] false) (amb 1 2))))))
  (is (close? {:a 0.5 :b 0.5} (worlds '(support (superpose {:a 3 :b 1}))) 1e-6))
  (is (close? {:c 1.0} (worlds '(without (amb :a :b :c) (amb :a :b))) 1e-6))
  (is (close? {11 0.5 9 0.5} (worlds '((amb inc dec) 10)) 1e-6))
  (is (close? {'(2 3) 0.5 '(11 21) 0.5}
              (worlds '(for-worlds [xs (amb [1 2] [10 20])] (map inc xs))) 1e-6)))

(deftest worlds-throws-beyond-capacity
  (testing "integers: the residue encoding's world capacity"
    (is (thrown-with-msg? Exception #"capacity"
                          (w/run '(worlds (amb 1 2 3 4 5 6 7 8 9 10 11 12))))))
  (testing "64 superposed lists at D = 2048"
    (let [lists (map (fn [i] (list 'quote (list (symbol (str "s" i)) i))) (range 64))]
      (is (thrown-with-msg? Exception #"capacity"
                            (w/run-worlds (cons 'amb lists)))))))

(deftest readout-is-load-independent
  ;; W0 acceptance: at N = 16k rows in M, worlds still read back
  (w/init!)
  (h/add-random! (vsc/item-memory) 12000 1)
  (is (< 12000 (h/mem-size (vsc/item-memory))))
  (is (close? {'a 0.5 'c 0.5} (worlds '(first (amb (quote (a b)) (quote (c d))))) 0.02))
  (w/init!))
