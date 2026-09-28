(ns vsc.substrate-test
  "The experiment knobs: off means bit-identical, on means they act."
  (:require
   [clojure.test :refer [deftest is testing]]
   [vsc.core :as vsc]
   [vsc.hdc :as h]))

(def ^:private forms
  '[(defn fact [n] (if (zero? n) 1 (* n (fact (dec n)))))
    (fact 4)
    {:a 1 :b [1 2] :c {:d "e"}}
    (keys {:a 1 :b 2 :c 3})
    #{1 2 3}
    (first (degrade (quote (a b c)) 100))])

(defn- fingerprint
  "Decoded results and memory digests of `forms` on a machine built with `opts`."
  [opts]
  (vsc/init! opts)
  [(mapv vsc/run forms) (vsc/digest)])

(deftest knobs-off-is-bit-identical
  (let [base (fingerprint {})]
    (testing "explicit zero knobs"
      (is (= base (fingerprint {:memory :codebook :op-noise 0 :probe-noise 0.0
                                :lesion 0 :memory-damage 0}))))
    (testing "lesion 0"
      (is (= base (fingerprint {:lesion 0.0}))))
    (testing "instrumentation on"
      (is (= base (fingerprint {:count-ops? true :log-margins? true})))
      (let [{:keys [counts margins]} (vsc/instruments)]
        (is (pos? (:bind counts)))
        (is (pos? (:rows counts)))
        (is (pos? (:n margins)))
        (is (<= 0.0 (:min margins)))
        (vsc/reset-instruments!)
        (is (= {} (:counts (vsc/instruments))))))
    (testing "damage! with the default σ = 0 changes nothing"
      (vsc/init! {})
      (mapv vsc/run forms)
      (let [d (vsc/digest)]
        (vsc/damage!)
        (is (= d (vsc/digest)))))
    (testing "set-knobs! back to zero"
      (vsc/init! {})
      (vsc/set-knobs! {:op-noise 0.5})
      (vsc/set-knobs! {})
      (is (= base [(mapv vsc/run forms) (vsc/digest)])))))

(deftest knobs-act-and-are-seeded
  (let [base (fingerprint {})
        top1s (fn [opts]
                (vsc/init! (assoc opts :log-margins? true :prelude? false))
                (vsc/run '(first (quote (a b c))))
                (mapv #(nth % 2) (vsc/margin-log)))]
    (doseq [opts [{:op-noise 0.05} {:lesion 0.01}]]
      (let [a (fingerprint opts)]
        (testing (pr-str opts)
          (is (not= (second base) (second a)) "the knob changes the stored rows")
          (is (= (first base) (first a)) "small noise is cleaned up")
          (is (= a (fingerprint opts)) "same seed, same noise"))))
    (testing "probe noise leaves M alone and moves the similarities"
      (is (= base (fingerprint {:probe-noise 0.05})))
      (is (not= (top1s {}) (top1s {:probe-noise 0.05})))
      (is (= (top1s {:probe-noise 0.05}) (top1s {:probe-noise 0.05}))))
    (testing "a different noise seed gives different noise"
      (is (not= (second (fingerprint {:op-noise 0.05 :noise-seed 1}))
                (second (fingerprint {:op-noise 0.05 :noise-seed 2})))))
    (testing "memory damage"
      (vsc/init! {:memory-damage 0.5})
      (vsc/run '(def xs (quote (a b c))))
      (let [d (vsc/digest)]
        (vsc/damage!)
        (is (not= (:M d) (:M (vsc/digest))))
        (is (= '(a b c) (vsc/run 'xs)))))
    (testing "large probe noise breaks cleanup"
      (vsc/init! {:prelude? false})
      (vsc/run '(def xs (quote (a b c d e f g h))))
      (vsc/set-knobs! {:probe-noise 20.0})
      (is (not= '(a b c d e f g h)
                (try (vsc/run 'xs) (catch Exception _ ::failed)))))))

(deftest lesion-masks-every-vector
  (vsc/init! {:prelude? false :dim 1024})
  (vsc/set-knobs! {:lesion 0.25})
  (let [s (vsc/space)
        v (h/bind s (h/unitary s) (h/unitary s))
        zeros (count (filter zero? (seq (libpython-clj2.python/->jvm
                                         (libpython-clj2.python/call-attr v "tolist")))))]
    (is (= 256 zeros))))

(defn- nonzeros [v]
  (count (remove zero? (libpython-clj2.python/->jvm (libpython-clj2.python/call-attr v "tolist")))))

(defn- dead-dim-0-seed
  "A noise seed whose lesion mask at fraction f kills dimension 0."
  [f]
  (first (filter (fn [k]
                   (vsc/set-knobs! {:lesion f :noise-seed k})
                   (zero? (libpython-clj2.python/->jvm
                           (libpython-clj2.python/get-item
                            (libpython-clj2.python/get-attr (vsc/space) "mask") 0))))
                 (range 1000))))

(deftest integers-survive-losing-dimension-0
  ;; P1 E4: 0 was B^0, a delta at index 0, so killing that one dimension
  ;; made 0 the zero vector and broke all arithmetic. Now n = B^n ⊗ Z.
  (vsc/init! {:prelude? false :dim 2048})
  (testing "0 is a dense vector"
    (is (< 2000 (nonzeros (vsc/encode 0)))))
  (let [k (dead-dim-0-seed 0.005)]
    (vsc/init! {:prelude? false :dim 2048})
    (vsc/run '(defn mul [a b] (if (zero? b) 0 (+ a (mul a (dec b))))))
    (vsc/set-knobs! {:lesion 0.005 :noise-seed k})
    (is (some? k))
    (is (= 42 (vsc/run '(+ 17 25))))
    (is (= [true false 0 -3] (vsc/run '[(zero? (- 5 5)) (zero? 1) (+ 0 0) (- 3)])))
    (is (= 12 (vsc/run '(mul 3 4))))))

(deftest decisions-scale-with-the-noise-floor
  ;; P1 E2: every structure failed at probe noise √3 (the kind check's
  ;; θ = 0.5) and at op noise 1.46 (the global lookup's θ = 0.4), for every D
  (testing "probe noise twice the old cliff"
    (vsc/init! {:prelude? false :dim 2048})
    (vsc/run '(def x (quote (a b c d e f g h))))
    (vsc/run '(def m {:a :x :b :y}))
    (vsc/set-knobs! {:probe-noise 3.5})
    (is (= '(a b c d e f g h) (vsc/run 'x)))
    (is (= :y (vsc/run '(get m :b)))))
  (testing "op noise past the old cliff"
    (vsc/init! {:prelude? false :dim 2048})
    (vsc/run '(def m {:a 1 :b 2}))
    (vsc/set-knobs! {:op-noise 2.5})
    (is (= {:a 1 :b 2} (vsc/run 'm))))
  (testing "eq? compares identities, so integers sharing residue bands differ"
    (vsc/init! {:prelude? false})
    (is (= [false true false] (vsc/run '[(= 0 5005) (= 5005 (+ 5000 5)) (zero? 5005)])))))

(deftest memory-backends
  (testing "the codebook snaps, the linear memory does not"
    (doseq [[backend snapped?] [[:codebook true] [:linear false]]]
      (vsc/init! {:prelude? false :memory backend})
      (let [s (vsc/space)
            mem (h/memory s backend {})
            rows (vec (repeatedly 50 #(h/unitary s)))
            _ (doseq [r rows] (h/mem-add! mem r 0))
            probe (h/degrade s (rows 7) 0.5)]
        (is (= 7 (first (h/nearest mem probe))) "nearest is the argmax either way")
        (is (= snapped? (> (h/sim s (h/clean mem probe) (rows 7)) 0.999)) (name backend)))))
  (testing "the interpreter runs on the linear backend for small programs"
    (vsc/init! {:prelude? false :memory :linear})
    (is (= 3 (vsc/run '(+ 1 2))))
    ;; before the calibrated eq? (P1 E6) no closure call survived a blend
    (vsc/run '(defn mul [a b] (if (zero? b) 0 (+ a (mul a (dec b))))))
    (is (= 12 (vsc/run '(mul 3 4)))))
  (testing "mhn is a slot for P2"
    (is (thrown? Exception (vsc/init! {:prelude? false :memory :mhn})))))

(deftest op-budget
  (vsc/init! {:prelude? false})
  (h/budget! (vsc/space) {:max-ops 100})
  (is (thrown? Exception (vsc/run '(quote (a b c d e f g h i j k l m n)))))
  (h/budget! (vsc/space) {}))
