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
    (is (= 3 (vsc/run '(+ 1 2)))))
  (testing "mhn is a slot for P2"
    (is (thrown? Exception (vsc/init! {:prelude? false :memory :mhn})))))

(deftest op-budget
  (vsc/init! {:prelude? false})
  (h/budget! (vsc/space) {:max-ops 100})
  (is (thrown? Exception (vsc/run '(quote (a b c d e f g h i j k l m n)))))
  (h/budget! (vsc/space) {}))
