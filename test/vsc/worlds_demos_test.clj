(ns vsc.worlds-demos-test
  "W2 demos (examples/worlds/*.clj), each against real Clojure: BFS levels
  computed on the host, and the probabilistic demos against exact
  enumeration (total variation distance)."
  (:require
   [clojure.test :refer [deftest is testing]]
   [vsc.core :as vsc]
   [vsc.worlds :as w]))

(defn- load-demo [file opts]
  (w/init! opts)
  (w/run-string (slurp file)))

(defn- dist
  "Exact distribution {value p} of a seq of [value p] pairs."
  [pairs]
  (reduce (fn [m [v p]] (update m v (fnil + 0.0) p)) {} pairs))

;; ---------------------------------------------------------------------------
;; BFS

(defn- bfs-levels
  "BFS levels from `start` in real Clojure, as a vector of sets."
  [graph start]
  (loop [frontier #{start} seen #{start} levels []]
    (if (empty? frontier)
      levels
      (let [nxt (set (remove seen (mapcat graph frontier)))]
        (recur nxt (into seen nxt) (conj levels frontier))))))

(defn- demo-graph
  "The edge list of examples/worlds/bfs.clj, read by the host reader."
  []
  (let [forms (vsc/read-forms (slurp "examples/worlds/bfs.clj"))
        [_ _ [_ edges]] (first (filter #(and (seq? %) (= 'def (first %)) (= 'graph (second %))) forms))]
    (into {} (map (fn [[k vs]] [k (set vs)])) edges)))

(deftest bfs-one-vector-per-level
  (let [levels (load-demo "examples/worlds/bfs.clj" {})]
    (is (= (bfs-levels (demo-graph) :n0) (mapv set levels)))
    (is (= [#{:n0} #{:n1 :n5} #{:n2 :n6 :n9} #{:n3 :n10}] (mapv set levels)))))

;; ---------------------------------------------------------------------------
;; parallel map

(deftest parallel-map-over-worlds
  (let [paired (load-demo "examples/worlds/parallel-map.clj" {})
        product (dist (for [a [2 11] b [3 21] c [4 31]] [(list a b c) 1/8]))]
    (testing "call-time choice keeps each list whole"
      (is (= 'worlds (:tag paired)))
      (is (< (w/tv-distance {'(2 3 4) 0.5 '(11 21 31) 0.5} (w/run-worlds 'paired)) 0.01)))
    (testing "run-time choice: the product of the per-position distributions"
      (is (< (w/tv-distance product (w/run-worlds 'mapped)) 0.02)))
    (testing "per-position marginals are exact"
      (is (< (w/tv-distance {3 0.5 21 0.5} (w/run-worlds 'second-elements)) 0.02)))
    (testing "worlds leave the recursion at their own depth"
      (is (< (w/tv-distance {2 0.5 4 0.5} (w/run-worlds 'lengths)) 0.02)))))

;; ---------------------------------------------------------------------------
;; the probabilistic reading

(deftest dice-against-exact-enumeration
  (load-demo "examples/worlds/dice.clj" {:dim 4096})
  (let [die (range 1 5)
        two (dist (for [a die b die] [(+ a b) 1/16]))
        high (let [ps (for [a die b die :when (>= (+ a b) 6)] [a 1])
                   z (count ps)]
               (dist (for [[a p] ps] [a (/ p z)])))]
    (is (< (w/tv-distance two (w/run-worlds 'two)) 0.02))
    (is (< (w/tv-distance high (w/run-worlds 'first-given-high)) 0.02))))

(def ^:private sprinkler-cpt
  {:rain 0.2
   :sprinkler {true 0.1 false 0.4}
   :wet {[true true] 0.99 [true false] 0.8 [false true] 0.9 [false false] 0.0}})

(defn- sprinkler-joint []
  (let [{:keys [rain sprinkler wet]} sprinkler-cpt
        p (fn [q x] (if x q (- 1.0 q)))]
    (dist (for [r [true false] s [true false] wt [true false]
                :let [pr (* (p rain r) (p (sprinkler r) s) (p (wet [r s]) wt))]
                :when (pos? pr)]
            [[r s wt] pr]))))

(deftest sprinkler-network-by-post-selection
  (let [rain-given-wet (load-demo "examples/worlds/sprinkler.clj" {})
        joint (sprinkler-joint)
        wet (filter (fn [[[_ _ wt] _]] wt) joint)
        z (reduce + (map second wet))
        exact (dist (for [[[r] p] wet] [r (/ p z)]))]
    (is (< (w/tv-distance joint (w/run-worlds 'joint)) 0.02))
    (is (< (w/tv-distance exact (w/worlds (w/eval-form 'rain-given-wet))) 0.02))
    (is (some? rain-given-wet))
    (testing "without call-time choice the correlation is lost: the prior"
      (let [naive (w/run-worlds 'rain-given-wet-naive)]
        (is (< (w/tv-distance {true 0.2 false 0.8} naive) 0.02))
        (is (> (w/tv-distance exact naive) 0.1))))))
