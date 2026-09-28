(ns vsc.experiments.tasks
  "Task generators for the experiment specs: each takes the run's params and
  returns a task map (see vsc.experiments)."
  (:require
   [vsc.core :as vsc]
   [vsc.hdc :as h]))

;; ---------------------------------------------------------------------------
;; E1: primitive cleanup, straight on the substrate

(defonce ^:private primed (atom nil))

(defn- primed-memory
  "A codebook holding `load` random unitary atoms, cached across the runs of
  one worker that share [dim load seed] (only the last one is kept)."
  [dim load seed]
  (let [k [dim load seed]]
    (if (= k (:key @primed))
      @primed
      (do (reset! primed nil)
          (System/gc)
          (let [s (h/space dim seed)
                mem (h/memory s :codebook {:capacity load})]
            (h/add-random! mem load 1)
            (reset! primed {:key k :space s :mem mem}))))))

(defn primitive-probe
  "Probe a codebook of `load` atoms with one of them under the substrate's
  probe noise; success = nearest is that atom. `trials` probes per run."
  [{:keys [dim load seed probe-noise trials] :or {trials 50}}]
  {:trials trials
   :native
   (fn []
     (let [{s :space mem :mem} (primed-memory dim load seed)
           rnd (java.util.Random. (+ (* 1000003 seed) (long (* 1000 probe-noise))))]
       (h/configure! s {:probe-noise probe-noise})
       (h/reset-instruments! s)
       (h/instrument! s {:count-ops? true :log-margins? true})
       (let [hits (count (filter true? (repeatedly trials
                                                   #(let [i (.nextInt rnd load)]
                                                      (= i (first (h/nearest mem (h/mem-get mem i))))))))]
         {:successes hits :trials trials :m-size load
          :counts (h/op-counts s) :margins (h/margin-stats s)})))})

;; ---------------------------------------------------------------------------
;; E2, E4: structure round trips through the interpreter

(defn- syms [n] (mapv #(symbol (str "a" %)) (range n)))
(defn- kws [prefix n] (mapv #(keyword (str prefix %)) (range n)))

(def ^:private programs
  '[(defn mul [a b] (if (zero? b) 0 (+ a (mul a (dec b)))))
    (defn fact [n] (if (zero? n) 1 (mul n (fact (dec n)))))])

(defn structure-value
  "The host value a :struct/:n task round-trips."
  [kind n]
  (case kind
    :list (apply list (syms n))
    :vec (syms n)
    :nested (vec (for [i (range n)] {:id i :tag (keyword (str "t" i))}))
    :map (zipmap (kws "k" n) (kws "v" n))
    :intmap (zipmap (kws "k" n) (range n))
    :set (set (kws "k" n))))

(defn structure
  "Store a structure noise-free, then read it back and decode it under the
  knobs. :struct is :list :vec :nested :map :intmap :set, or :arith
  ((+ 17 25)) and :program ((fact 4) on a hand-rolled multiply)."
  [{:keys [struct n]}]
  (case struct
    :arith {:form '(+ 17 25) :expect 42}
    :program {:setup programs :form '(fact 4) :expect 24}
    (let [v (structure-value struct n)]
      {:setup [(list 'def 'x (list 'quote v))] :form 'x :expect v})))

;; ---------------------------------------------------------------------------
;; E3: capacity of maps and sets, enforcement off

(defn- shuffle-by
  "A deterministic shuffle of xs (seeded by n)."
  [n xs]
  (let [l (java.util.ArrayList. ^java.util.Collection xs)]
    (java.util.Collections/shuffle l (java.util.Random. (long n)))
    (vec l)))

(defn capacity
  "A map (or set) of n entries, stored as a quoted literal (no re-peeling at
  def time). :probe :get asks (get m k) for present keys; :absent asks
  (contains? m z) for keys that are not there. successes = correct answers,
  over min(n, trials) present or `trials` absent keys."
  [{:keys [struct n probe trials] :or {trials 24}}]
  (let [v (structure-value struct n)
        present (take (min n trials) (shuffle-by n (kws "k" n)))
        absent (kws "z" trials)
        ask (fn [form] (vsc/run form))]
    {:setup [(list 'def 'm (list 'quote v))]
     :trials (if (= probe :get) (count present) (count absent))
     :run (fn []
            (let [safe (fn [f] (try (boolean (f)) (catch Exception _ false)))
                  ok (case probe
                       :get (mapv (fn [k]
                                    (safe #(if (= struct :set)
                                             (true? (ask (list 'contains? 'm k)))
                                             (= (get v k) (ask (list 'get 'm k))))))
                                  present)
                       :absent (mapv (fn [z] (safe #(false? (ask (list 'contains? 'm z))))) absent))]
              {:successes (count (filter true? ok)) :trials (count ok)}))}))

;; ---------------------------------------------------------------------------
;; E5: interference over a run

(def ^:private countdown-def
  '(defn cd [n acc] (if (zero? n) acc (cd (dec n) (cons n acc)))))

(defn countdown
  "(cd n ()) builds (1 … n) by recursion: every step stores environment and
  list cells, so M grows with n."
  [{:keys [n]}]
  {:setup [countdown-def] :form (list 'cd n ()) :expect (apply list (range 1 (inc n)))})

(defn vocabulary
  "A fixed program, (cd 10 ()), run on a memory preloaded with `load` extra
  random atoms: the load, not the program, fills M."
  [{:keys [load]}]
  {:setup [countdown-def]
   :prepare #(h/add-random! (vsc/item-memory) (long load) 1)
   :form '(cd 10 ()) :expect (apply list (range 1 11))})

;; ---------------------------------------------------------------------------
;; E6: cost

(defn micro
  "Seconds per substrate op on a memory of `load` atoms, measured in Python."
  [{:keys [dim load memory seed] :or {memory :codebook}}]
  {:native
   (fn []
     (let [s (h/space dim seed)
           mem (h/memory s memory {:capacity (max load 16)})]
       (h/add-random! mem load 1)
       (let [b (h/bench s mem (if (> (* dim load) 20000000) 10 40))]
         {:successes 1 :trials 1 :m-size load
          :metrics (into {} (for [[k v] b] [k (* 1e6 v)]))})))})

(defn program-cost
  "(fact 4) on a hand-rolled multiply, on a memory preloaded with `load`
  random atoms."
  [{:keys [load]}]
  {:setup programs
   :prepare #(h/add-random! (vsc/item-memory) (long load) 1)
   :form '(fact 4) :expect 24})
