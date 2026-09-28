(ns vsc.core
  "A vector-symbolic Clojure.

  After Tomkins-Flanagan & Kelly (2024), \"Hey Pentti, We Did It!: A Fully
  Vector-Symbolic Lisp\", but for a Clojure dialect: lists, vectors, maps,
  sets, keywords, integers, closures and the environment are all single HRR
  vectors. The evaluator below only inspects expressions through VSA
  operations (bind, unbind, bundle, similarity, cleanup). The host reader turns
  text into vectors and the printer turns vectors back into text; nothing in
  between is host data.

  Encoding (L, R, K unitary role vectors, M the cleanup memory):

    cons(a, b)   = p  with  M: p ↦ ν(L⊗a + R⊗b)   ;; paper eq. 2
    first(c)     = M(L⊘M(c))                       ;; paper eq. 3
    rest(c)      = M(R⊘M(c))                       ;; paper eq. 4
    [a b]        = same, labelled as a vector cell
    {k v, ...}   = p  with  M: p ↦ ν(ν(Σ k⊗v) + ν(K⊗Σ k) + C⊗n)
    #{x, ...}    = p  with  M: p ↦ ν(ν(Σ x) + C⊗n)
    (fn ...)     = p  with  M: p ↦ ν(L⊗form + R⊗env)   ;; a closure
    n            = B^n, B unitary; + is ⊗, - is ⊘, zero? is sim(x, B^0);
                   B has one frequency band per prime modulus (residue
                   number system), so integers clean up by readout, not lookup
    ()  []  {}  #{}  = dedicated atoms

  Every value is one near-orthogonal unit vector. Symbols, keywords and
  strings are random unitary atoms. A composite is a random pointer p that M
  associates with its trace (heteroassociative cleanup, as in the authors'
  own implementation: plain superposed tuples are similar to each other when
  they share parts, and cleanup then confuses them). Equal traces are
  hash-consed onto one pointer, so structural equality is still vector
  similarity. M labels every key with its kind, so the kind of any value is
  whatever M recognises it as."
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
   [vsc.hdc :as h]))

;; ---------------------------------------------------------------------------
;; the machine

(def ^:const default-dim 2048)

;; similarity thresholds
(def ^:const theta-eq 0.995)     ;; two vectors are the same value
(def ^:const theta-atom 0.5)     ;; a probe is recognised by a memory
(def ^:const theta-def 0.4)      ;; a global binding matches a symbol

;; memory labels = kinds
(def ^:private label->kind
  {1 :sym 2 :kw 3 :str 4 :num 5 :nil 6 :true 7 :false 8 :prim
   10 :list 11 :vec 12 :map 13 :set 14 :fn 15 :box})
(def ^:private kind->label (zipmap (vals label->kind) (keys label->kind)))

(def ^:private coll-kinds [:list :vec :map :set])

(def special-forms '[quote if do def defn fn let cond and or])

(defonce ^:private machine (atom nil))

(defn- m [k] (get @machine k))
(defn- sp [] (m :space))

(defn- intern-atom!
  "The unique vector for host value `x` of `kind`, creating it on first use."
  ([kind x] (intern-atom! kind x nil))
  ([kind x v]
   (let [k [kind x]]
     (or (get @(m :atoms) k)
         (let [v (or v (h/unitary (sp)))
               i (h/mem-add! (m :M) v (kind->label kind))
               v (h/mem-get (m :M) i)]
           (swap! (m :atoms) assoc k v)
           (swap! (m :lexicon) assoc i k)
           v)))))

;; ---------------------------------------------------------------------------
;; primitive VSA vocabulary

(defn- bind [a b] (h/bind (sp) a b))
(defn- unbind [a c] (h/unbind (sp) a c))
(defn- nu [v] (h/normalize (sp) v))
(defn- bundle [& vs] (apply h/bundle (sp) vs))
(defn- sim ^double [a b] (h/sim (sp) a b))
(defn- role [r] (get (m :roles) r))
(defn- tag [k] (get (m :tags) k))

(defn cleanup
  "M(p): the stored trace nearest to `p`; integers are cleaned up by readout."
  [p]
  (h/cleanup (m :C) p))

(defn- pointer
  "M ↞ (p ↦ trace) for a fresh pointer p, or the pointer of an equal trace."
  [kind trace]
  (h/mem-get (m :M) (h/intern! (m :M) (h/unitary (sp)) trace (kind->label kind))))

(defn- deref-ptr [p] (h/deref-ptr (m :M) p))

(defn eq? [a b] (> (sim a b) theta-eq))

(defn kind
  "[kind empty?] of vector `v`: what M or the number readout recognises it as."
  [v]
  (let [[label i s] (h/recognize (m :C) v)
        k (label->kind label)]
    (if (> s theta-atom)
      [k (and (not= k :num) (contains? (m :empty-idx) i))]
      [:unknown false])))

(defn- kind-of [v] (first (kind v)))

(defn- lexicon-entry
  "Host value of an atom, by cleaning it up and reading the lexicon."
  [v]
  (let [[label i] (h/recognize (m :C) v)]
    (if (= :num (label->kind label))
      [:num i]
      (get @(m :lexicon) i))))

;; ---------------------------------------------------------------------------
;; data structures

(defn- cell [kind a b]
  (pointer kind (nu (bundle (bind (role :L) a) (bind (role :R) b)))))

(defn car [c] (h/part (m :C) (role :L) c))
(defn cdr [c] (h/part (m :C) (role :R) c))

(defn- empty-coll? [v] (second (kind v)))

(defn- nil-vec [] (m :nil))
(defn- vnil? [v] (> (sim v (nil-vec)) theta-atom))
(defn- truthy? [v] (not (or (vnil? v) (> (sim v (m :false)) theta-atom))))
(defn- bool [x] (if x (m :true) (m :false)))

(defn- mk-seq [tag-kind xs]
  (reduce (fn [acc x] (cell tag-kind x acc)) (tag tag-kind) (reverse xs)))

(defn mk-list [xs] (mk-seq :list xs))
(defn mk-vec [xs] (mk-seq :vec xs))

(defn- cells
  "Host seq of the elements of a list/vector cell chain."
  [c]
  (lazy-seq
   (when-not (or (vnil? c) (empty-coll? c))
     (cons (car c) (cells (cdr c))))))

(defn- distinct-by-eq [xs]
  (reduce (fn [acc x] (if (some #(eq? x %) acc) acc (conj acc x))) [] xs))

(declare num-vec)

;; Superpositions need near-orthogonal constituents. Integers are not (in the
;; residue encoding, n and n+p agree on p's band), so an integer is boxed
;; behind a hash-consed pointer wherever it enters a map or set.

(defn- slot [x] (if (= :num (kind-of x)) (pointer :box x) x))
(defn- unslot [y] (if (= :box (kind-of y)) (cleanup (deref-ptr y)) y))

(defn- recall-slot
  "The map/set constituent nearest to `p` (never a bare integer)."
  [p]
  (h/clean (m :M) p))

(defn- noise-floor
  "Similarity that chance alone reaches only at ~4.5σ, σ = 1/√D."
  []
  (/ 4.5 (Math/sqrt (m :dim))))

(defn- peel-n
  "The n constituents of superposition `v`, by explaining away."
  [v n]
  (mapv (fn [[_ i]] (h/mem-get (m :M) i)) (h/peel [(m :M)] v -1.0 n)))

(defn- member?
  "Is slot `x` one of the n constituents superposed in `probe`? A clear hit
  or miss is decided by one similarity, O(1); the band in between, where
  noise could go either way, by explaining away all n."
  [probe x parts n]
  (let [s (sim probe x)
        share (/ 1.0 (Math/sqrt (* parts n)))
        sigma (/ 1.0 (Math/sqrt (m :dim)))]
    (cond
      (> s (max (* 0.5 share) (noise-floor))) true
      (< s (* 3 sigma)) false
      :else (boolean (some #(eq? x %) (peel-n probe n))))))

(defn capacity
  "Most constituents a map (parts 3) or set (parts 2) holds reliably: each
  one's share 1/√(parts·n) of the trace must stay 1.25× above the noise floor."
  [parts]
  (long (/ 1.0 (* parts (Math/pow (* 1.25 (noise-floor)) 2)))))

(defn- check-capacity [what parts n]
  (when (and (m :enforce-capacity?) (> n (capacity parts)))
    (throw (ex-info (str what " of " n " exceeds the capacity of a D=" (m :dim)
                         " space (" (capacity parts) "); init! with a larger :dim")
                    {:n n :capacity (capacity parts)}))))

(defn- coll-size
  "Host count of a map/set, read from its C⊗n field."
  ^long [cv]
  (if (empty-coll? cv)
    0
    (let [[_ n] (h/recognize (m :C) (unbind (role :C) (deref-ptr cv)))]
      n)))

(defn mk-map
  "Map from a seq of [k v] vector pairs; later keys win.
  Trace: ν(ν(Σ k⊗v) + ν(K⊗Σ k) + C⊗n)."
  [entries]
  (let [entries (reduce (fn [acc [k v]] (conj (filterv #(not (eq? k (first %))) acc) [k v]))
                        [] entries)
        _ (check-capacity "map" 3 (count entries))
        slots (mapv (fn [[k v]] [(slot k) (slot v)]) entries)]
    (if (empty? entries)
      (tag :map)
      (pointer :map
               (nu (bundle (nu (apply bundle (map (fn [[k v]] (bind k v)) slots)))
                           (nu (bind (role :K) (apply bundle (map first slots))))
                           (bind (role :C) (num-vec (count slots)))))))))

(defn- map-keys [mv]
  (let [n (coll-size mv)]
    (if (zero? n) [] (mapv unslot (peel-n (unbind (role :K) (deref-ptr mv)) n)))))

(defn- map-contains? [mv k]
  (let [n (coll-size mv)]
    (and (pos? n) (member? (unbind (role :K) (deref-ptr mv)) (slot k) 3 n))))

(defn- map-get [mv k]
  (when (map-contains? mv k)
    (unslot (recall-slot (unbind (slot k) (deref-ptr mv))))))

(defn- map-entries [mv]
  (let [trace (deref-ptr mv)]
    (mapv (fn [k] [k (unslot (recall-slot (unbind (slot k) trace)))]) (map-keys mv))))

(defn mk-set
  "Trace: ν(ν(Σ x) + C⊗n)."
  [xs]
  (let [xs (distinct-by-eq xs)]
    (check-capacity "set" 2 (count xs))
    (if (empty? xs)
      (tag :set)
      (pointer :set (nu (bundle (nu (apply bundle (map slot xs)))
                                (bind (role :C) (num-vec (count xs)))))))))

(defn- set-elems [sv]
  (let [n (coll-size sv)]
    (if (zero? n) [] (mapv unslot (peel-n (deref-ptr sv) n)))))

(defn- set-contains? [sv x]
  (let [n (coll-size sv)]
    (and (pos? n) (member? (deref-ptr sv) (slot x) 2 n))))

(defn- closure [form env] (cell :fn form env))

;; integers: n = B^n in a residue number system (see hdc.py, Numbers)

(defn- num-vec [n]
  (let [half (h/num-half-range (m :nums))]
    (if (<= (abs n) half)
      (h/num-vec (m :nums) n)
      (throw (ex-info (str "integer out of range ±" half ": " n) {:n n})))))

(defn- zero-vec [] (num-vec 0))
(defn- v-inc [x] (bind (m :base) x))
(defn- v-dec [x] (unbind (m :base) x))
;; not theta-atom: n shares a band with 0 for every modulus dividing it
(defn- v-zero? [x] (eq? x (zero-vec)))

;; ---------------------------------------------------------------------------
;; host <-> vector

(defn encode
  "Host Clojure data → vector (the reader side)."
  [x]
  (cond
    (nil? x) (nil-vec)
    (true? x) (m :true)
    (false? x) (m :false)
    (int? x) (num-vec x)
    (string? x) (intern-atom! :str x)
    (keyword? x) (intern-atom! :kw x)
    (symbol? x) (intern-atom! :sym x)
    (seq? x) (mk-list (map encode x))
    (vector? x) (mk-vec (map encode x))
    (map? x) (mk-map (map (fn [[k v]] [(encode k) (encode v)]) x))
    (set? x) (mk-set (map encode x))
    :else (throw (ex-info (str "cannot encode " (pr-str x)) {:x x}))))

(defn decode
  "Vector → host Clojure data (the printer side)."
  [v]
  (let [[k e?] (kind v)
        atom-value #(second (lexicon-entry v))]
    (case k
      :nil nil
      :true true
      :false false
      (:num :sym :kw :str) (atom-value)
      :prim (tagged-literal 'prim (atom-value))
      :list (if e? () (apply list (map decode (cells v))))
      :vec (if e? [] (mapv decode (cells v)))
      :map (into {} (map (fn [[k x]] [(decode k) (decode x)])) (map-entries v))
      :set (into #{} (map decode) (set-elems v))
      :fn (tagged-literal 'fn (decode (car v)))
      (tagged-literal 'hdv '?))))

(defn- show [v] (pr-str (decode v)))

(defn- fail [msg & vs]
  (throw (ex-info (apply str msg (map show vs)) {})))

;; ---------------------------------------------------------------------------
;; environments: a Lisp 1.5 association list of cons cells, plus the global
;; function memory F probed with L⊗sym (paper eq. 11)

(defn- env-extend [env sym val] (cell :list (cell :list sym val) env))

(defn- global-slot [sym]
  (when (pos? (h/mem-size (m :F)))
    (let [[i _] (h/nearest (m :F) (bind (role :L) sym))
          slot (h/mem-get (m :F) i)]
      (when (> (sim (unbind (role :L) slot) sym) theta-def)
        [i slot]))))

(defn- global-value
  "F(L⊗sym) read out by the memory backend, then its R field cleaned up."
  [sym]
  (when (pos? (h/mem-size (m :F)))
    (let [slot (h/clean (m :F) (bind (role :L) sym))]
      (when (> (sim (unbind (role :L) slot) sym) theta-def)
        (cleanup (unbind (role :R) slot))))))

(defn define!
  "F ↞ cons(sym, val), replacing an earlier definition of `sym`."
  [sym val]
  (let [slot (nu (bundle (bind (role :L) sym) (bind (role :R) val)))]
    (if-let [[i _] (global-slot sym)]
      (h/mem-put! (m :F) i slot)
      (h/mem-add! (m :F) slot 0))
    sym))

(defn- lookup [sym env]
  (loop [env env]
    (if (empty-coll? env)
      (or (global-value sym)
          (fail "unable to resolve symbol: " sym))
      (let [binding (car env)]
        (if (eq? (car binding) sym)
          (cdr binding)
          (recur (cdr env)))))))

;; ---------------------------------------------------------------------------
;; eval

(declare veval apply-fn)

(defn- special-form [head]
  (when (= :sym (kind-of head))
    (let [[_ label s] (h/recall (m :SF) head)]
      (when (> s theta-atom) (nth special-forms label)))))

(defn- eval-body [body env]
  (reduce (fn [_ form] (veval form env)) (nil-vec) (cells body)))

(defn- bind-params [env params args]
  (loop [env env ps (cells params) args args]
    (cond
      (empty? ps) (if (seq args) (fail "too many arguments for params " params) env)
      (eq? (first ps) (m :amp)) (env-extend env (second ps) (mk-list args))
      (empty? args) (fail "too few arguments for params " params)
      :else (recur (env-extend env (first ps) (first args)) (rest ps) (rest args)))))

(defn- eval-special [sf args env]
  (let [[a b c] (cells args)]
    (case sf
      quote a
      if (if (truthy? (veval a env)) (veval b env) (if c (veval c env) (nil-vec)))
      do (eval-body args env)
      def (define! a (veval b env))
      defn (define! a (closure (cell :list (m :fn-sym) args) (tag :list)))
      fn (closure (cell :list (m :fn-sym) args) env)
      let (let [env (reduce (fn [env [sym x]] (env-extend env sym (veval x env)))
                            env (partition 2 (cells a)))]
            (eval-body (cdr args) env))
      cond (or (some (fn [[t x]] (when (truthy? (veval t env)) (veval x env)))
                     (partition 2 (cells args)))
               (nil-vec))
      and (reduce (fn [_ x] (let [v (veval x env)] (if (truthy? v) v (reduced v))))
                  (m :true) (cells args))
      or (reduce (fn [_ x] (let [v (veval x env)] (if (truthy? v) (reduced v) v)))
                 (nil-vec) (cells args)))))

(defn veval
  "Evaluate expression vector `e` in environment vector `env`."
  [e env]
  (let [[k e?] (kind e)]
    (case k
      :sym (lookup e env)
      :list (if e?
              e
              (let [head (car e)
                    args (cdr e)]
                (if-let [sf (special-form head)]
                  (eval-special sf args env)
                  (apply-fn (veval head env) (mapv #(veval % env) (cells args))))))
      :vec (if e? e (mk-vec (mapv #(veval % env) (cells e))))
      :map (mk-map (mapv (fn [[k x]] [(veval k env) (veval x env)]) (map-entries e)))
      :set (mk-set (mapv #(veval % env) (set-elems e)))
      :unknown (throw (ex-info "cannot evaluate an unrecognised vector" {}))
      e)))

(defn- apply-closure [f args]
  (let [form (car f)
        env (cdr f)
        [x & more] (cells (cdr form))
        named? (= :sym (kind-of x))
        params (if named? (first more) x)
        body (cdr (if named? (cdr (cdr form)) (cdr form)))
        ;; (fn name [..] ..) can call itself through its local name
        env (bind-params (if named? (env-extend env x f) env) params args)]
    (eval-body body env)))

(defn apply-fn [f args]
  (case (kind-of f)
    :prim (let [[i _ s] (h/recall (m :P) f)]
            (when (< s theta-atom) (fail "unknown primitive " f))
            ((:f (nth (m :prims) i)) args))
    :fn (apply-closure f args)
    :kw (let [[coll default] args]
          (or (when (= :map (kind-of coll)) (map-get coll f)) default (nil-vec)))
    :map (or (map-get f (first args)) (second args) (nil-vec))
    :set (if (set-contains? f (first args)) (cleanup (first args)) (nil-vec))
    (fail "not a function: " f)))

;; ---------------------------------------------------------------------------
;; primitives: every one takes and returns vectors

(defn- seq-kind? [v] (#{:list :vec} (kind-of v)))

(defn- as-cells
  "Elements of any collection as a host seq of vectors."
  [coll]
  (case (kind-of coll)
    (:list :vec) (cells coll)
    :map (map (fn [[k v]] (mk-vec [k v])) (map-entries coll))
    :set (set-elems coll)
    :nil ()
    (fail "not a collection: " coll)))

(defn- v-count [coll]
  (if (and (#{:map :set} (kind-of coll)) (not (empty-coll? coll)))
    (cleanup (unbind (role :C) (deref-ptr coll)))
    (reduce (fn [n _] (v-inc n)) (zero-vec) (as-cells coll))))

(defn- v-nth [coll i]
  (loop [c coll i i]
    (cond
      (or (vnil? c) (empty-coll? c)) (fail "index out of bounds")
      (v-zero? i) (car c)
      :else (recur (cdr c) (v-dec i)))))

(defn- v= [a b]
  (if (and (seq-kind? a) (seq-kind? b))
    (let [xs (as-cells a) ys (as-cells b)]
      (and (= (count xs) (count ys)) (every? true? (map v= xs ys))))
    (eq? a b)))

(defn- int-of [v]
  (let [[kind n] (lexicon-entry v)]
    (if (= kind :num) n (fail "not an integer: " v))))

(def ^:private primitives
  [['cons (fn [[x coll]]
            (cell :list x (if (or (vnil? coll) (= :list (kind-of coll)))
                            (if (vnil? coll) (tag :list) coll)
                            (mk-list (as-cells coll)))))]
   ['first (fn [[c]] (or (first (as-cells c)) (nil-vec)))]
   ['rest (fn [[c]] (if (and (seq-kind? c) (not (empty-coll? c)))
                      (cdr c)
                      (mk-list (rest (as-cells c)))))]
   ['next (fn [[c]] (let [r (rest (as-cells c))] (if (seq r) (mk-list r) (nil-vec))))]
   ['seq (fn [[c]] (let [xs (as-cells c)]
                     (cond (empty? xs) (nil-vec)
                           (= :list (kind-of c)) c
                           :else (mk-list xs))))]
   ['empty? (fn [[c]] (bool (or (vnil? c) (empty? (as-cells c)))))]
   ['list (fn [xs] (mk-list xs))]
   ['vector (fn [xs] (mk-vec xs))]
   ['vec (fn [[c]] (mk-vec (as-cells c)))]
   ['hash-map (fn [xs] (mk-map (map vec (partition 2 xs))))]
   ['hash-set (fn [xs] (mk-set xs))]
   ['set (fn [[c]] (mk-set (as-cells c)))]
   ['get (fn [[coll k default]]
           (or (case (kind-of coll)
                 :map (map-get coll k)
                 :set (when (set-contains? coll k) (cleanup k))
                 :vec (v-nth coll k)
                 nil)
               default (nil-vec)))]
   ['contains? (fn [[coll k]]
                 (bool (case (kind-of coll)
                         :map (map-contains? coll k)
                         :set (set-contains? coll k)
                         false)))]
   ['assoc (fn [[mv & kvs]]
             (mk-map (concat (when-not (vnil? mv) (map-entries mv)) (map vec (partition 2 kvs)))))]
   ['dissoc (fn [[mv & ks]]
              (mk-map (remove (fn [[k _]] (some #(eq? k %) ks)) (map-entries mv))))]
   ['keys (fn [[mv]] (let [ks (map-keys mv)] (if (seq ks) (mk-list ks) (nil-vec))))]
   ['vals (fn [[mv]] (let [es (map-entries mv)] (if (seq es) (mk-list (map second es)) (nil-vec))))]
   ['conj (fn [[coll & xs]]
            (case (kind-of coll)
              (:list :nil) (reduce #(cell :list %2 %1) (if (vnil? coll) (tag :list) coll) xs)
              :vec (mk-vec (concat (cells coll) xs))
              :set (mk-set (concat (set-elems coll) xs))
              :map (mk-map (concat (map-entries coll)
                                   (map (fn [e] (vec (cells e))) xs)))
              (fail "cannot conj onto " coll)))]
   ['count (fn [[c]] (v-count c))]
   ['nth (fn [[c i]] (v-nth c i))]
   ['= (fn [[a & bs]] (bool (every? #(v= a %) bs)))]
   ['not (fn [[x]] (bool (not (truthy? x))))]
   ['inc (fn [[x]] (v-inc x))]
   ['dec (fn [[x]] (v-dec x))]
   ['+ (fn [xs] (reduce bind (zero-vec) xs))]
   ['- (fn [[x & ys]] (if ys (reduce #(unbind %2 %1) x ys) (h/inverse (sp) x)))]
   ['zero? (fn [[x]] (bool (v-zero? x)))]
   ['nil? (fn [[x]] (bool (vnil? x)))]
   ['some? (fn [[x]] (bool (not (vnil? x))))]
   ['list? (fn [[x]] (bool (= :list (kind-of x))))]
   ['vector? (fn [[x]] (bool (= :vec (kind-of x))))]
   ['map? (fn [[x]] (bool (= :map (kind-of x))))]
   ['set? (fn [[x]] (bool (= :set (kind-of x))))]
   ['coll? (fn [[x]] (bool (#{:list :vec :map :set} (kind-of x))))]
   ['fn? (fn [[x]] (bool (#{:fn :prim} (kind-of x))))]
   ['keyword? (fn [[x]] (bool (= :kw (kind-of x))))]
   ['symbol? (fn [[x]] (bool (= :sym (kind-of x))))]
   ['string? (fn [[x]] (bool (= :str (kind-of x))))]
   ['number? (fn [[x]] (bool (= :num (kind-of x))))]
   ['eval (fn [[x]] (veval x (tag :list)))]
   ['apply (fn [[f & args]] (apply-fn f (concat (butlast args) (as-cells (last args)))))]
   ['println (fn [xs] (println (str/join " " (map #(let [d (decode %)] (if (string? d) d (pr-str d))) xs))) (nil-vec))]
   ;; the substrate, exposed
   ['bind (fn [[a b]] (bind a b))]
   ['unbind (fn [[a c]] (unbind a c))]
   ['bundle (fn [xs] (nu (apply bundle xs)))]
   ['cleanup (fn [[x]] (cleanup x))]
   ['degrade (fn [[x pct]] (h/degrade (sp) x (/ (int-of pct) 100.0)))]
   ['similarity (fn [[a b]] (num-vec (Math/round (* 100 (sim a b)))))]
   ['memory-size (fn [_] (num-vec (h/mem-size (m :M))))]])

;; ---------------------------------------------------------------------------
;; boot

(declare run-string)

(def knob-keys
  "Substrate noise knobs, all off by default (see hdc.py, Space.configure)."
  [:op-noise :probe-noise :lesion :noise-seed])

(defn init!
  "Build a fresh machine: space, cleanup memories, roles, tags, numbers,
  primitives, then load the prelude (written in the vector-symbolic dialect).

  Options:
    :dim :seed :prelude?
    :memory        cleanup backend, :codebook (default), :linear or :mhn
    :memory-opts   options for the backend (e.g. {:beta 16 :mode :snap})
    :op-noise σ    noise on every bind/unbind/bundle output
    :probe-noise σ noise on every cleanup probe
    :lesion f      a fixed fraction f of dimensions is dead everywhere
    :noise-seed    seed of the noise stream (default: derived from :seed)
    :memory-damage σ  remembered for `damage!`, which experiments call once
                   the memory is filled; init! itself does not damage
    :count-ops? :log-margins?  instrumentation (see `instruments`)
    :enforce-capacity?  refuse maps and sets beyond `capacity` (default true)

  The knobs act from the start, so a noisy init! also loads the prelude
  noisily; `set-knobs!` switches them later."
  ([] (init! {}))
  ([{:keys [dim seed prelude? memory memory-opts memory-damage enforce-capacity?]
     :or {dim default-dim seed 42 prelude? true memory :codebook memory-opts {}
          enforce-capacity? true}
     :as opts}]
   (let [s (h/space dim seed)
         _ (h/configure! s opts)
         _ (h/instrument! s opts)
         mem (fn [nm] (h/memory s memory (assoc memory-opts :name nm)))
         M (mem "M")]
     (reset! machine {:space s :M M :dim dim :backend memory
                      :F (mem "F") :SF (mem "SF") :P (mem "P")
                      :knobs (select-keys opts knob-keys)
                      :memory-damage memory-damage
                      :enforce-capacity? enforce-capacity?
                      :roles (zipmap [:L :R :K :C] (repeatedly #(h/unitary s)))
                      :atoms (atom {}) :lexicon (atom {})})
     ;; the empty collections are atoms of their collection's kind
     (let [tags (into {} (for [k coll-kinds] [k (intern-atom! k (symbol (str "empty-" (name k))))]))]
       (swap! machine assoc
              :tags tags
              :empty-idx (set (for [v (vals tags)] (first (h/nearest M v))))))
     (swap! machine assoc
            :nil (intern-atom! :nil nil)
            :true (intern-atom! :true true)
            :false (intern-atom! :false false))
     (let [nums (h/numbers s)]
       (swap! machine assoc
              :nums nums
              :base (h/num-vec nums 1)
              :C (h/cleanup-machine s M nums)))
     (swap! machine assoc :fn-sym (encode 'fn) :amp (encode '&))
     (doseq [[i sf] (map-indexed vector special-forms)]
       (h/mem-add! (m :SF) (encode sf) i))
     (swap! machine assoc :prims
            (vec (for [[i [name f]] (map-indexed vector primitives)]
                   (let [pv (intern-atom! :prim name)]
                     (h/mem-add! (m :P) pv i)
                     (define! (encode name) pv)
                     {:name name :f f}))))
     (when prelude?
       (run-string (slurp (io/resource "vsc/prelude.clj"))))
     :ready)))

(defn read-forms [src]
  (read-string (str "[" src "\n]")))

(defn eval-form
  "Encode host form, evaluate it in vector space, return the result vector."
  [form]
  (veval (encode form) (tag :list)))

(defn run
  "Evaluate a host form and decode the result."
  [form]
  (decode (eval-form form)))

(defn run-string
  "Evaluate every form in `src`, returning the decoded value of the last."
  [src]
  (reduce (fn [_ form] (run form)) nil (read-forms src)))

(defn stats []
  {:dim (m :dim)
   :backend (m :backend)
   :traces (h/mem-size (m :M))
   :globals (h/mem-size (m :F))})

;; ---------------------------------------------------------------------------
;; experiment hooks: knobs and instruments live in the substrate (hdc.py), so
;; no code path of the interpreter can bypass them

(defn space "The machine's HRR space (the Python object)." [] (sp))

(defn set-knobs!
  "Replace the noise knobs (:op-noise :probe-noise :lesion :noise-seed);
  missing ones are off. Setting :lesion also lesions every stored row."
  [knobs]
  (swap! machine assoc :knobs (select-keys knobs knob-keys))
  (h/configure! (sp) knobs))

(defn damage!
  "Synaptic damage: noise of norm σ added once to every stored row of M, F,
  SF and P. Without σ, the :memory-damage given to init!."
  ([] (damage! (or (m :memory-damage) 0.0)))
  ([sigma] (when (pos? sigma) (h/damage! (sp) sigma)) sigma))

(defn instrument!
  "Switch {:count-ops? :log-margins?} on or off; missing keys are unchanged."
  [opts]
  (h/instrument! (sp) opts))

(defn reset-instruments! [] (h/reset-instruments! (sp)))

(defn instruments
  "{:counts {kind n} :margins {:n :min :p05 :median ..}} since the last reset."
  []
  {:counts (h/op-counts (sp)) :margins (h/margin-stats (sp))})

(defn margin-log [] (h/margin-log (sp)))

(defn item-memory "The item memory M (atoms and pointers)." [] (m :M))

(defn digest
  "{memory sha1} over the stored rows of M, F, SF and P: two runs agree
  bit for bit iff their digests do."
  []
  (into {} (for [k [:M :F :SF :P]] [k (h/mem-digest (m k))])))
