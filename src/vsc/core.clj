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
    [x₀ … xₙ₋₁]  = v = ν(Σᵢ Πⁱ(V⊗xᵢ)), stored in M as itself (no pointer),
                   with M: v ↦ N⊗n; nth i = M(V⊘Π⁻ⁱ(v)), Π a random
                   permutation; count = M(N⊘M(v))
    {k v, ...}   = p  with  M: p ↦ ν(ν(Σ k⊗Π(v)) + ν(K⊗Σ k) + C⊗n)
    #{x, ...}    = p  with  M: p ↦ ν(ν(Σ x) + C⊗n)
    (fn ...)     = p  with  M: p ↦ ν(L⊗form + R⊗env)   ;; a closure
    n            = B^n⊗Z, B and Z unitary; inc is B⊗n, + is (m⊗n)⊘Z, zero?
                   is n = Z; B has one frequency band per prime modulus
                   (residue number system), so integers clean up by readout,
                   not lookup
    ()  []  {}  #{}  = dedicated atoms

  Every value is one unit vector. Symbols, keywords and strings are random
  unitary atoms. A list, map, set or closure is a random pointer p that M
  associates with its trace (heteroassociative cleanup, as in the authors'
  own implementation: plain superposed tuples are similar to each other when
  they share parts, and cleanup then confuses them). Equal traces are
  hash-consed onto one pointer, so structural equality is still vector
  similarity. M labels every key with its kind, so the kind of any value is
  whatever M recognises it as.

  Lists and vectors make opposite trade-offs, and the program picks one per
  value (docs/DECISION-sequences.md). A list is a chain of pointers: robust,
  unbounded, opaque (two lists that differ in one element are unrelated
  vectors). A vector is its own content: vectors that share elements at the
  same positions are similar, nth is one readout instead of a walk down the
  chain, and equal vectors are equal without hash-consing; but it holds
  about D/48 elements, it tolerates less noise than a list (each element
  has share 1/√n), and a superposition of vectors keeps only each
  position's marginal."
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
   [vsc.hdc :as h]))

;; ---------------------------------------------------------------------------
;; the machine

(def ^:const default-dim 2048)

;; Decisions. Every decision is either an argmax (a cleanup) or a
;; comparison with the chance floor z/√D, the similarity that unrelated
;; vectors reach only at z standard deviations. Nothing compares a noisy
;; similarity with a fixed number: P1 (E2) found such thresholds failing
;; 7–15× below what the cleanup itself survives, at the same noise for
;; every D. With chance floors, robustness scales with √D like cleanup does.
(def ^:const theta-eq 0.995)     ;; exact copies: the fast path of eq?
(def ^:const theta-atom 0.5)     ;; not used here; only vsc.machine's own datapath
(def ^:const z-recognise 4.5)    ;; a probe is recognised by a memory at all
(def ^:const z-differ 3.0)       ;; below it, eq? says "different" without cleanup

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

;; [W1 hook] superposition hooks, installed by vsc.worlds/init! and reset by
;; every init!. Without them every hooked site below runs its plain code.
(defn- hook [k] (get (m :hooks) k))
(defn set-hooks! "Install {:apply :branch :construct :decode :special} fns." [hs]
  (swap! machine assoc :hooks hs))

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

(defn- chance
  "The chance floor z/√D: unrelated unit vectors have similarity N(0, 1/D)."
  ^double [z]
  (/ (double z) (Math/sqrt (m :dim))))

(defn- identity-of
  "[label index] of what the cleanup recognises `v` as (index is n for an
  integer), or nil if nothing clears the chance floor."
  [v]
  (let [[label i s] (h/recognize (m :C) v)]
    (when (> s (chance z-recognise)) [label i])))

(defn eq?
  "Are `a` and `b` the same value? Exact copies (similarity > θ_eq) and
  unrelated vectors (similarity at chance level) are told apart by one
  similarity. Everything in between, noisy or blended vectors and integers
  that share residue bands (sim(0, 5005) ≈ 2/3), is decided by cleaning up
  both sides and comparing what they are: an argmax on each side, no
  threshold on the similarity itself."
  [a b]
  (let [s (sim a b)]
    (cond
      (> s (or (m :theta-eq) theta-eq)) true
      (< s (chance z-differ)) false
      :else (let [x (identity-of a)] (and (some? x) (= x (identity-of b)))))))

(defn kind
  "[kind empty?] of vector `v`: what M or the number readout recognises it as."
  [v]
  ;; a misread cdr can end a form early, so destructuring its operands
  ;; yields nil; say so here rather than let numpy fail on None
  (when (nil? v)
    (throw (ex-info "missing operand: a noisy read truncated a form" {})))
  (let [[label i s] (h/recognize (m :C) v)
        k (label->kind label)]
    (if (> s (chance z-recognise))
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

;; [W1 hook] the trace goes to M unnormalised: M stores it unit length either
;; way, and a :proj memory records its norm as the exact field gain. Under
;; vsc.worlds each field is unit length too: a superposition Σ wᵢvᵢ (Σ wᵢ = 1)
;; is shorter than one value, and would drown in the other field's crosstalk
(defn- cell [kind a b]
  (let [f (if (hook :construct) nu identity)]
    (pointer kind (bundle (bind (role :L) (f a)) (bind (role :R) (f b))))))

(defn car [c] (h/part (m :C) (role :L) c))
(defn cdr [c] (h/part (m :C) (role :R) c))

(defn- empty-coll? [v] (second (kind v)))

(defn- nil-vec [] (m :nil))
(defn- vnil? [v] (eq? v (nil-vec)))
(defn- truthy? [v] (not (or (vnil? v) (eq? v (m :false)))))
(defn- bool [x] (if x (m :true) (m :false)))

(defn- mk-seq [tag-kind xs]
  (reduce (fn [acc x] (cell tag-kind x acc)) (tag tag-kind) (reverse xs)))

;; [W1 hook] constructors: with superposed elements the hook enumerates worlds
(defn- construct [k f xs] (if-let [h (hook :construct)] (h k f xs) (f xs)))

(defn mk-list [xs] (construct :list #(mk-seq :list %) xs))

(declare mk-vec-plain)

(defn mk-vec
  "Vector of the elements `xs`: ν(Σᵢ Πⁱ(V⊗xᵢ)), stored in M as itself,
  labelled :vec, with its count N⊗n as the trace it points to. Position i is the permutation power Πⁱ; V keeps a
  nested vector's positions apart from its parent's, because a random
  permutation does not commute with binding: Πⁱ(V⊗Πʲ(V⊗y)) ≠ Πⁱ⁺ʲ(V⊗y)."
  [xs]
  (construct :vec mk-vec-plain xs))

(declare vec-elems)

(defn- cells
  "Host seq of the elements of a list cell chain, or of a vector. Every cell
  is a row of M, so a chain longer than M is a cycle (a misread cdr), not a
  list."
  ([c] (cells c (h/mem-size (m :M))))
  ([c budget]
   (lazy-seq
    (when-not (vnil? c)
      (let [[k e?] (kind c)]
        (cond
          e? nil
          (= k :vec) (vec-elems c)
          :else (do (when (neg? budget)
                      (throw (ex-info "cyclic cell chain" {})))
                    (cons (car c) (cells (cdr c) (dec budget))))))))))

(defn- distinct-by-eq [xs]
  (reduce (fn [acc x] (if (some #(eq? x %) acc) acc (conj acc x))) [] xs))

(declare num-vec)

;; Superpositions need near-orthogonal constituents. Integers are not (in the
;; residue encoding, n and n+p agree on p's band), and neither are vectors
;; (they are similar by design), so both are boxed behind a hash-consed
;; pointer wherever they enter a map or set.

(defn- slot [x] (if (#{:num :vec} (kind-of x)) (pointer :box x) x))
(defn- unslot [y] (if (= :box (kind-of y)) (cleanup (deref-ptr y)) y))

(defn- permute [v k] (h/permute (sp) v k))

;; Binding commutes, so with entries k⊗v the entry 1⊗2 = 2⊗1 answers the
;; probe 2 with 1 at full strength: (get {1 2 2 3} 2) returned 1. An entry
;; is therefore k⊗Π(v), directed by the fixed random permutation Π. A cyclic
;; shift would not do: it commutes with circular convolution, so
;; shift(1)⊗2 = shift(2)⊗1. Π goes on the value, not the key: keys are
;; unitary, so unbinding the key stays exact, and a permuted key is not
;; unitary. Binding and permuting are one op (hdc bind_dir): as two, op
;; noise compounds before any cleanup, and maps failed from σ = 2 instead
;; of about 3.

(defn- entry [k v] (h/bind-dir (sp) k v))
(defn- entry-value [trace k] (h/unbind-dir (sp) k trace))

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
  one's share 1/√(parts·n) of the trace must stay 1.25× above the noise
  floor. A vector counts as 1.5 parts (calibrated, see vec-parts)."
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
      ;; every constituent is a row of M, so a larger count is a misread
      ;; (an overloaded or noisy trace), not a size to peel that many times
      (if (<= 0 n (h/mem-size (m :M)))
        n
        (throw (ex-info (str "unreadable collection size " n) {:n n}))))))

(declare mk-map-plain mk-set-plain)

(defn mk-map
  "Map from a seq of [k v] vector pairs; later keys win.
  Trace: ν(ν(Σ k⊗Π(v)) + ν(K⊗Σ k) + C⊗n)."
  [entries]
  (construct :map mk-map-plain entries))

(defn- mk-map-plain [entries]
  (let [entries (reduce (fn [acc [k v]] (conj (filterv #(not (eq? k (first %))) acc) [k v]))
                        [] entries)
        _ (check-capacity "map" 3 (count entries))
        slots (mapv (fn [[k v]] [(slot k) (slot v)]) entries)]
    (if (empty? entries)
      (tag :map)
      (pointer :map
               (nu (bundle (nu (apply bundle (map (fn [[k v]] (entry k v)) slots)))
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
    (unslot (recall-slot (entry-value (deref-ptr mv) (slot k))))))

(defn- map-entries [mv]
  (let [trace (deref-ptr mv)]
    (mapv (fn [k] [k (unslot (recall-slot (entry-value trace (slot k))))]) (map-keys mv))))

(defn mk-set
  "Trace: ν(ν(Σ x) + C⊗n)."
  [xs]
  (construct :set mk-set-plain xs))

(defn- mk-set-plain [xs]
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

;; vectors: positional superpositions, stored as themselves
;;
;; The count N⊗n is not part of the vector: it is the trace the vector's
;; row points to, read through the memory. In the content, a count term
;; made every two vectors of the same length similar (0.2 even at weight
;; ½), and the :proj memory, which keeps every row above chance, then read
;; unrelated vectors into superpositions as spurious worlds. Under the
;; role C it also correlated vectors with maps and sets of that size.
;; Without it, unrelated vectors are orthogonal, and each element's share
;; is 1/√n.
;;
;; Capacity is calibrated, not derived: every element is one cleanup with
;; no explaining away. Keyword and integer elements read back exactly up to
;; 48 of them at D = 2048 and start failing at 56 (5 seeds). A vector counts
;; as 1.5 parts: about D/48, 43 at D = 2048.
;;
;; Integer elements are boxed, as in maps and sets: the residue readout
;; failed from 16 elements at D = 2048, where the codebook is still exact.
;; A box is hash-consed per value, so vectors sharing integers at the same
;; positions stay similar. Nested vectors are not boxed, so similarity
;; reaches into them.

(def ^:private vec-parts 1.5)

(defn- vec-slot [x] (if (= :num (kind-of x)) (pointer :box x) x))


(defn- mk-vec-plain [xs]
  (let [xs (vec xs)
        n (count xs)
        ;; as in `cell`: under vsc.worlds every element is unit length
        f (if (hook :construct) nu identity)]
    (if (zero? n)
      (tag :vec)
      (do
        (check-capacity "vector" vec-parts n)
        (let [v (nu (apply bundle (map-indexed (fn [i x] (permute (bind (role :V) (f (vec-slot x))) i)) xs)))]
          ;; equal vectors are the same vector already; the dedupe only
          ;; keeps M from storing a second copy
          (h/mem-get (m :M) (h/store-valued! (m :M) v (bind (role :N) (num-vec n))
                                             (kind->label :vec) (m :vec-dedupe))))))))

(defn- vec-size
  "Host count of a non-empty vector: M(N ⊘ M(v))."
  ^long [v]
  (let [[_ n] (h/recognize (m :C) (unbind (role :N) (deref-ptr v)))
        most (if (m :enforce-capacity?) (capacity vec-parts) (m :dim))]
    (if (<= 1 n most)
      n
      (throw (ex-info (str "unreadable vector size " n) {:n n})))))

(defn- vec-nth
  "Element i of vector v: M(V ⊘ Π⁻ⁱ(v)), unboxed. Linear up to the cleanup,
  so on a superposition of vectors it reads the superposition of their
  elements."
  [v i]
  (unslot (cleanup (unbind (role :V) (permute v (- i))))))

(defn- vec-elems
  "Host seq of the elements of a non-empty vector, read one by one (not
  chunked: `first` reads one element)."
  [v]
  (let [n (vec-size v)]
    ((fn step [i] (lazy-seq (when (< i n) (cons (vec-nth v i) (step (inc i)))))) 0)))

(defn- closure [form env] (cell :fn form env))

;; integers: n = B^n in a residue number system (see hdc.py, Numbers)

(defn- num-vec [n]
  (let [half (h/num-half-range (m :nums))]
    (if (<= (abs n) half)
      (h/num-vec (m :nums) n)
      (throw (ex-info (str "integer out of range ±" half ": " n) {:n n})))))

(defn- zero-vec [] (m :zero))
(defn- v-inc [x] (bind (m :base) x))
(defn- v-dec [x] (unbind (m :base) x))
;; the order matters: y ⊘ x first would pass through B^(x−y), and B^0 is a
;; delta at index 0 again; every intermediate here carries Z²
(defn- v-add [x y] (unbind (m :zero) (bind x y)))    ;; (B^x Z ⊗ B^y Z) ⊘ Z
(defn- v-sub [x y] (unbind y (bind (m :zero) x)))    ;; (Z ⊗ B^x Z) ⊘ B^y Z
;; eq?, not a similarity: n shares a band with 0 for every modulus dividing it
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

(declare decode-plain)

(defn decode
  "Vector → host Clojure data (the printer side)."
  [v]
  ;; [W1 hook] vsc.worlds prints superpositions as #worlds {value weight}
  (if-let [h (hook :decode)] (h v) (decode-plain v)))

(def int-print-range
  "Superposed integers are read back only against [-r, r]: beyond it the
  residue encoding has chimeras that match a superposition as well as its
  true worlds do (646657 for (+ (bundle 1 2) 10); docs/W0 section 2)."
  1024)

(defn- num-value
  "[W1] A clean integer by its residues (full ±range); a superposed one
  (similarity to its best single integer below 0.99) by a bounded-range
  readout, as #worlds {n weight} when it holds more than one world."
  [v]
  (let [[_ n s] (h/recognize (m :C) v)
        r int-print-range]
    (if (or (> s 0.99) (not (h/more-ints? (m :nums) v n (- r) r)))
      n
      (let [ws (->> (h/int-worlds (m :nums) v (- int-print-range) int-print-range 2.0)
                    (filterv (fn [[_ w]] (> w 0.02))))]
        (case (count ws)
          0 n
          1 (ffirst ws)
          (let [t (reduce + (map second ws))]
            (tagged-literal 'worlds (into {} (map (fn [[k w]] [k (/ (Math/round (* 1e4 (/ w t))) 1e4)])) ws))))))))

(defn decode-plain
  "decode without the superposition hook."
  [v]
  (let [[k e?] (kind v)
        atom-value #(second (lexicon-entry v))]
    (case k
      :nil nil
      :true true
      :false false
      :num (num-value v)
      (:sym :kw :str) (atom-value)
      :prim (tagged-literal 'prim (atom-value))
      :list (if e? () (apply list (map decode (cells v))))
      :vec (if e? [] (mapv decode (vec-elems v)))
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

;; A slot binds sym iff its L field cleans up to sym itself (eq?): the slot
;; ν(L⊗x + R⊗v) gives L⊘slot ≈ 0.71·x, and a slot of another symbol gives
;; chance similarity. No threshold on the 0.71 (P1: θ = 0.4 failed at
;; op noise 1.46 for every D).

(defn- global-slot [sym]
  (when (pos? (h/mem-size (m :F)))
    (let [[i _] (h/nearest (m :F) (bind (role :L) sym))
          slot (h/mem-get (m :F) i)]
      (when (eq? (unbind (role :L) slot) sym)
        [i slot]))))

(defn- global-value
  "F(L⊗sym) read out by the memory backend, then its R field cleaned up."
  [sym]
  (when (pos? (h/mem-size (m :F)))
    (let [slot (h/clean (m :F) (bind (role :L) sym))]
      (when (eq? (unbind (role :L) slot) sym)
        ;; [W1 hook] vsc.worlds reads a superposed value without crosstalk
        (if-let [h (hook :global)]
          (h sym slot)
          (cleanup (unbind (role :R) slot)))))))

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

(defn- special-form
  "The special form `head` names, if any: SF's best match for the head must
  be the head itself (eq?), so the decision is two argmaxes, not a
  threshold on a noisy similarity."
  [head]
  (let [[label i s] (h/recognize (m :C) head)]
    (when (and (= :sym (label->kind label)) (> s (chance z-recognise)))
      (let [atom (h/mem-get (m :M) i)
            [j sf] (h/recall (m :SF) atom)]
        (when (and (>= j 0) (eq? atom (h/mem-get (m :SF) j)))
          ;; [W1 hook] labels past special-forms are forms added by add-special!
          (or (get special-forms sf) (get (m :extra-specials) sf)))))))

(defn- eval-body [body env]
  (reduce (fn [_ form] (veval form env)) (nil-vec) (cells body)))

(defn- bind-params [env params args]
  (loop [env env ps (cells params) args args]
    (cond
      (empty? ps) (if (seq args) (fail "too many arguments for params " params) env)
      (eq? (first ps) (m :amp)) (env-extend env (second ps) (mk-list args))
      (empty? args) (fail "too few arguments for params " params)
      :else (recur (env-extend env (first ps) (first args)) (rest ps) (rest args)))))

(defn- branch
  "[W1 hook] Every two-way decision of if/cond/and/or: `tv` is the value of
  the test expression `test-form`, and each branch is a fn of the
  environment and of the test value as that branch sees it (vsc.worlds may
  run both, each with its own share of the worlds)."
  [env test-form tv then-fn else-fn]
  (if-let [h (hook :branch)]
    (h env test-form tv then-fn else-fn)
    (if (truthy? tv) (then-fn env tv) (else-fn env tv))))

(defn- eval-special [sf args env]
  (let [[a b c] (cells args)
        ;; and/or return the test value itself, as the branch sees it
        test-value (fn [_ v] v)]
    (case sf
      quote a
      if (branch env a (veval a env)
                 (fn [e _] (veval b e))
                 (fn [e _] (if c (veval c e) (nil-vec))))
      do (eval-body args env)
      def (define! a (veval b env))
      defn (define! a (closure (cell :list (m :fn-sym) args) (tag :list)))
      fn (closure (cell :list (m :fn-sym) args) env)
      let (let [env (reduce (fn [env [sym x]] (env-extend env sym (veval x env)))
                            env (partition 2 (cells a)))]
            (eval-body (cdr args) env))
      cond ((fn clauses [env cs]
              (if-let [[t x] (first cs)]
                (branch env t (veval t env)
                        (fn [e _] (veval x e))
                        (fn [e _] (clauses e (rest cs))))
                (nil-vec)))
            env (partition 2 (cells args)))
      and ((fn go [env xs]
             (if-let [[x & more] (seq xs)]
               (let [v (veval x env)]
                 (if more (branch env x v (fn [e _] (go e more)) test-value) v))
               (m :true)))
           env (cells args))
      or ((fn go [env xs]
            (if-let [[x & more] (seq xs)]
              (let [v (veval x env)]
                (if more (branch env x v test-value (fn [e _] (go e more))) v))
              (nil-vec)))
          env (cells args))
      ;; [W1 hook] forms added by add-special! (vsc.worlds: for-worlds)
      (if-let [h (hook :special)]
        (h sf args env)
        (fail "unknown special form " (encode sf))))))

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
      :vec (if e? e (mk-vec (mapv #(veval % env) (vec-elems e))))
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

(declare apply-plain)

(defn apply-fn [f args]
  ;; [W1 hook] vsc.worlds lifts or enumerates applications to superpositions
  (if-let [h (hook :apply)] (h f args) (apply-plain f args)))

(defn apply-plain
  "apply-fn without the superposition hook."
  [f args]
  (case (kind-of f)
    :prim (let [[i _ s] (h/recall (m :P) f)]
            (when (< s (chance z-recognise)) (fail "unknown primitive " f))
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
  (let [[k e?] (kind coll)]
    (cond
      (and (not e?) (#{:map :set} k)) (cleanup (unbind (role :C) (deref-ptr coll)))
      (and (not e?) (= :vec k)) (cleanup (unbind (role :N) (deref-ptr coll)))
      :else (reduce (fn [n _] (v-inc n)) (zero-vec) (as-cells coll)))))

(declare int-of)

(defn- vec-get
  "Element i of vector `coll`, or nil out of range. The index is read once
  (an integer cleanup) to choose the permutation power Π⁻ⁱ: a position is
  not a vector the machine could bind with."
  [coll i]
  (when (and (not (empty-coll? coll)) (= :num (kind-of i)))
    (let [j (int-of i)]
      (when (< -1 j (vec-size coll)) (vec-nth coll j)))))

(defn- v-nth [coll i]
  (if (= :vec (kind-of coll))
    (or (vec-get coll i) (fail "index out of bounds"))
    (loop [c coll i i]
      (cond
        (or (vnil? c) (empty-coll? c)) (fail "index out of bounds")
        (v-zero? i) (car c)
        :else (recur (cdr c) (v-dec i))))))

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
   ['rest (fn [[c]] (if (and (= :list (kind-of c)) (not (empty-coll? c)))
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
                 :vec (vec-get coll k)
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
   ['+ (fn [xs] (reduce v-add (zero-vec) xs))]
   ['- (fn [[x & ys]] (if ys (reduce v-sub x ys) (v-sub (zero-vec) x)))]
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

(defn- default-memory []
  (keyword (or (not-empty (System/getenv "VSC_MEMORY")) "codebook")))

(defn init!
  "Build a fresh machine: space, cleanup memories, roles, tags, numbers,
  primitives, then load the prelude (written in the vector-symbolic dialect).

  Options:
    :dim :seed :prelude?
    :memory        cleanup backend, :codebook (default, or $VSC_MEMORY),
                   :proj (superposition-preserving), :linear or :mhn
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
     ;; [W1 hook] $VSC_MEMORY picks the default backend (tests on :proj)
     :or {dim default-dim seed 42 prelude? true memory (default-memory) memory-opts {}
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
                      :roles (assoc (zipmap [:L :R :K :C] (repeatedly #(h/unitary s)))
                                    ;; drawn aside, so the main stream (and
                                    ;; every atom after it) is unchanged
                                    :V (h/aux-unitary s 1)
                                    :N (h/aux-unitary s 2))
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
              :base (h/num-step nums)
              :zero (h/num-offset nums)
              :C (h/cleanup-machine s M nums)))
     ;; Two vectors of n elements that differ in one have similarity
     ;; 1 − 1/n: above n = 200 (D ≳ 12800) that passes θ_eq, so the exact
     ;; copy thresholds follow the vector capacity
     (let [cap (capacity vec-parts)]
       (swap! machine assoc
              :theta-eq (max theta-eq (- 1.0 (/ 0.25 cap)))
              :vec-dedupe (max 0.999 (- 1.0 (/ 0.125 cap)))))
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

;; ---------------------------------------------------------------------------
;; [W1 hook] extension points for vsc.worlds

(defn add-primitive!
  "Register primitive `name` (a symbol) with host fn `f` of the argument
  vectors, and define it globally."
  [name f]
  (let [i (count (m :prims))
        pv (intern-atom! :prim name)]
    (h/mem-add! (m :P) pv i)
    (define! (encode name) pv)
    (swap! machine update :prims conj {:name name :f f})
    name))

(defn prim-name
  "The symbol of primitive vector `f`, or nil."
  [f]
  (let [[i _ s] (h/recall (m :P) f)]
    (when (> s (chance z-recognise)) (:name (nth (m :prims) i)))))

(defn add-special!
  "Recognise symbol `sym` as a special form; eval-special hands it to the
  :special hook."
  [sym]
  (let [label (+ (count special-forms) (count (m :extra-specials)))]
    (h/mem-add! (m :SF) (encode sym) label)
    (swap! machine update :extra-specials assoc label sym)
    sym))

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
