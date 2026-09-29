(ns vsc.hdc
  "Thin bridge to the Python HDC substrate (resources/vsc/hdc.py).

  Vectors are opaque Python (numpy) objects. Clojure never looks inside a
  vector; it only combines, compares and cleans them up through this ns."
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
   [libpython-clj2.python :as py]))

(defn- python-executable
  "The project venv's python (numpy wheels bundle OpenBLAS; a distro numpy on
  reference BLAS makes every cleanup ~40x slower), else whatever is on PATH."
  []
  (or (System/getenv "VSC_PYTHON")
      (let [venv (io/file ".venv/bin/python")]
        (when (.exists venv) (.getPath venv)))))

(defonce ^:private module
  (delay
    (if-let [exe (python-executable)]
      (py/initialize! :python-executable exe)
      (py/initialize!))
    (let [dir (-> (io/resource "vsc/hdc.py") io/file .getParent)
          sys (py/import-module "sys")]
      (py/call-attr (py/get-attr sys "path") "insert" 0 dir)
      (py/import-module "hdc"))))

(defn space
  "A fresh HRR space of dimension `dim`."
  [dim seed]
  (py/call-attr @module "Space" dim seed))

(defn- py-kw
  "Clojure option map -> Python keyword args (:op-noise -> op_noise)."
  [opts]
  (into {} (for [[k v] opts :when (some? v)]
             [(str/replace (name k) "-" "_") (if (keyword? v) (name v) v)])))

(defn configure!
  "Set the noise knobs of space `s`: :op-noise, :probe-noise, :lesion,
  :noise-seed. Missing knobs are off."
  [s knobs]
  (py/call-attr-kw s "configure" [] (py-kw (select-keys knobs [:op-noise :probe-noise :lesion :noise-seed]))))

(defn damage!
  "Add noise of norm ~`sigma` once to every stored row of every memory of `s`."
  [s sigma]
  (py/call-attr s "damage" sigma))

(defn memory
  "A fresh cleanup memory over space `s`. `backend` is :codebook (the lookup
  table), :linear or :mhn; `opts` go to the backend's constructor."
  ([s] (memory s :codebook {}))
  ([s backend opts]
   (py/call-attr-kw @module "make_memory" [s (name backend)] (py-kw opts))))

(defn numbers
  "Integer encoding n = B^n over space `s` (residue number system)."
  [s]
  (py/call-attr @module "Numbers" s))

(defn num-half-range ^long [nums] (py/get-attr nums "half"))

(defn num-vec [nums n] (py/call-attr nums "vec" n))

(defn num-step "B: inc is B ⊗ n." [nums] (py/call-attr nums "step"))

(defn num-offset "Z, the vector of 0: n = B^n ⊗ Z." [nums] (py/call-attr nums "offset"))

(defn read-num
  "[n similarity] for the integer vector nearest to `v`."
  [nums v]
  (vec (py/->jvm (py/call-attr nums "read" v))))

;; ---------------------------------------------------------------------------
;; algebra, parameterised by a space `s`

(defn unitary [s] (py/call-attr s "unitary"))
(defn identity-vec [s] (py/call-attr s "identity"))
(defn bind [s a b] (py/call-attr s "bind" a b))
(defn unbind [s a c] (py/call-attr s "unbind" a c))
(defn inverse [s a] (py/call-attr s "inverse" a))
(defn power [s a k] (py/call-attr s "power" a k))
(defn bundle [s & vs] (apply py/call-attr s "bundle" vs))
(defn normalize [s v] (py/call-attr s "normalize" v))
(defn sim ^double [s a b] (py/call-attr s "sim" a b))
(defn degrade [s v amount] (py/call-attr s "degrade" v amount))

;; ---------------------------------------------------------------------------
;; cleanup memory

(defn mem-size ^long [m] (py/call-attr m "size"))

(defn mem-add!
  "Store `v` under `label`; with `dedupe`, reuse a trace more similar than it."
  ([m v label] (py/call-attr m "add" v label nil))
  ([m v label dedupe] (py/call-attr m "add" v label dedupe)))

(defn mem-put! [m i v] (py/call-attr m "put" i v))
(defn mem-digest "sha1 of the stored rows." [m] (py/call-attr m "digest"))
(defn mem-get [m i] (py/call-attr m "get" i))
(defn mem-label ^long [m i] (py/call-attr m "label" i))

(defn clean
  "M(v): the memory backend's autoassociative readout (for the codebook, the
  nearest stored row)."
  [m v]
  (py/call-attr m "clean" v))

(defn add-random!
  "Store `k` fresh random unitary atoms under `label`; the new size."
  [m k label]
  (py/call-attr m "add_random" k label))

(defn nearest
  "[index similarity] of the closest trace."
  [m v]
  (vec (py/->jvm (py/call-attr m "nearest" v))))

(defn recall
  "[index label similarity] of the closest trace."
  [m v]
  (vec (py/->jvm (py/call-attr m "recall" v))))

(defn peel
  "[memory-index row] pairs of the traces superposed in `v`, found by
  explaining away over `mems`."
  ([mems v theta] (peel mems v theta 256))
  ([mems v theta limit]
   (mapv vec (py/->jvm (py/call-attr @module "peel" (py/->py-list mems) v theta limit)))))

(defn intern!
  "Index of the pointer for `trace`: an existing one for an equal trace of the
  same label, else the fresh `key`."
  [m key trace label]
  (py/call-attr m "intern" key trace label))

(defn deref-ptr
  "The trace pointer `p` refers to."
  [m p]
  (py/call-attr m "deref" p))

(defn cleanup-machine
  "The combined cleanup of item memory `mem` and integers `nums`: the plain
  Cleanup, or ProjCleanup (superposition-preserving) over a :proj memory."
  [s mem nums]
  (py/call-attr @module "make_cleanup" s mem nums))

(defn recognize
  "[label index similarity] of `v`; for integers, index is the integer."
  [c v]
  (vec (py/->jvm (py/call-attr c "recognize" v))))

(defn cleanup [c v] (py/call-attr c "cleanup" v))

(defn part
  "M(role ⊘ M(p))."
  [c role p]
  (py/call-attr c "part" role p))

;; ---------------------------------------------------------------------------
;; W1 superposition (hdc.py, section W1)

(defn readout
  "[worlds residual] of `v`: worlds a vector of [label index weight] (index
  is n itself for integers), found by matching pursuit over the item memory
  and the integers in [lo, hi]; residual the unexplained energy fraction."
  [c v {:keys [int-lo int-hi margin kmax] :or {int-lo -1024 int-hi 1024 margin 1.0 kmax 64}}]
  (let [[ws r] (py/->jvm (py/call-attr @module "readout" c v int-lo int-hi margin kmax))]
    [(mapv vec ws) r]))

(defn superposing?
  "Does cleanup `c` preserve superpositions (a ProjCleanup)?"
  [c]
  (py/has-attr? c "set_roles"))

(defn configure-cleanup!
  "ProjCleanup only: the integer readout range, and the cell roles L, R that
  make field reads exact under superposition."
  [c {:keys [int-lo int-hi roles]}]
  (when (superposing? c)
    (when int-lo (py/call-attr c "set_int_range" int-lo int-hi))
    (when roles (apply py/call-attr c "set_roles" roles))))

(defn int-worlds
  "[[n weight] ...] of an integer superposition, read against [lo, hi] only;
  `margin` is the detection margin in standard deviations."
  ([nums v lo hi] (int-worlds nums v lo hi 1.0))
  ([nums v lo hi margin]
   (mapv vec (py/->jvm (py/call-attr @module "int_worlds" nums v lo hi margin)))))

(defn associate!
  "Store row key -> trace in memory `m` as it is (no hash-consing)."
  [m key trace label]
  (py/call-attr @module "associate" m key trace label))

(defn follow
  "cleanup(deref(p)) in one lookup over every world of p; nil when p hits
  no stored row."
  [c p]
  (py/call-attr @module "follow" c p))

(defn more-ints?
  "Does integer vector `v` hold integer worlds in [lo, hi] beyond B^n?"
  [nums v n lo hi]
  (py/call-attr @module "more_ints" nums v n lo hi))

(defn lincomb
  "Σ wᵢ vᵢ (a bundle: op noise applies)."
  [s vs ws]
  (py/call-attr @module "lincomb" s (py/->py-list vs) (py/->py-list (map double ws))))

(defn coef
  "a·b/|b|: the coefficient of the unit direction b in a."
  ^double [a b]
  (py/call-attr @module "coef" a b))

;; ---------------------------------------------------------------------------
;; instrumentation

(defn instrument!
  "Switch op counting and the cleanup margin log of space `s` on or off."
  [s {:keys [count-ops? log-margins?]}]
  (py/call-attr s "instrument" count-ops? log-margins?))

(defn reset-instruments! [s] (py/call-attr s "reset_instruments"))

(defn budget!
  "Abort (a Python BudgetExceeded) past `max-ops` substrate ops, `max-rows`
  rows in any memory, or `seconds` from now; nil lifts a limit."
  [s {:keys [max-ops max-rows seconds]}]
  (py/call-attr s "budget" max-ops max-rows seconds))

(defn- jvm-map [x] (into {} (map (fn [[k v]] [(keyword (str/replace k "_" "-")) v])) (py/->jvm x)))

(defn op-counts
  "{kind count} of the substrate ops since the last reset. :rows is the
  number of stored rows scanned, the real cost of a table cleanup."
  [s]
  (jvm-map (py/call-attr s "get_counts")))

(defn margin-log
  "[kind winner top1 margin] of every logged cleanup, in order."
  [s]
  (mapv vec (py/->jvm (py/call-attr s "get_log"))))

(defn winner-log
  "[kind winner] of every logged cleanup, for diffs against a reference run."
  [s]
  (mapv vec (py/->jvm (py/call-attr s "log_indices"))))

(defn margin-stats
  "Summary of the margin log: :n :min :p05 :median :s1-min :s1-median."
  [s]
  (jvm-map (py/call-attr s "margin_stats")))

(defn bench
  "{op seconds-per-call} measured inside Python on memory `m` of space `s`."
  [s m k]
  (jvm-map (py/call-attr @module "bench" s m k)))

(defn collect!
  "Run Python's cyclic garbage collector."
  []
  (py/call-attr @module "collect"))

(defn versions
  "{:numpy version :blas description} of the substrate."
  []
  (jvm-map (py/call-attr @module "versions")))
