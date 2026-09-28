(ns vsc.hdc
  "Thin bridge to the Python HDC substrate (resources/vsc/hdc.py).

  Vectors are opaque Python (numpy) objects. Clojure never looks inside a
  vector; it only combines, compares and cleans them up through this ns."
  (:require
   [clojure.java.io :as io]
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

(defn memory
  "A fresh lookup-table cleanup memory."
  [dim]
  (py/call-attr @module "Memory" dim))

(defn numbers
  "Integer encoding n = B^n over space `s` (residue number system)."
  [s]
  (py/call-attr @module "Numbers" s))

(defn num-half-range ^long [nums] (py/get-attr nums "half"))

(defn num-vec [nums n] (py/call-attr nums "vec" n))

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
(defn mem-get [m i] (py/call-attr m "get" i))
(defn mem-label ^long [m i] (py/call-attr m "label" i))

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

(defn cleanup-machine [s mem nums] (py/call-attr @module "Cleanup" s mem nums))

(defn recognize
  "[label index similarity] of `v`; for integers, index is the integer."
  [c v]
  (vec (py/->jvm (py/call-attr c "recognize" v))))

(defn cleanup [c v] (py/call-attr c "cleanup" v))

(defn part
  "M(role ⊘ M(p))."
  [c role p]
  (py/call-attr c "part" role p))
