;; W0 spike: what the *current* dialect already does with superpositions.
;;
;; No new language code: `bundle`, `bind`, `similarity` and `cleanup` are
;; already primitives. This shows, at the language level, the two facts the
;; W0 design rests on:
;;   1. arithmetic lifts exactly: + is ⊗, and ⊗ distributes over bundling
;;   2. the pointer memory is a hardmax (codebook) memory, so every
;;      structural op on a superposition collapses it to one world
;;
;; Run: VSC_PYTHON=.../.venv/bin/python clojure -M:jvm -i spikes/w0/amb_demo.clj

(require '[vsc.core :as vsc])

(vsc/init! {:dim 2048 :seed 1})

(defn show [src]
  (let [v (try (pr-str (vsc/run-string src))
               (catch Exception e (str "ERROR " (ex-message e))))]
    (println (format "%-58s => %s" src v))))

(println "-- 1. arithmetic lifting (similarity is printed ×100)")
(doseq [src ["(similarity (+ (bundle 1 2) 10) (bundle 11 12))"
             "(similarity (+ (bundle 1 2) 10) 11)"
             "(similarity (+ (bundle 1 2) 10) 12)"
             "(similarity (+ (bundle 1 2) 10) 13)"
             ;; the printer's single-number readout picks one world
             "(+ (bundle 1 2) 10)"
             ;; sharing: x+x on a superposition is run-time choice
             "(let [x (bundle 1 2)] (similarity (+ x x) 3))"
             "(let [x (bundle 1 2)] (similarity (+ x x) 2))"
             "(let [x (bundle 1 2)] (similarity (+ x x) 4))"]]
  (show src))

(println "-- 2. structural ops snap: the codebook pointer memory collapses")
(doseq [src ["(similarity (bundle (quote (a b)) (quote (c d))) (quote (a b)))"
             "(first (bundle (quote (a b)) (quote (c d))))"
             "(first (bundle (quote (a b)) (quote (c d)) (quote (c d))))"
             "(first (bundle (quote (a b)) (quote (a x))))"
             "(similarity (cleanup (bundle (quote a) (quote c))) (quote a))"]]
  (show src))

(println "-- 3. a superposed test: if is host control flow, so it picks one branch")
(doseq [src ["(similarity (bundle true false) true)"
             "(if (bundle true false) :then :else)"
             "(if (bundle false nil) :then :else)"]]
  (show src))
