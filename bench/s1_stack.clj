;; S1 stack needs: the deepest (countdown n) that fits in a thread with a
;; small, fixed stack, for the host evaluator and for S1 on top of it. Host
;; Clojure. Run with
;;
;;   VSC_PYTHON=.venv/bin/python clojure -M:jvm -i bench/s1_stack.clj

(require '[vsc.core :as vsc] '[clojure.java.io :as io])

(vsc/init!)
(vsc/run-string (slurp (io/resource "vsc/eval.clj")))
(vsc/run (list 'vsc-run-all (list 'quote (vsc/read-forms (slurp (io/resource "vsc/prelude.clj"))))))

(def countdown '(defn countdown [n] (if (zero? n) 0 (countdown (dec n)))))
(defn s1-form [form] (list 'vsc-eval (list 'quote form) ()))
(vsc/run countdown)
(vsc/run (s1-form countdown))

(defn fits?
  "Does evaluating `form` finish in a thread with a `kb` KiB stack?"
  [kb form]
  (let [result (promise)
        t (Thread. nil
                   ;; an overflow inside a Python call surfaces wrapped, not
                   ;; as a StackOverflowError, so any throwable counts
                   #(deliver result (try (vsc/run form) :ok
                                         (catch Throwable e
                                           (binding [*out* *err*]
                                             (println "  fails:" (pr-str form)
                                                      (.getName (class e))))
                                           :overflow)))
                   "s1-stack" (* 1024 kb))]
    (.start t)
    (.join t)
    (= :ok @result)))

(defn deepest
  "Largest n with (countdown n) fitting: gallop up from 1, then bisect.
  Probes stay cheap until the first failure, and every probe grows M, so
  starting low matters more than the number of probes."
  [kb wrap]
  (let [ok? #(fits? kb (wrap (list 'countdown %)))
        [lo hi] (loop [n 1] (if (ok? n) (recur (* 2 n)) [(quot n 2) (dec n)]))]
    (loop [lo lo hi hi]
      (if (>= lo hi)
        lo
        (let [mid (quot (+ lo hi 1) 2)]
          (if (ok? mid) (recur mid hi) (recur lo (dec mid))))))))

(println "| stack | host depth | S1 depth | host / S1 |")
(println "|---|---|---|---|")
;; one stack size only: every probe grows M and nothing collects it, so later
;; probes get slower. A 256 + 1024 KiB sweep was stopped after 14 h, with the
;; host at 1 MiB bracketed to 1024 <= depth < 1280 (docs/results-S1.md). A
;; further point needs a fresh vsc/init! per probe and a time limit.
(doseq [kb [256]]
  (let [host (deepest kb identity)
        s1 (deepest kb s1-form)]
    (println (format "| %d KiB | %d | %d | %.1f |" kb host s1 (/ (double host) (max s1 1))))
    (flush)))

(shutdown-agents)
