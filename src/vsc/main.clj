(ns vsc.main
  "Run files or a line REPL for the vector-symbolic Clojure."
  (:require
   [vsc.core :as vsc]))

(defn- report [form]
  (try
    (prn (vsc/run form))
    (catch clojure.lang.ExceptionInfo e
      (println "error:" (ex-message e)))))

(defn -main [& files]
  (vsc/init!)
  (if (seq files)
    (doseq [f files form (vsc/read-forms (slurp f))]
      (println "=>" (pr-str form))
      (report form))
    (do
      (println "vector-symbolic clojure — every value is one HRR vector. ctrl-d quits.")
      (loop []
        (print "vsc> ")
        (flush)
        (when-let [line (read-line)]
          (doseq [form (try (vsc/read-forms line)
                            (catch Exception e (println "read error:" (ex-message e)) []))]
            (report form))
          (recur)))))
  (shutdown-agents))
