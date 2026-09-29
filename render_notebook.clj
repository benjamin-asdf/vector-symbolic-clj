;; Render a notebook to _notebook/ via Quarto, headless (Quarto renders the
;; $…$ math that Clay's plain :html format leaves as text).
;; Usage: clojure -M:jvm:clay render_notebook.clj [notebook.clj]
;; Default: the overview, notebooks/vector_symbolic_clojure.clj.
(require '[scicloj.clay.v2.api :as clay])
(let [path (or (first *command-line-args*) "notebooks/vector_symbolic_clojure.clj")
      title ({"notebooks/vector_symbolic_clojure.clj" "A Vector-Symbolic Clojure"
              "notebooks/building_the_interpreter.clj" "Building a Vector-Symbolic Interpreter"}
             path)]
  (clay/make! {:source-path path
               :format [:quarto :html]
               :show false
               :browse false
               :live-reload false
               :quarto (cond-> {} title (assoc :pagetitle title))
               :base-target-path "_notebook"
               ;; both notebooks share the target: never wipe it
               :clean-up-target-dir false}))
(shutdown-agents)
(System/exit 0)
