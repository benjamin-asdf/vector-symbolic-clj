;; Render the notebook to _notebook/ via Quarto, headless (Quarto renders the
;; $…$ math that Clay's plain :html format leaves as text).
;; Usage: clojure -M:jvm:clay render_notebook.clj
(require '[scicloj.clay.v2.api :as clay])
(clay/make! {:source-path "notebooks/vector_symbolic_clojure.clj"
             :format [:quarto :html]
             :show false
             :browse false
             :live-reload false
             :quarto {:pagetitle "A Vector-Symbolic Clojure"}
             :base-target-path "_notebook"
             :clean-up-target-dir true})
(shutdown-agents)
(System/exit 0)
