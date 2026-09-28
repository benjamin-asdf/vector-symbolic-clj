;; S1: a metacircular evaluator for the whole dialect, written in the dialect.
;;
;; Load this file at the host level; it defines `vsc-eval` and its helpers as
;; ordinary host globals. `(vsc-eval form env)` then evaluates `form` one level
;; up the tower: its closures, environments and global table are vector data
;; built and inspected by the host evaluator, which only runs this program.
;;
;; Representation
;;   frame        a VSA map {sym val ...} of at most `vsc-frame-size` names.
;;                `contains?` and `get` are one unbind and one cleanup each, so
;;                a frame is searched in O(1) host steps, not walked.
;;   environment  a list of frames, innermost first. A call pushes one frame
;;                holding all its params, built by a single `hash-map`; a
;;                `let` pushes a fresh frame and binds into it one name at a
;;                time, opening a new frame when one is full. Chaining frames
;;                lifts the capacity of a single map (~21 entries at D=2048)
;;                to any number of names in scope; only one call's params must
;;                fit in one map.
;;                Shadowing: inner frames first; within a frame, assoc
;;                replaces. Closures share the frames they capture.
;;   globals      `vsc-globals`, one frame list per level: `def`/`defn` inside
;;                vsc-eval bind there and never touch the host's F memory. A
;;                symbol found in neither falls back to the host's globals via
;;                `(eval sym)`: that is how primitives are reached.
;;   closure      (:vsc/closure name params body env), a tagged list; `name`
;;                is nil for an anonymous fn, else the fn's own name, which is
;;                bound to the closure on entry (named fns and defn).
;;   primitive    whatever non-closure value the host resolves; applied with
;;                the host `apply`. Keywords, maps and sets are callable the
;;                same way. `apply`, `eval` and `fn?` are intercepted, because
;;                they must see this level's closures and globals.
;;
;; The a-list alternative ((sym val) ...) was measured too (docs/results-S1.md):
;; it has no capacity limit at all, but every lookup is an interpreted walk
;; over all enclosing bindings, and every primitive reference walks all of
;; them plus every global.

(def vsc-special-forms (quote #{quote if do def defn fn let cond and or}))

(def vsc-frame-size 10)

(def vsc-globals ())

;; there is no `throw`: applying the message string makes the host fail with
;; "not a function: <message>"
(defn vsc-error [msg x] (msg x))

;; ---------------------------------------------------------------------------
;; environments

;; the value of `sym` in the innermost global frame binding it, else the
;; host's global of that name
(defn vsc-find-global [sym frames]
  (cond (empty? frames) (eval sym)
        (contains? (first frames) sym) (get (first frames) sym)
        :else (vsc-find-global sym (rest frames))))

;; the value of `sym` in the innermost frame binding it, then the globals
(defn vsc-lookup [sym frames]
  (cond (empty? frames) (vsc-find-global sym vsc-globals)
        (contains? (first frames) sym) (get (first frames) sym)
        :else (vsc-lookup sym (rest frames))))

;; bind `sym` in the innermost frame, or in a new one when that is full
(defn vsc-extend [frames sym val]
  (if (or (empty? frames) (= (count (first frames)) vsc-frame-size))
    (cons (hash-map sym val) frames)
    (cons (assoc (first frames) sym val) (rest frames))))

(defn vsc-define [sym val]
  (def vsc-globals (vsc-extend vsc-globals sym val))
  sym)

;; ---------------------------------------------------------------------------
;; closures

(defn vsc-closure [name params body env] (list :vsc/closure name params body env))

(defn vsc-closure? [f] (and (list? f) (= (first f) :vsc/closure)))

;; params and args → the flat (k v ...) list of one call frame, built with a
;; single hash-map (one call binds at most a map's capacity of names)
(defn vsc-bind-params [ps args kvs]
  (cond (empty? ps) (if (empty? args) kvs (vsc-error "too many arguments for params" ps))
        (= (first ps) (quote &)) (cons (nth ps 1) (cons args kvs))
        (empty? args) (vsc-error "too few arguments for params" ps)
        :else (vsc-bind-params (rest ps) (rest args) (cons (first ps) (cons (first args) kvs)))))

;; ---------------------------------------------------------------------------
;; eval

(defn vsc-evlis [xs env]
  (if (empty? xs) () (cons (vsc-eval (first xs) env) (vsc-evlis (rest xs) env))))

(defn vsc-eval-body [body env]
  (cond (empty? body) nil
        (empty? (rest body)) (vsc-eval (first body) env)
        :else (do (vsc-eval (first body) env) (vsc-eval-body (rest body) env))))

;; ([k v] ...) → (k' v' ...), evaluated, for one hash-map call
(defn vsc-eval-kvs [es env]
  (if (empty? es)
    ()
    (cons (vsc-eval (first (first es)) env)
          (cons (vsc-eval (nth (first es) 1) env) (vsc-eval-kvs (rest es) env)))))

(defn vsc-let-bind [bs env]
  (if (empty? bs)
    env
    (vsc-let-bind (rest (rest bs)) (vsc-extend env (first bs) (vsc-eval (nth bs 1) env)))))

(defn vsc-eval-cond [cs env]
  (cond (empty? cs) nil
        (vsc-eval (first cs) env) (vsc-eval (nth cs 1) env)
        :else (vsc-eval-cond (rest (rest cs)) env)))

(defn vsc-eval-and [xs env]
  (if (empty? xs)
    true
    (let [v (vsc-eval (first xs) env)]
      (if (and v (not (empty? (rest xs)))) (vsc-eval-and (rest xs) env) v))))

(defn vsc-eval-or [xs env]
  (if (empty? xs)
    nil
    (let [v (vsc-eval (first xs) env)]
      (if (or v (empty? (rest xs))) v (vsc-eval-or (rest xs) env)))))

(defn vsc-fn [args env]
  (if (symbol? (first args))
    (vsc-closure (first args) (nth args 1) (rest (rest args)) env)
    (vsc-closure nil (first args) (rest args) env)))

(defn vsc-eval-special [sf args env]
  (cond (= sf (quote if)) (if (vsc-eval (first args) env)
                            (vsc-eval (nth args 1) env)
                            (if (empty? (rest (rest args))) nil (vsc-eval (nth args 2) env)))
        (= sf (quote let)) (vsc-eval-body (rest args) (vsc-let-bind (first args) (cons {} env)))
        (= sf (quote fn)) (vsc-fn args env)
        (= sf (quote quote)) (first args)
        (= sf (quote cond)) (vsc-eval-cond args env)
        (= sf (quote and)) (vsc-eval-and args env)
        (= sf (quote or)) (vsc-eval-or args env)
        (= sf (quote do)) (vsc-eval-body args env)
        ;; defn closes over the empty env, like the host: globals are dynamic
        (= sf (quote defn)) (vsc-define (first args) (vsc-fn args ()))
        :else (vsc-define (first args) (vsc-eval (nth args 1) env))))

(defn vsc-eval [e env]
  (cond (symbol? e) (vsc-lookup e env)
        (list? e) (cond (empty? e) e
                        (contains? vsc-special-forms (first e))
                        (vsc-eval-special (first e) (rest e) env)
                        :else (vsc-apply (vsc-eval (first e) env) (vsc-evlis (rest e) env)))
        (vector? e) (vec (vsc-evlis e env))
        (map? e) (apply hash-map (vsc-eval-kvs (seq e) env))
        (set? e) (set (vsc-evlis e env))
        :else e))

;; ---------------------------------------------------------------------------
;; apply

;; (a b [c d]) → (a b c d)
(defn vsc-spread [xs]
  (if (empty? (rest xs))
    (apply list (first xs))
    (cons (first xs) (vsc-spread (rest xs)))))

;; the fn's own name goes first in the frame, so a param of the same name wins
(defn vsc-call-frame [name f ps args]
  (apply hash-map (if name (cons name (cons f (vsc-bind-params ps args ()))) (vsc-bind-params ps args ()))))

(defn vsc-apply-closure [f args]
  (vsc-eval-body (nth f 3) (cons (vsc-call-frame (nth f 1) f (nth f 2) args) (nth f 4))))

;; the host primitives that must see this level's closures and globals
(def vsc-hooks (hash-set apply eval fn?))

(defn vsc-apply-hook [f args]
  (cond (= f apply) (vsc-apply (first args) (vsc-spread (rest args)))
        (= f eval) (vsc-eval (first args) ())
        :else (or (vsc-closure? (first args)) (fn? (first args)))))

;; vsc-closure? is inlined here: this is the hottest path after lookup
(defn vsc-apply [f args]
  (cond (and (list? f) (= (first f) :vsc/closure)) (vsc-apply-closure f args)
        (contains? vsc-hooks f) (vsc-apply-hook f args)
        :else (apply f args)))

;; evaluate a sequence of top-level forms at this level, returning the last value
(defn vsc-run-all [forms] (vsc-eval-body forms ()))
