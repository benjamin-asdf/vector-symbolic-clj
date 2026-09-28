;; Real-Clojure equivalent of bench/metacircular.clj: the object programs run
;; directly as host lambdas. `k` comes from the bench file.
(fn [{:keys [k]}]
  [(((fn [x] (fn [y] (+ x y))) 3) 4)
   (((fn [f] (fn [n] ((f f) n)))
     (fn [self] (fn [n] (if (zero? n) 0 (+ n ((self self) (+ n -1)))))))
    k)])
