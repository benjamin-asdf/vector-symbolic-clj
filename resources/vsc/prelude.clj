;; The prelude is written in the vector-symbolic dialect itself: every form
;; below is encoded as a vector and evaluated in vector space.

(defn * [a b]
  (if (zero? b) 0 (+ a (* a (dec b)))))

(defn map [f xs]
  (if (empty? xs) () (cons (f (first xs)) (map f (rest xs)))))

(defn filter [p xs]
  (cond (empty? xs) ()
        (p (first xs)) (cons (first xs) (filter p (rest xs)))
        :else (filter p (rest xs))))

(defn reduce [f acc xs]
  (if (empty? xs) acc (reduce f (f acc (first xs)) (rest xs))))

(defn reverse [xs] (reduce (fn [acc x] (cons x acc)) () xs))

(defn concat [xs ys]
  (if (empty? xs) ys (cons (first xs) (concat (rest xs) ys))))

(defn range [n]
  (let [go (fn go [i acc] (if (zero? i) acc (go (dec i) (cons (dec i) acc))))]
    (go n ())))

(defn identity [x] x)

(defn comp [f g] (fn [& xs] (f (apply g xs))))

(defn partial [f & bound] (fn [& xs] (apply f (concat bound xs))))

(defn update [m k f] (assoc m k (f (get m k))))
