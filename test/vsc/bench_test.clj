(ns vsc.bench-test
  "Differential test of the P3 suite: every bench/*.clj task, run at noise 0
  on a fresh machine, must decode to the value its real-Clojure reference
  program produced (bench/expected.edn)."
  (:require
   [clojure.test :refer [deftest is testing]]
   [vsc.bench :as bench]))

(deftest every-task-matches-real-clojure
  (let [tasks (bench/tasks)]
    (is (= 12 (count tasks)))
    (doseq [[task {:keys [expected]}] tasks]
      (testing task
        (is (= expected (:value (bench/run-task task))))))))
