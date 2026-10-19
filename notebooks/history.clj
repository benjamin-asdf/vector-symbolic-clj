^:kindly/hide-code
(ns history)

;; # How the Vector-Symbolic Clojure Got Here
;;
;; > Generated with Claude Code, and I am not ashamed. My own blog posts of
;; > this kind are part of the training data and have sculpted its style.
;;
;; *A companion to [A Vector-Symbolic Clojure](../), which describes the
;; design as it is. This page records how it got that way: what broke,
;; roughly in the order we found it, and what changed. The overview
;; numbers its seven traps by theme, so the numbers here differ. September–October 2026.*
;;
;; The starting point was the design of Tomkins-Flanagan and Kelly's "Hey
;; Pentti, We Did It!" (2024): list cells as bundles of their parts,
;; binding by circular convolution, and thresholds "a little under 1 and a
;; little over 0". Each section below is one place where running real
;; programs disagreed with it, or with our own first choices. The paper
;; draft (§4) has the measurements, and `scripts/failure_modes.clj`
;; rebuilds each failure next to the current code.

;; ## 1. Bundled list cells
;;
;; Our first `(fact 5)` failed with "unable to resolve symbol: a". Its list
;; cells were the paper's: a cell for (a . b) was just L⊗a + R⊗b. Recursion
;; builds near-copies such as ((a . 5) . ()) and ((a . 4) . ()), 89%
;; similar as bundles, so a read beat the wrong sibling by only 0.05. We
;; did not isolate which lookup failed before replacing the cells. A
;; rebuild on the current code shows the same message in 3 of 3 seeds at
;; noise σ = 2, where cells with IDs give 120.
;;
;; The change: every cell got a random pointer, shared by cells with equal
;; contents (hash-consing). The margin rose to 0.66 and the noise tolerated
;; by 2.8×.

;; ## 2. Integers: from a table to clocks, and the chimeras
;;
;; Integers started as powers Bⁿ of one vector, recognised by scanning the
;; whole integer range on every cleanup. That scan dominated the run time.
;; Following Kymn et al. (2025) and Hanley et al. (2025), the encoding moved
;; to residues: one band of the vector per prime, read like six clocks.
;; Cleanup became 72 checks instead of a scan.
;;
;; The clocks brought chimeras: a sum of 11 and 12 matches 62 fake numbers
;; exactly as well as its members. The readout over integers that showed
;; this was replaced before the first commit, so the end-to-end failure no
;; longer reproduces. The fix stayed: integers inside maps and sets sit
;; behind their own pointers.

;; ## 3. Map entries
;;
;; On the first day, a benchmark task found `(get {1 2 2 3} 2)` returning
;; 1: the entry 1 ↦ 2 was 1⊗2, which is also 2⊗1. The first fix stored
;; every map value behind its own pointer, which made the pair directed at
;; the cost of one extra memory row per value. In October a directed bind,
;; k⊗Π(v), replaced it. Done as two operations, a bind and then a permutation, it made
;; maps fail from noise σ = 2 instead of about 3, because the substrate's
;; noise compounds between operations. As one operation it matches the old
;; robustness.

;; ## 4. Zero in one dimension
;;
;; B⁰ is the identity of convolution, a vector with all its weight in
;; dimension 0. Damaging that one dimension, at a lesion of 0.5%, destroyed
;; all arithmetic. In review we also saw that 5005 = 5·7·11·13 is 4/6
;; similar to zero on the clocks, so a threshold `zero?` would accept it.
;;
;; The change: n = Bⁿ⊗Z with a random Z, so zero is spread over all
;; dimensions; `zero?` and `=` compare identities. `(+ 17 25)` then came
;; out right in 300 of 300 runs up to a 10% lesion.

;; ## 5. Fixed thresholds
;;
;; Every stored structure failed at noise σ ≈ 1.5–1.7, for every vector
;; size, while the cleanup underneath survived σ ≈ 17. The interpreter
;; recognised a value's kind at cosine > 0.5 and a global at > 0.4, and
;; those cutoffs do not grow with the dimension. A 4-list round trip
;; worked 10 of 10 times at σ = 1 and 0 of 10 from σ = 2.
;;
;; The change: decide by argmax, or relative to the chance level 1/√D.
;; The same round trip now passes 10 of 10 up to σ = 3. One cutoff was
;; missed and fixed later, in naming the primitive of a superposed call.
;; The CEK machine's own datapath still has fixed cutoffs: 0.5 in its PRIM
;; instruction, 0.6 to decide a jump or a halt, and 0.4 as the floor under
;; the nearest rule key, below which no rule matches.

;; ## 6. Vectors as pointer chains, then as superpositions
;;
;; Clojure vectors started as lists: chains of pointer cells. In October
;; they became positional superpositions, ν(Σᵢ Πⁱ(V⊗xᵢ)), so that vectors
;; with the same elements in the same places are similar and `nth` is one
;; readout. Lists stayed pointer chains. Which of the two to use became the
;; programmer's choice.
;;
;; The first version kept the vector's length as a field inside it, under
;; the same role that maps and sets use for their size. Reading the set
;; `#{[1 2] [1 3]}` then pulled stored vectors in as members. Under a role
;; of its own, any two vectors of one length were still 20% alike, which
;; the superposition memory took as present. The length moved out of the
;; vector, next to it in the memory.

;; ## 7. Tuples of worlds
;;
;; With vectors as superpositions, the dice and sprinkler demos built
;; their joint worlds as vectors, and read back far from the exact joints
;; (total variation 0.43–0.82) without any capacity error. A sum of
;; vectors keeps each position's distribution and loses which values went
;; together: [1 1] + [2 2] is [1 2] + [2 1]. Joint worlds went back to
;; lists.

;; ## 8. Hopfield networks
;;
;; The original paper suggests a modern Hopfield network as the cleanup
;; memory. We tried it on unrelated items, on look-alikes and under damage:
;; it ties the lookup table at best and loses on look-alikes, and a
;; superposition needed a different memory altogether. The overview has
;; the numbers.
;;
;; -----
;;
;; *Generated by Claude on Benjamin Schwerdtner's behalf.*
