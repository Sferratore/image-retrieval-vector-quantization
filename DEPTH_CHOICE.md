# TSVQ Depth Selection

## Overview

The TSVQ tree depth `d` controls how many leaf codewords are produced per region:
2^d leaves total. A deeper tree means more codewords and finer discrimination, but
also fewer training vectors per leaf — making each binary split less reliable.

We tested every depth from d=2 to d=6 and compared the results against the original
flat VQ baseline (K=12 centroids per region).

---

## Results

| Category   | Flat VQ | d=2    | d=3    | d=4    | d=5    | d=6    |
|------------|---------|--------|--------|--------|--------|--------|
| dinosaurs  | 0.9096  | 0.9722 | 0.9327 | 0.8785 | 0.8278 | 0.7949 |
| horses     | 0.5214  | 0.5205 | 0.5224 | 0.5275 | 0.5185 | 0.5042 |
| buses      | 0.4693  | 0.3464 | 0.4035 | 0.4285 | 0.4343 | 0.4381 |
| africans   | 0.4685  | 0.4554 | 0.4720 | 0.4692 | 0.4713 | 0.4704 |
| food       | 0.3958  | 0.3518 | 0.3752 | 0.3972 | 0.4114 | 0.4150 |
| flowers    | 0.3600  | 0.4251 | 0.3833 | 0.3526 | 0.3296 | 0.3104 |
| elephants  | 0.3240  | 0.3454 | 0.3355 | 0.3272 | 0.3183 | 0.3096 |
| mountains  | 0.2685  | 0.2845 | 0.2835 | 0.2776 | 0.2674 | 0.2570 |
| buildings  | 0.2599  | 0.2569 | 0.2690 | 0.2678 | 0.2636 | 0.2569 |
| beaches    | 0.2263  | 0.2648 | 0.2454 | 0.2319 | 0.2196 | 0.2103 |
| **MAP**    | 0.4203  | **0.4223** | **0.4223** | 0.4158 | 0.4062 | 0.3967 |

---

## Observations

**d=2 and d=3 are tied at MAP=0.4223**, both beating the flat VQ baseline (0.4203).
From d=4 onward, performance drops monotonically. d=6 is the worst result (0.3967).

Two opposing trends drive the results across depths:

- **Categories with distinctive color** (dinosaurs, flowers, beaches, elephants) perform
  better at shallow depths. Their color distributions are compact enough that a small
  number of well-placed codewords capture them accurately. Deeper trees split these
  compact clusters into too many tiny fragments, degrading the codebook.

- **Categories with complex color distributions** (buses, food, africans) perform better
  at deeper depths. They need more codewords to represent their variety. However, beyond
  d=5, even these categories stop improving as leaves become too sparse to split reliably.

---

## Why d=3

d=2 and d=3 produce the same overall MAP, but d=3 is the better practical choice for
one key reason: **split stability**.

Each region contains approximately 256 blocks. With d=2 (4 leaves), each leaf is built
from ~64 vectors on average. With d=3 (8 leaves), each leaf is built from ~32 vectors.
Both are reasonable. However, d=3 produces a more expressive codebook — 8 codewords
instead of 4 — which captures more variation within each region without sacrificing
stability.

At d=2, the codebook is so coarse that many distinct blocks end up assigned to the same
codeword, reducing discriminative power even if the average MAP happens to match. d=3
represents a better balance between codebook richness and split reliability.

**Final choice: d=3** (8 leaves per region, ~32 vectors per leaf).
