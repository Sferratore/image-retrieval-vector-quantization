# UPDATE: Tree-Structured Vector Quantization (TSVQ)

## What Changes

The current system uses **flat Vector Quantization (VQ)**: for each spatial region of each database image, k-means is run with K=12 clusters, producing a flat list of 12 codewords. Every region is described at the same fixed level of detail regardless of its color complexity.

This update replaces flat VQ with **Tree-Structured Vector Quantization (TSVQ)**: instead of a flat codebook, a binary tree of codewords is built recursively. During query, the tree is traversed top-down to the leaves to find the nearest codeword.

Files affected:
- `src/build_codebooks.py` — tree building replaces flat k-means
- `src/retrieve.py` — tree traversal replaces flat nearest-codeword search

---

## How Flat VQ Works (Current)

Given all block feature vectors in a region:

1. Run k-means with K=12 → find 12 centroids all at once
2. Save those 12 centroids as the codebook

During query, for each query block vector:

1. Compute the distance to all 12 codewords
2. Take the nearest one
3. Record the squared distance as the quantization error

**Cost per block: K distance computations = 12**

---

## How TSVQ Works (New)

### Building the tree

Given all block feature vectors in a region:

1. **Level 0 (root):** run k-means with k=2 on all vectors.
   - k-means places 2 centroids at random positions, assigns every vector to the nearest centroid, then recomputes each centroid as the average of its assigned vectors. It repeats this until assignments stop changing.
   - To reduce the impact of the random starting positions, this whole process is run 5 times (`n_init=5`) from different starting points. The result with the **lowest distortion** (vectors closest to their centroids) is kept — lowest distortion means the two groups are the most compact and well-separated.
   - This produces 2 centroids and 2 groups of vectors.

2. **Level 1:** take each of the 2 groups separately and run k-means with k=2 on each, using the same `n_init=5` logic. Each group is split into its own 2 most natural subgroups → 4 centroids total.

3. **Level 2:** split each of the 4 groups again → 8 centroids.

4. **Continue until depth d:** at depth d the tree has 2^d leaf centroids.

At depth d the tree has 2^d leaf codewords. With d=4 we get 16 leaves — comparable to K=16 flat VQ — but the structure of those codewords is fundamentally different.
WARNING: The best depth is whichever gives the best result and so, needs to be tested. We expect it to rise at first (more codewords = better representation) and then plateau or drop when the tree gets too deep (overfitting to individual images, too few vectors per leaf to compute reliable centroids)

```
                    [root]
                   /      \
              [node]      [node]
             /     \      /    \
           [n]    [n]   [n]   [n]
           / \   / \   / \   / \
          L  L  L  L  L  L  L  L   ← leaves (codewords)
```

### Querying the tree

For each query block vector:

1. Start at the root
2. Compare the vector to the root's 2 children centroids → go to the closer one
3. At that node, compare to its 2 children → go to the closer one
4. Repeat until reaching a leaf
5. Record the squared distance to the leaf centroid as the quantization error

**Cost per block: 2 × depth = i.e. 2 × 4 = 8 distance computations instead of 16**

---

## What This Improves

### 1. Speed during query

In flat VQ, finding the nearest codeword for a block means computing the distance to every codeword and picking the smallest. With K=16 codewords, that is 16 distance computations per block.

In TSVQ, instead of checking all codewords at once, you walk the tree level by level. At each level you only compare against 2 centroids — the two children of the current node — and follow the closer one. With a tree of depth 4 (also 16 leaves), you only do 2 comparisons per level × 4 levels = 8 computations per block instead of 16.

The deeper the tree (the more codewords), the bigger the advantage:

| Depth | Leaves | Flat VQ cost | TSVQ cost |
|-------|--------|--------------|-----------|
| 3     | 8      | 8            | 6         |
| 4     | 16     | 16           | 8         |
| 5     | 32     | 32           | 10        |
| 6     | 64     | 64           | 12        |

### 2. More effective codebook structure

This is the most important improvement.

Both flat VQ and TSVQ use `n_init=5` — k-means is run 5 times from different random starting points and the result with the lowest distortion is kept. However, `n_init=5` is far more effective in TSVQ than in flat VQ because of the size of the problem it has to solve.

In flat VQ, those 5 attempts are trying to find the best arrangement of 12 clusters all at once across all 256 vectors. With 12 groups and random starting points, many combinations are possible and a bad initialization is hard to recover from — two codewords may end up redundantly close together while a large area of the feature space is left uncovered.

In TSVQ, each of those 5 attempts only needs to find the best split into 2 groups within an already well-defined subset of vectors. With only 2 groups, there are very few wrong arrangements possible and `n_init=5` is almost always sufficient to find the true best split.

Furthermore, each split operates on a progressively more coherent group.

## What Does Not Change

- Feature extraction (6-dim vectors: mean + variance per L, u, v channel)
- Normalization (NORM_SCALE applied in extract_block_features)
- Regional grid (4×4 = 16 regions per image)
- MSE aggregation (average across regions → single similarity score)
- Ranking (ascending MSE order)
- Evaluation (precision-recall, MAP)
