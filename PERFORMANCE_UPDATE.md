# Performance Update — Flat VQ Retrieval

Two performance fixes were applied to `src/retrieve.py` and `src/evaluate.py` to reduce
the evaluation runtime from days to minutes. Neither fix changes the results — the output
is mathematically identical. They only change how efficiently the computation is carried out.

---

## Fix 1 — Eliminate Redundant Disk Reads

### The Problem

`evaluate.py` runs a full retrieval for every one of the 1000 database images used as queries.
Each retrieval call internally loaded all 1000 `.npz` codebook files from disk:

```python
# Inside retrieve() — called 1000 times by evaluate.py
for cb_file in CODEBOOKS_DIR.rglob("*.npz"):
    data = np.load(cb_file, allow_pickle=True)   # disk read every time
```

Since the database never changes between queries, this meant reading the same 1000 files
from disk 1000 times each — **1,000,000 total disk reads** for data that could have been
loaded once.

### The Fix

A new function `load_all_codebooks()` was added to `retrieve.py`. It reads all `.npz` files
once and returns them as a list of dicts in RAM:

```python
def load_all_codebooks():
    db = []
    for cb_file in CODEBOOKS_DIR.rglob("*.npz"):
        data = np.load(cb_file, allow_pickle=True)
        db.append({
            'codebooks': data['codebooks'],
            'stem':      cb_file.stem,
            'category':  cb_file.parent.name,
        })
    return db
```

`retrieve()` was updated to accept an optional `db` parameter. If provided, it iterates
over the in-memory list instead of hitting the disk. If not provided (standalone use),
it falls back to loading from disk automatically — so `retrieve.py` still works unchanged
when called on its own.

`evaluate.py` was updated to call `load_all_codebooks()` once before the query loop and
pass the result into every `retrieve()` call:

```python
db = load_all_codebooks()   # load once

for img_path in img_paths:
    results = retrieve(img_path, top_k=1000, db=db)   # iterate over RAM
```

### Impact

Disk reads drop from **1,000,000 to 1,000**. However, since disk I/O was a small fraction
of total runtime compared to the computation bottleneck described below, this fix alone
produces a modest wall-clock improvement. Its main value is correctness — redundant I/O
of identical data is always wasteful regardless of speed.

---

## Fix 2 — Vectorize the Distance Computation

### The Problem

`region_mse()` computed the quantization error for one spatial region by looping over
each query vector in Python and calling a numpy operation on each individually:

```python
def region_mse(query_vectors, codebook):
    total_error = 0.0
    for vec in query_vectors:                              # Python loop: 256 iterations
        squared_dists = np.sum((codebook - vec) ** 2, axis=1)  # op on shape (k,)
        total_error += squared_dists.min()
    return total_error / len(query_vectors)
```

With 256 blocks per region, 16 regions per image pair, and 1,000,000 image pairs in a
full evaluation, this amounts to roughly **4 billion Python loop iterations**, each
launching a small numpy call on a `(k,)` array.

The problem is not the math — numpy does the math fast. The problem is the **overhead of
each numpy call**: Python must look up the function, type-check the arguments, allocate
an output array, call into C, and return to Python. On a tiny `(k,)` array this setup
cost can exceed the actual computation time. Paying it 4 billion times is the bottleneck.

### The Fix

The loop was eliminated entirely. Instead of processing one vector at a time, all `n`
vectors in the region are compared to all `k` codewords simultaneously using numpy
broadcasting:

```python
def region_mse(query_vectors, codebook):
    # (n, 1, 6) - (k, 6) broadcasts to (n, k, 6) — all distances in one shot
    dists = np.sum((query_vectors[:, np.newaxis, :] - codebook) ** 2, axis=2)  # (n, k)
    return float(np.mean(dists.min(axis=1)))
```

`query_vectors[:, np.newaxis, :]` reshapes the `(n, 6)` array to `(n, 1, 6)`. NumPy
then broadcasts the subtraction against `codebook` of shape `(k, 6)`, producing an
`(n, k, 6)` array of differences. Summing over the last axis gives `(n, k)` squared
distances — every vector against every codeword in a single C call. `dists.min(axis=1)`
picks the nearest codeword for each of the `n` vectors, and `np.mean` averages them.

### Impact

The Python loop over 256 vectors is replaced by a single numpy operation. Instead of
256 numpy calls on `(k,)` arrays, there is 1 numpy call on an `(n, k)` array. The math
is identical; the overhead drops by a factor of ~256. Since the math in each call is negligible, setup time drops by 256x.

This is the dominant fix. Combined with Fix 1, the full evaluation runtime drops from
approximately **2 days to under an hour**.

---

## Summary

| Fix | What changed | Files touched |
|-----|-------------|---------------|
| 1 — Eliminate redundant disk reads | Added `load_all_codebooks()`; `retrieve()` accepts pre-loaded `db`; `evaluate.py` loads once before loop | `retrieve.py`, `evaluate.py` |
| 2 — Vectorize distance computation | Replaced Python loop over vectors in `region_mse()` with numpy broadcasting | `retrieve.py` |
