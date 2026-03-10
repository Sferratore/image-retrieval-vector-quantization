# Image Retrieval via Vector Quantization

A content-based image retrieval system built on **Vector Quantization (VQ)**. Given a query image, the system retrieves the most visually similar images from the Corel-1K database by comparing regional color codebooks.

---

## How It Works

The system represents each database image as a set of regional codebooks built from its block color statistics. To compare a query image against a database image, the query's block feature vectors are matched against the database image's codebooks and the mean squared error (MSE) is computed. A low MSE means the two images have similar color distributions in the same spatial regions.

### Phase 1 — Preprocessing

Each raw image is resized to a fixed **256×256** thumbnail and saved as a viewable JPEG. This standardizes all images to the same resolution before any feature extraction.

```
data/raw/Corel-1K/<category>/<stem>.jpg  →  data/processed/Corel-1K/<category>/<stem>.jpg
```

Script: `src/preprocess.py`

---

### Phase 2 — Feature Extraction

Each image is converted from RGB to **CIE L\*u\*v\*** color space, which is perceptually uniform (equal distances correspond to equal perceived color differences).

The image is then divided into non-overlapping **4×4 pixel blocks**. For each block, a **6-dimensional feature vector** is extracted:

```
[ mean_L,  mean_u,  mean_v,  var_L,  var_u,  var_v ]
```

- **Mean** per channel: the average color of the block
- **Variance** per channel: how uniform the color is inside the block

All features are normalized to the same scale using fixed theoretical maxima:

```
mean channels:     divided by 255    (max uint8 value)
variance channels: divided by 16256  (max possible variance = 255²/4)
```

This ensures that mean and variance contribute equally to distances. Without normalization, variance (range 0–16256) would dominate over mean (range 0–255).

---

### Phase 3 — Regional Codebook Building

The block grid is divided into a **4×4 spatial grid** of 16 regions. This encodes spatial information — a blue sky at the top of an image is treated differently from blue water at the bottom.

For each region of each database image, **k-means clustering** (K=12) is run on the block feature vectors in that region. k-means places 12 centroids at random positions, assigns every vector to its nearest centroid, then recomputes each centroid as the average of its assigned vectors, repeating until assignments stabilize. To reduce the impact of the random initialization, this whole process is run **5 times** (`n_init=5`) from different starting points and the result with the lowest distortion is kept. The resulting 12 cluster centers form the **codebook** for that region.

Each database image is therefore represented by **16 codebooks** (one per region), each containing 12 codewords of 6 features. These are saved to disk.

```
data/codebooks/Corel-1K/<category>/<stem>.npz
```

---

### Phase 4 — Query and Similarity Scoring

Given a query image:

1. Apply the same preprocessing (resize, BGR→Luv, block extraction, normalization)
2. Split into the same 4×4 regional grid
3. For each region, collect the raw block feature vectors — **no k-means is run on the query**

To compare the query against a database image:

- For each of the 16 regions, **quantize** the query's block vectors using the database image's regional codebook: find the nearest codeword (cluster center) for each query vector
- Compute the **Mean Squared Error (MSE)** between each query vector and its nearest codeword in the region. 
- Calculate region MSE average.
- Average region MSE of all 16 regions → single similarity score

**Lower score = the query is well compressed by the database image's codebook = higher similarity.**

---

### Phase 5 — Ranking

All 1000 database images are scored against the query and sorted by ascending MSE. The top-K results are returned with their image ID, category, and score.

---

## Evaluation

The system was evaluated on the full **Corel-1K** dataset (1000 images, 10 categories, 100 images each). Every image was used as a query and the remaining 999 images were ranked. Precision-recall was computed using standard **11-point interpolation**, averaged over all 1000 queries. Codebooks are built using **TSVQ with tree depth d = 4**, producing 2⁴ = 16 leaf codewords per region.

### Overall Precision-Recall

| Recall | Precision |
|--------|-----------|
| 0.0    | 0.8672    |
| 0.1    | 0.6258    |
| 0.2    | 0.5269    |
| 0.3    | 0.4665    |
| 0.4    | 0.4170    |
| 0.5    | 0.3754    |
| 0.6    | 0.3356    |
| 0.7    | 0.3004    |
| 0.8    | 0.2633    |
| 0.9    | 0.2270    |
| 1.0    | 0.1685    |

**Mean Average Precision (MAP): 0.4158**

### Per-Category MAP

| Category   | MAP    |
|------------|--------|
| dinosaurs  | 0.8785 |
| horses     | 0.5275 |
| buses      | 0.4285 |
| africans   | 0.4692 |
| food       | 0.3972 |
| flowers    | 0.3526 |
| elephants  | 0.3272 |
| mountains  | 0.2776 |
| buildings  | 0.2678 |
| beaches    | 0.2319 |

Categories with distinctive color distributions (dinosaurs, horses, buses) score significantly higher. Categories with overlapping color distributions (beaches, mountains, buildings) are harder to separate using color statistics alone.

---

## Project Structure

```
├── data/
│   ├── raw/           # Original Corel-1K images
│   ├── processed/     # Resized 256×256 thumbnails
│   ├── codebooks/     # Per-image regional VQ codebooks (.npz)
│   └── queries/       # Example query images per category
├── src/
│   ├── preprocess.py      # Phase 1: resize raw images
│   ├── build_codebooks.py # Phase 2: build regional VQ codebooks
│   ├── retrieve.py        # Phase 3: query and rank
│   └── evaluate.py        # Precision-recall evaluation
├── Pipfile
└── Pipfile.lock
```

---

## Installation

**Requirements:** Python 3.13, Pipenv

```bash
# Clone the repository
git clone https://github.com/Sferratore/image_retrieval_vector_quantization.git
cd image_retrieval_vector_quantization

# Install dependencies
pipenv install
```

---

## Usage

### 1. Preprocess images

```bash
pipenv run python src/preprocess.py
```

### 2. Build codebooks

```bash
pipenv run python src/build_codebooks.py
```

### 3. Retrieve similar images

Place a query image at `data/query.jpg`, then:

```bash
pipenv run python src/retrieve.py
```

### 4. Run full evaluation

```bash
pipenv run python src/evaluate.py
```

Outputs a precision-recall table, per-category MAP, and saves a plot to `precision_recall.png`.
