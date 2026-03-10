import numpy as np
import cv2
import sys
from pathlib import Path

# --- Paths ---
ROOT          = Path(__file__).parent.parent
CODEBOOKS_DIR = ROOT / "data" / "codebooks"

# --- Parameters: must match build_codebooks.py exactly ---
# If any of these differ from what was used to build the codebooks,
# the feature vectors will be incompatible and results will be meaningless.
IMG_SIZE   = 256   # thumbnail side in pixels
BLOCK_SIZE = 4     # block side in pixels
GRID       = 4     # spatial grid is GRID x GRID regions
DEPTH      = 4     # TSVQ tree depth: must match build_codebooks.py

# Fixed normalization scale — must match build_codebooks.py exactly.
# Divides each feature by its theoretical maximum so all 6 dimensions
# are in roughly the same [0, 1] range and contribute equally to distances.
NORM_SCALE = np.array([255, 255, 255, 16256, 16256, 16256], dtype=np.float64)


# ===========================================================================
# STEP 1 — PREPROCESSING
# Identical pipeline to preprocess.py.
# The query must go through the exact same transformations as the DB images.
# ===========================================================================

def preprocess(img_path):
    """
    Load an image from disk, resize it to IMG_SIZE x IMG_SIZE,
    and convert it from BGR to CIE Luv* color space.

    This must be identical to what was done on the database images,
    otherwise the feature vectors are not comparable.

    Returns:
        img: np.ndarray of shape (IMG_SIZE, IMG_SIZE, 3) in Luv color space
    """
    # Load the image from disk as a BGR array
    img = cv2.imread(str(img_path))

    # Resize to the fixed thumbnail size used during DB preprocessing
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert from BGR (OpenCV default) to CIE Luv* color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

    return img


# ===========================================================================
# STEP 2 — FEATURE EXTRACTION
# Identical to build_codebooks.py.
# Divide the image into blocks and describe each block with 9 numbers.
# ===========================================================================

def extract_block_features(img):
    """
    Divide the image into non-overlapping BLOCK_SIZE x BLOCK_SIZE blocks
    and compute a 6-dim feature vector per block: [mean_L, mean_u, mean_v, var_L, var_u, var_v].

    Returns:
        np.ndarray of shape (n_blocks, 6)
    """
    h, w, _ = img.shape

    # Collect one feature vector per block
    feats = []

    # Slide a BLOCK_SIZE x BLOCK_SIZE window over the image without overlap
    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):

            # Crop the current block from the image
            block = img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            # Mean color per channel across the block's pixels → [mean_L, mean_u, mean_v]
            mean = block.mean(axis=(0, 1))

            # Color variance per channel across the block's pixels → [var_L, var_u, var_v]
            var  = block.var(axis=(0, 1))

            # Combine into one 6-dim feature vector and store it
            feats.append(np.concatenate([mean, var]))

    # Stack all vectors into a 2D array: one row per block
    feats = np.array(feats)

    # Divide each feature by its known maximum so all dimensions are in ~[0, 1].
    # Must use the same NORM_SCALE as build_codebooks.py.
    return feats / NORM_SCALE


def assign_regions(n_blocks_y, n_blocks_x):
    """
    Return a 2D map of shape (n_blocks_y, n_blocks_x) where each cell holds
    the region ID (0 to GRID*GRID-1) of that block.
    """
    # Start with a grid of zeros, one cell per block
    region_map = np.zeros((n_blocks_y, n_blocks_x), dtype=int)

    # Number of block-rows and block-columns that fit in one region
    ry = n_blocks_y // GRID
    rx = n_blocks_x // GRID

    r = 0

    # Iterate over the GRID x GRID spatial regions (row by row, left to right)
    for gy in range(GRID):
        for gx in range(GRID):

            # Compute the block-index boundaries of this region
            y0, y1 = gy * ry, (gy + 1) * ry
            x0, x1 = gx * rx, (gx + 1) * rx

            # Stamp all blocks inside this region with the current region ID
            region_map[y0:y1, x0:x1] = r
            r += 1

    return region_map


# ===========================================================================
# STEP 3 — QUERY SIGNATURE
# Extract per-region feature vectors for the query.
# Unlike the DB pipeline, we do NOT run k-means here — we keep the raw
# block vectors so we can quantize them against each DB codebook later.
# ===========================================================================

def get_query_signature(img):
    """
    Compute the query "signature": a list of GRID*GRID arrays, one per region.
    Each array contains the raw feature vectors of all blocks in that region.

    Returns:
        signature: list of GRID*GRID arrays, each of shape (n_blocks_in_region, 6)
    """
    h, w, _ = img.shape
    n_blocks_y = h // BLOCK_SIZE
    n_blocks_x = w // BLOCK_SIZE

    # Extract all block feature vectors for the query image: (n_blocks, 6)
    features = extract_block_features(img)

    # Get the region ID for every block, flattened to 1D: (n_blocks,)
    region_ids = assign_regions(n_blocks_y, n_blocks_x).flatten()

    # Group feature vectors by region ID
    signature = []
    for r in range(GRID * GRID):

        # Select only the blocks that belong to region r
        region_feats = features[region_ids == r]   # shape: (n_blocks_in_region, 6)
        signature.append(region_feats)

    return signature


# ===========================================================================
# STEP 4 — SIMILARITY SCORE
# Compare the query signature against one DB image's codebooks.
# For each region, quantize the query vectors using the DB codebook and
# measure the reconstruction error (MSE). Lower error = more similar.
# ===========================================================================

def region_mse(query_vectors, tree):
    """
    Quantize all query vectors in a region by traversing the TSVQ tree in
    parallel and return the MSE.

    All 256 vectors descend the tree together at each depth step using numpy
    broadcasting — no Python loop over individual vectors.

    Query vectors are already normalized by NORM_SCALE (done in extract_block_features),
    and so are the tree centroids (built from normalized features). The spaces match.

    Args:
        query_vectors: np.ndarray (n, 6) — already normalized
        tree:          np.ndarray (n_nodes, 6) — TSVQ flat tree in the same normalized space

    Returns:
        float: mean squared quantization error for this region
    """
    # All vectors start at the root (index 0)
    node_indices = np.zeros(len(query_vectors), dtype=np.intp)  # shape (n,)

    for _ in range(DEPTH):
        left  = 2 * node_indices + 1                                      # shape (n,)
        right = 2 * node_indices + 2                                      # shape (n,)

        dist_left  = np.sum((tree[left]  - query_vectors) ** 2, axis=1)  # shape (n,)
        dist_right = np.sum((tree[right] - query_vectors) ** 2, axis=1)  # shape (n,)

        # Each vector independently descends toward its nearer child
        node_indices = np.where(dist_left <= dist_right, left, right)    # shape (n,)

    # Compute squared distance from each vector to its leaf codeword
    leaf_vecs = tree[node_indices]                                        # shape (n, 6)
    return float(np.mean(np.sum((leaf_vecs - query_vectors) ** 2, axis=1)))


def score_against_db_image(query_signature, codebooks):
    """
    Compute the overall similarity score between the query and one DB image
    by averaging the MSE across all GRID*GRID regions.

    Args:
        query_signature: list of GRID*GRID arrays (query feature vectors per region)
        codebooks:       list of GRID*GRID arrays (codewords per region, from .npz)

    Returns:
        float: mean regional MSE (lower = more similar)
    """
    regional_mses = []

    # Compute MSE independently for each spatial region
    for r in range(GRID * GRID):
        mse = region_mse(query_signature[r], codebooks[r])
        regional_mses.append(mse)

    # Aggregate: simple mean across all regions
    return float(np.mean(regional_mses))


# ===========================================================================
# STEP 5 — RETRIEVAL
# Run the full pipeline: preprocess the query, compute scores against all
# DB images, sort by ascending score, return the top-k results.
# ===========================================================================

def load_all_codebooks():
    """
    Load every database codebook from disk into a list in memory.

    Call this once before a batch of queries so that each retrieve() call
    iterates over RAM instead of re-reading the same files from disk.

    Returns:
        list of dicts, each with keys: 'codebooks', 'stem', 'category'
    """
    db = []
    for cb_file in CODEBOOKS_DIR.rglob("*.npz"):
        data = np.load(cb_file, allow_pickle=True)
        db.append({
            'codebooks': data['codebooks'],
            'stem':      cb_file.stem,
            'category':  cb_file.parent.name,
        })
    return db


def retrieve(query_path, top_k=10, db=None):
    """
    Given a query image path, return the top_k most similar database images.

    Args:
        query_path: path to the query image (any format readable by OpenCV)
        top_k:      number of results to return
        db:         pre-loaded codebook list from load_all_codebooks().
                    If None, codebooks are loaded from disk on every call.

    Returns:
        list of dicts, each with keys: 'rank', 'image', 'category', 'score'
    """
    # --- Preprocess the query and extract its signature ---
    img             = preprocess(query_path)
    query_signature = get_query_signature(img)

    # --- Load codebooks from disk if not pre-loaded ---
    if db is None:
        db = load_all_codebooks()

    results = []

    # --- Score the query against every DB image ---
    for entry in db:
        score = score_against_db_image(query_signature, entry['codebooks'])
        results.append({
            'image':    entry['stem'],
            'category': entry['category'],
            'score':    score,
        })

    # --- Sort by ascending score: lower MSE = more similar ---
    results.sort(key=lambda x: x['score'])

    # --- Attach rank numbers and return the top-k ---
    for i, r in enumerate(results, 1):
        r['rank'] = i

    return results[:top_k]


# ===========================================================================
# ENTRY POINT
# Usage: python retrieve.py <query_image_path> [top_k]
# ===========================================================================

if __name__ == "__main__":
    query_path = ROOT / "data" / "query.jpg"
    top_k      = 10

    if not query_path.exists():
        print(f"Query image not found: {query_path}")
        sys.exit(1)

    print(f"\nQuery: {query_path}")
    print(f"Retrieving top {top_k} results...\n")

    results = retrieve(query_path, top_k)

    # Print ranked results as a table
    print(f"{'Rank':<6} {'Image':<10} {'Category':<15} {'Score'}")
    print("-" * 45)
    for r in results:
        print(f"{r['rank']:<6} {r['image']:<10} {r['category']:<15} {r['score']:.6f}")
