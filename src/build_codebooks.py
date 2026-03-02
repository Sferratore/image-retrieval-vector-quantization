import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans

# Input: preprocessed images as .jpg (128x128, CIE Luv*)
# Output: per-image .npz files each containing GRID*GRID codebooks
INPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "codebooks"

BLOCK_SIZE = 2   # size in pixels of each block (BLOCK_SIZE x BLOCK_SIZE)
GRID = 3         # image is divided into a GRID x GRID spatial grid of regions
K = 8            # number of codewords per codebook (VQ codebook size)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_block_features(img):
    """
    Divide the image into non-overlapping blocks of size BLOCK_SIZE x BLOCK_SIZE
    and extract a 6-dim feature vector from each block.

    Feature vector: [mean_L, mean_u, mean_v, var_L, var_u, var_v]

    Returns:
        feats: np.ndarray of shape (n_blocks, 6)
    """
    # unpack spatial dimensions; _ discards the channel count (always 3)
    h, w, _ = img.shape

    # will accumulate one feature vector per block
    feats = []

    # slide over rows in steps of BLOCK_SIZE (non-overlapping)
    for y in range(0, h, BLOCK_SIZE):
        # slide over columns in steps of BLOCK_SIZE (non-overlapping)
        for x in range(0, w, BLOCK_SIZE):
            # crop the block: rows [y, y+BLOCK_SIZE), cols [x, x+BLOCK_SIZE)
            # shape: (BLOCK_SIZE, BLOCK_SIZE, 3)
            block = img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            # average pixel value for each channel (L, u, v) across the block
            # axis=(0,1) collapses the two spatial axes → shape (3,)
            mean = block.mean(axis=(0,1))

            # spread of pixel values for each channel across the block
            # axis=(0,1) collapses the two spatial axes → shape (3,)
            var = block.var(axis=(0,1))

            # join mean and variance into a single 6-dim vector [mean_L, mean_u, mean_v, var_L, var_u, var_v]
            feat = np.concatenate([mean, var])

            feats.append(feat)

    # stack all per-block vectors into a 2D array → shape (n_blocks, 6)
    return np.array(feats)


def assign_regions(n_blocks_y, n_blocks_x):
    """
    Assign each block to one of GRID*GRID spatial regions.
    Regions are numbered row-major from 0 to GRID*GRID-1.

    Returns:
        region_map: np.ndarray of shape (n_blocks_y, n_blocks_x) with region IDs
    """
    region_map = np.zeros((n_blocks_y, n_blocks_x), dtype=int)

    ry = n_blocks_y // GRID  # block rows per region
    rx = n_blocks_x // GRID  # block cols per region

    r = 0
    for gy in range(GRID):
        for gx in range(GRID):
            y0 = gy * ry
            y1 = (gy + 1) * ry
            x0 = gx * rx
            x1 = (gx + 1) * rx
            region_map[y0:y1, x0:x1] = r
            r += 1

    return region_map


def build_codebooks(img):
    """
    Build one VQ codebook per spatial region for a single image.

    For each of the GRID*GRID regions:
      - collect block feature vectors belonging to that region
      - run k-means with K clusters to obtain the codebook (cluster centers)

    Returns:
        codebooks: list of GRID*GRID arrays, each of shape (k, 6)
    """
    h, w, _ = img.shape

    n_blocks_y = h // BLOCK_SIZE
    n_blocks_x = w // BLOCK_SIZE

    features = extract_block_features(img)             # (n_blocks, 6)
    region_ids = assign_regions(n_blocks_y, n_blocks_x).flatten()  # (n_blocks,)

    codebooks = []

    for r in range(GRID * GRID):
        region_feats = features[region_ids == r]       # blocks belonging to region r

        k = min(K, len(region_feats))                  # guard: k <= number of samples

        kmeans = KMeans(n_clusters=k, n_init=5)
        kmeans.fit(region_feats)

        codebooks.append(kmeans.cluster_centers_)      # (k, 6)

    return codebooks


# --- Main loop: process every preprocessed image ---
for file in INPUT_DIR.rglob("*.jpg"):

    img = cv2.imread(str(file))                        # load (128, 128, 3) Luv image

    codebooks = build_codebooks(img)                   # list of GRID*GRID codebooks

    save_path = OUTPUT_DIR / (file.stem + ".npz")

    np.savez(save_path, codebooks=codebooks)           # save all codebooks for this image

print("Finished building codebooks.")
