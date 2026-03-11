import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans

# Input: preprocessed images as .jpg (256x256, BGR)
# Output: per-image .npz files each containing GRID*GRID TSVQ trees
INPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "codebooks"

BLOCK_SIZE = 4   # size in pixels of each block (BLOCK_SIZE x BLOCK_SIZE)
GRID       = 4   # image is divided into a GRID x GRID spatial grid of regions
DEPTH      = 3   # TSVQ tree depth: 2^DEPTH leaf codewords per region

# Fixed normalization scale: divides each feature by its theoretical maximum
# so all 6 features end up in roughly the same [0, 1] range.
# mean channels:     max is 255 (uint8 pixel value)
# variance channels: max is 255²/4 = 16256 (worst-case for any block)
NORM_SCALE = np.array([255, 255, 255, 16256, 16256, 16256], dtype=np.float64)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_block_features(img):
    """
    Divide the image into non-overlapping blocks of size BLOCK_SIZE x BLOCK_SIZE
    and extract a 9-dim feature vector from each block.

    Feature vector: [mean_L, mean_u, mean_v, var_L, var_u, var_v]

    Returns:
        feats: np.ndarray of shape (n_blocks, 6)
    """
    # img is a 3D array: (height, width, 3 channels).
    # We read its dimensions here. The _ discards the third value (Determines L, u o v)
    # because we don't need to store the channel count explicitly.
    h, w, _ = img.shape

    # Empty list that will collect one 6-number vector for each block.
    # At the end this list will have as many entries as there are blocks in the image.
    feats = []

    # Outer loop: move down the image one block at a time.
    # y is the row index of the top-left corner of the current block.
    # We jump BLOCK_SIZE pixels at each step so blocks don't overlap.
    for y in range(0, h, BLOCK_SIZE):

        # Inner loop: move across the image one block at a time.
        # x is the column index of the top-left corner of the current block.
        # Same non-overlapping step as above.
        for x in range(0, w, BLOCK_SIZE):

            # Cut out the current block from the image using array slicing.
            # Rows from y to y+BLOCK_SIZE, columns from x to x+BLOCK_SIZE.
            # The result is a small 3D array of shape (BLOCK_SIZE, BLOCK_SIZE, 3).
            block = img[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            # Compute the average pixel value for each color channel (L, u, v)
            # across all pixels inside this block.
            # axis=(0,1) means "collapse rows and columns" so we get one average
            # per channel. Result is an array of 3 numbers: [mean_L, mean_u, mean_v].
            mean = block.mean(axis=(0,1))

            # Compute the variance of pixel values for each channel inside this block.
            # Variance measures how much the pixel values differ from the mean.
            # A low variance means the block is uniform in color;
            # a high variance means there is a lot of color variation inside it.
            # Same axis logic as above. Result: [var_L, var_u, var_v].
            var = block.var(axis=(0,1))

            # Concatenate mean and variance into a single 6-dim vector:
            # [mean_L, mean_u, mean_v, var_L, var_u, var_v].
            feat = np.concatenate([mean, var])

            # Append this block's feature vector to the list.
            feats.append(feat)

    # Convert the Python list of 6-number vectors into a 2D NumPy array.
    # Final shape: (number_of_blocks, 6). Each row is one block, each column is one feature.
    feats = np.array(feats)

    # Divide each feature by its known maximum so all dimensions are in ~[0, 1].
    # This prevents high-scale features (variance) from dominating the distance.
    return feats / NORM_SCALE


def assign_regions(n_blocks_y, n_blocks_x):
    """
    Assign each block to one of GRID*GRID spatial regions.
    Regions are numbered row-major from 0 to GRID*GRID-1.

    Returns:
        region_map: np.ndarray of shape (n_blocks_y, n_blocks_x) with region IDs
    """
    # Create a 2D grid of zeros with the same layout as the block grid.
    # Each cell corresponds to one block and will be filled with a region ID (0 to GRID*GRID-1).
    # For a 128x128 image with BLOCK_SIZE=2 and GRID=3: shape is (64, 64).
    region_map = np.zeros((n_blocks_y, n_blocks_x), dtype=int)

    # How many block-rows fit in one region vertically.
    # e.g. 64 block-rows / 3 = 21 block-rows per region (integer division).
    ry = n_blocks_y // GRID

    # How many block-columns fit in one region horizontally.
    # e.g. 64 block-cols / 3 = 21 block-cols per region (integer division).
    rx = n_blocks_x // GRID

    # Counter that gives each region a unique ID, starting from 0.
    r = 0

    # Outer loop: iterate over the rows of the GRID (top to bottom).
    # gy goes 0, 1, 2 for a 3x3 grid.
    for gy in range(GRID):

        # Inner loop: iterate over the columns of the GRID (left to right).
        # gx goes 0, 1, 2 for a 3x3 grid.
        for gx in range(GRID):

            # Compute the top block-row index of this region in the block grid.
            y0 = gy * ry

            # Compute the bottom block-row index (exclusive) of this region.
            y1 = (gy + 1) * ry

            # Compute the leftmost block-column index of this region.
            x0 = gx * rx

            # Compute the rightmost block-column index (exclusive) of this region.
            x1 = (gx + 1) * rx

            # Fill all cells in this rectangular area of the block grid with
            # the current region ID r. Every block inside this rectangle now
            # knows it belongs to region r.
            region_map[y0:y1, x0:x1] = r

            # Move to the next region ID for the next iteration.
            r += 1

    # Return the completed map. Each cell holds the region ID of that block.
    # Example for GRID=3, the map looks conceptually like:
    # [ 0  0 ... 1  1 ... 2  2 ... ]
    # [ 0  0 ... 1  1 ... 2  2 ... ]
    # [ 3  3 ... 4  4 ... 5  5 ... ]
    # [ 3  3 ... 4  4 ... 5  5 ... ]
    # [ 6  6 ... 7  7 ... 8  8 ... ]
    # [ 6  6 ... 7  7 ... 8  8 ... ]
    return region_map


def build_tsvq_region(vectors, tree, node_idx, current_depth):
    """
    Recursively build a TSVQ binary tree for one region, modifying tree in place.

    At each node, split the vectors into 2 groups using k-means (k=2, n_init=5).
    The 2 resulting centroids are stored as the left and right children of the
    current node in the flat tree array. Then recurse on each group.

    The flat tree array uses heap indexing: node at index i has children at
    indices 2i+1 (left) and 2i+2 (right). Root is at index 0.

    Args:
        vectors:       feature vectors for this node, shape (n, 6)
        tree:          flat tree array of shape (2^(DEPTH+1)-1, 6), modified in place
        node_idx:      index of the current node in the flat tree array
        current_depth: depth of the current node (root = 0, leaves = DEPTH)
    """
    # Base case: reached the desired leaf depth, stop recursing
    if current_depth == DEPTH:
        return

    left_idx  = 2 * node_idx + 1
    right_idx = 2 * node_idx + 2

    # If too few distinct points to split, copy the current centroid to both
    # children so the tree remains valid without a meaningful split
    if len(np.unique(vectors, axis=0)) < 2:
        tree[left_idx]  = tree[node_idx]
        tree[right_idx] = tree[node_idx]
        build_tsvq_region(vectors, tree, left_idx,  current_depth + 1)
        build_tsvq_region(vectors, tree, right_idx, current_depth + 1)
        return

    # Split into 2 groups; run 5 times and keep the lowest-distortion result
    kmeans = KMeans(n_clusters=2, n_init=5)
    kmeans.fit(vectors)

    # Store the 2 child centroids at their positions in the flat tree array
    tree[left_idx]  = kmeans.cluster_centers_[0]
    tree[right_idx] = kmeans.cluster_centers_[1]

    # Recurse on each group
    build_tsvq_region(vectors[kmeans.labels_ == 0], tree, left_idx,  current_depth + 1)
    build_tsvq_region(vectors[kmeans.labels_ == 1], tree, right_idx, current_depth + 1)


def build_codebooks(img):
    """
    Build one TSVQ tree per spatial region for a single image.

    For each of the GRID*GRID regions:
      - collect block feature vectors (already normalized to ~[0,1])
      - recursively build a binary tree of depth DEPTH using k-means (k=2)

    The tree is stored as a flat array using heap indexing:
      - node at index i has children at 2i+1 and 2i+2
      - root is at index 0 (its centroid = overall mean of the region)
      - leaves are at indices 2^DEPTH - 1 to 2^(DEPTH+1) - 2

    Returns:
        trees: np.ndarray of shape (GRID*GRID, n_nodes, 6)
               where n_nodes = 2^(DEPTH+1) - 1
    """
    h, w, _ = img.shape

    n_blocks_y = h // BLOCK_SIZE
    n_blocks_x = w // BLOCK_SIZE

    features   = extract_block_features(img)                        # (n_blocks, 6)
    region_ids = assign_regions(n_blocks_y, n_blocks_x).flatten()  # (n_blocks,)

    # Total nodes in a complete binary tree of depth DEPTH
    n_nodes = 2 ** (DEPTH + 1) - 1

    trees = np.zeros((GRID * GRID, n_nodes, 6))

    for r in range(GRID * GRID):
        region_feats = features[region_ids == r]

        # Root centroid = overall mean of the region's vectors
        trees[r, 0] = region_feats.mean(axis=0)

        # Recursively build the tree starting from the root
        build_tsvq_region(region_feats, trees[r], 0, 0)

    return trees


# --- Main loop: process every preprocessed image ---
for file in INPUT_DIR.rglob("*.jpg"):

    img = cv2.imread(str(file))                        # load (128, 128, 3) BGR image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)         # convert to Luv inline

    codebooks = build_codebooks(img)                   # shape (GRID*GRID, n_nodes, 6) TSVQ trees

    # Preserve the category subfolder (e.g. data/codebooks/Corel-1K/dinosaurs/400.npz)
    # so that category labels can be read directly from the path in retrieve.py
    relative = file.relative_to(INPUT_DIR)             # e.g. Corel-1K/dinosaurs/400.jpg
    save_dir = OUTPUT_DIR / relative.parent            # e.g. data/codebooks/Corel-1K/dinosaurs
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / (file.stem + ".npz")

    np.savez(save_path, codebooks=codebooks)           # save all codebooks for this image

print("Finished building codebooks.")
