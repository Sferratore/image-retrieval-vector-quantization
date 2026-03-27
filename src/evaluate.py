import sys
import shutil
import numpy as np
from pathlib import Path

# Allow importing retrieve() from retrieve.py in the same folder
sys.path.insert(0, str(Path(__file__).parent))
from retrieve import retrieve, load_all_codebooks

# --- Paths ---
ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
QUERY_PATH    = ROOT / "data" / "query.jpg"

# Standard 11 recall levels used in image retrieval evaluation
RECALL_LEVELS = np.linspace(0, 1, 11)

# Number of relevant images per query: 100 images per category minus the query itself
N_RELEVANT = 99


# ===========================================================================
# PRECISION-RECALL COMPUTATION
# ===========================================================================

def compute_pr_curve(results, query_stem, query_category):
    """
    Given a ranked list of results (all DB images), compute precision and
    recall at every rank position, excluding the query image itself.

    A result is "relevant" if it belongs to the same category as the query.

    Returns:
        precisions: list of precision values at each rank
        recalls:    list of recall values at each rank
    """
    precisions = []
    recalls    = []
    n_correct  = 0
    rank       = 0

    for r in results:
        # Skip the query image itself — it always scores near 0 against its own codebook
        if r['image'] == query_stem:
            continue

        rank += 1

        if r['category'] == query_category:
            n_correct += 1

        precisions.append(n_correct / rank)
        recalls.append(n_correct / N_RELEVANT)

    return precisions, recalls


def interpolate_precision(recalls, precisions):
    """
    Standard 11-point interpolated precision-recall curve.

    At each of the 11 fixed recall levels (0.0, 0.1, ..., 1.0), the
    interpolated precision is the maximum precision achieved at any recall
    >= that level. This smooths the jagged raw curve.

    Returns:
        np.ndarray of shape (11,)
    """
    recalls    = np.array(recalls)
    precisions = np.array(precisions)

    interp = []
    for r in RECALL_LEVELS:
        # All precision values where recall is at least r
        mask = recalls >= r
        interp.append(precisions[mask].max() if mask.any() else 0.0)

    return np.array(interp)


# ===========================================================================
# MAIN EVALUATION LOOP
# ===========================================================================

# Collect all query images from data/processed/
img_paths = sorted(PROCESSED_DIR.rglob("*.jpg"))
n_queries = len(img_paths)

print(f"Found {n_queries} images. Starting evaluation...\n")

# Load all codebooks into RAM once — avoids re-reading from disk on every query
db = load_all_codebooks()

# Accumulate interpolated PR curves: one row per query
all_curves = []

# Also accumulate per-category curves11
category_curves = {}

for i, img_path in enumerate(img_paths, 1):
    query_stem     = img_path.stem
    query_category = img_path.parent.name

    # Step 1 — Copy current image to data/query.jpg so the user can inspect it
    shutil.copy(img_path, QUERY_PATH)

    # Step 2 — Retrieve all DB images ranked by similarity
    # top_k=1000 retrieves every image in the database
    results = retrieve(img_path, top_k=1000, db=db)

    # Step 3 — Compute precision-recall curve for this query
    precisions, recalls = compute_pr_curve(results, query_stem, query_category)
    interp = interpolate_precision(recalls, precisions)

    all_curves.append(interp)

    # Accumulate per-category
    if query_category not in category_curves:
        category_curves[query_category] = []
    category_curves[query_category].append(interp)

    print(f"  [{i:4d}/{n_queries}] {query_category}/{query_stem}  AP={interp.mean():.4f}", end="\r")

print("\n\nEvaluation complete.\n")


# ===========================================================================
# RESULTS
# ===========================================================================

all_curves = np.array(all_curves)           # (n_queries, 11)
mean_curve = all_curves.mean(axis=0)        # average across all queries

# Print overall precision-recall table
print("=== Overall Precision-Recall (averaged over all queries) ===")
print(f"{'Recall':<10} {'Precision':<10}")
for r, p in zip(RECALL_LEVELS, mean_curve):
    print(f"  {r:.1f}        {p:.4f}")

map_score = mean_curve.mean()
print(f"\nMean Average Precision (MAP): {map_score:.4f}\n")

# Print per-category MAP
print("=== Per-Category MAP ===")
for category in sorted(category_curves):
    cat_curves = np.array(category_curves[category])   # (100, 11)
    cat_map    = cat_curves.mean(axis=0).mean()
    print(f"  {category:<15} MAP={cat_map:.4f}")


# ===========================================================================
# PLOT
# ===========================================================================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: overall curve ---
    axes[0].plot(RECALL_LEVELS, mean_curve, 'b-o', linewidth=2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Overall Precision-Recall (MAP={map_score:.4f})")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True)

    # --- Right: per-category curves ---
    for category in sorted(category_curves):
        cat_mean = np.array(category_curves[category]).mean(axis=0)
        axes[1].plot(RECALL_LEVELS, cat_mean, label=category, linewidth=1.5)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Per-Category Precision-Recall")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    plt.tight_layout()

    out_path = ROOT / "precision_recall.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved to {out_path}")

except ImportError:
    print("matplotlib not installed — skipping plot.")
    print("Install with: pipenv install matplotlib")
