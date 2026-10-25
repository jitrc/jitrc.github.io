"""Reproduces the numbers in the essay "Inventing Failure Modes".

    python -m venv .venv
    .venv/bin/pip install sentence-transformers umap-learn hdbscan scikit-learn
    .venv/bin/python scripts/cluster-experiment/run.py

Corpus: 48 failures across four designed modes (12 paraphrases each) plus 10 designed
one-offs. Measures how k-means and UMAP+HDBSCAN treat the one-offs, and whether
distance-from-known-modes detects novelty better than any clustering signal.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
import umap
import hdbscan

from corpus import texts, truth, MODES

EMBED_MODEL = "all-MiniLM-L6-v2"
emb = SentenceTransformer(EMBED_MODEL).encode(
    texts, normalize_embeddings=True, show_progress_bar=False)
emb_5d = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=5,
                   random_state=42).fit_transform(emb)
is_oneoff = truth == "oneoff"

print(f"corpus: {len(texts)} failures = {len(texts) - is_oneoff.sum()} mode failures "
      f"+ {is_oneoff.sum()} one-offs\n")

# --- k-means, told the true number of modes --------------------------------
print("== k-means, k=4 (the number of modes that exist) ==")
km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(emb)
print(f"points assigned to a cluster: {len(km.labels_)}/{len(texts)} — no point can be noise")
print(f"one-offs forced into a 'mode': {is_oneoff.sum()}/{is_oneoff.sum()}")
print("\ndesigned mode -> cluster sizes")
for m in list(MODES) + ["oneoff"]:
    print(f"  {m:11s} {np.bincount(km.labels_[truth == m], minlength=4).tolist()}")
print(f"ARI vs ground truth: {adjusted_rand_score(truth, km.labels_):.3f}")
for k in (4, 5, 6):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
    print(f"  k={k}: silhouette {silhouette_score(emb, lab):.3f}, "
          f"ARI {adjusted_rand_score(truth, lab):.3f}")

# --- HDBSCAN noise label ---------------------------------------------------
print("\n== UMAP(5d) + HDBSCAN(min_cluster_size=5): does -1 find the one-offs? ==")
print(f"{'min_samples':>12} {'clusters':>9} {'noise':>6} {'one-offs in noise':>18} {'mode failures in noise':>23}")
for ms in (1, 2, 3, 5, 8, 12, 20):
    lab = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=ms).fit_predict(emb_5d)
    n = lab == -1
    n_clusters = len(set(lab)) - (1 if n.any() else 0)
    print(f"{ms:>12} {n_clusters:>9} {int(n.sum()):>6} "
          f"{int((n & is_oneoff).sum())}/10{'':<14} {int((n & ~is_oneoff).sum()):>23}")

cl = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3).fit(emb_5d)
print(f"\nGLOSH outlier score, ROC AUC for 'is a one-off': "
      f"{roc_auc_score(is_oneoff, cl.outlier_scores_):.3f}")
print(f"membership uncertainty, ROC AUC: "
      f"{roc_auc_score(is_oneoff, 1 - cl.probabilities_):.3f}")

# --- novelty as distance from known modes ----------------------------------
# A loop that has run before already owns labelled examples. Six per mode is enough.
rng = np.random.default_rng(0)
history = np.concatenate([rng.choice(np.where(truth == m)[0], 6, replace=False) for m in MODES])
rest = np.setdiff1d(np.arange(len(texts)), history)
centroids = {m: emb[history[truth[history] == m]].mean(axis=0) for m in MODES}
centroids = {m: c / np.linalg.norm(c) for m, c in centroids.items()}
novelty = np.array([1 - max(float(emb[i] @ c) for c in centroids.values()) for i in rest])
print(f"\n== distance from known modes ({len(rest)} held-out points) ==")
print(f"ROC AUC: {roc_auc_score(is_oneoff[rest], novelty):.3f}")
print(f"top 10 by novelty: {sum(is_oneoff[rest][i] for i in np.argsort(novelty)[::-1][:10])}/10 are one-offs")
