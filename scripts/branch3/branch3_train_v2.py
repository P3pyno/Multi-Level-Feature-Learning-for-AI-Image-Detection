import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

from scripts.project_paths import (
    BRANCH3_CLIP_V2_GLOBAL_NPY,
    BRANCH3_CLIP_V2_QUADS_NPY,
    BRANCH3_CLIP_V2_META_CSV,
    BRANCH3_SEMANTIC_V2_MODEL,
    BRANCH3_FEATURES_V2_PARQUET,
    GLOBAL_SPLIT_JSON,
)
from scripts.split_utils import get_or_create_global_path_split

GLOBAL_PATH = BRANCH3_CLIP_V2_GLOBAL_NPY
QUAD_PATH = BRANCH3_CLIP_V2_QUADS_NPY
META_PATH = BRANCH3_CLIP_V2_META_CSV

OUT_MODEL = BRANCH3_SEMANTIC_V2_MODEL
OUT_FEATURES = BRANCH3_FEATURES_V2_PARQUET


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


def cosine(a, b):
    return np.sum(a * b, axis=-1)


def fit_real_reference(real_emb, k=5):
    centroid = real_emb.mean(axis=0, keepdims=True)
    centroid = l2_normalize(centroid)

    cov = LedoitWolf()
    cov.fit(real_emb)

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(real_emb)

    return {"centroid": centroid.astype(np.float32), "cov": cov, "knn": knn, "k": k}


def semantic_features(global_emb, quad_emb, ref):
    # ---------- global distribution features ----------
    mah_sq = ref["cov"].mahalanobis(global_emb)
    mahal = np.sqrt(np.maximum(mah_sq, 0.0))

    dists, _ = ref["knn"].kneighbors(global_emb, return_distance=True)
    knn_mean = dists.mean(axis=1)
    knn_min = dists.min(axis=1)

    centroid = ref["centroid"]
    cos_centroid = cosine(global_emb, centroid)
    l2_centroid = np.linalg.norm(global_emb - centroid, axis=1)

    # ---------- quadrant pairwise consistency ----------
    q0, q1, q2, q3 = quad_emb[:, 0], quad_emb[:, 1], quad_emb[:, 2], quad_emb[:, 3]

    pairwise = np.stack([
        cosine(q0, q1),
        cosine(q0, q2),
        cosine(q0, q3),
        cosine(q1, q2),
        cosine(q1, q3),
        cosine(q2, q3),
    ], axis=1)

    pair_mean = pairwise.mean(axis=1)
    pair_std = pairwise.std(axis=1)
    pair_min = pairwise.min(axis=1)
    pair_max = pairwise.max(axis=1)

    # ---------- global vs local consistency ----------
    gq = np.stack([
        cosine(global_emb, q0),
        cosine(global_emb, q1),
        cosine(global_emb, q2),
        cosine(global_emb, q3),
    ], axis=1)

    gq_mean = gq.mean(axis=1)
    gq_std = gq.std(axis=1)
    gq_min = gq.min(axis=1)
    gq_max = gq.max(axis=1)

    feats = np.stack([
        mahal,
        knn_mean,
        knn_min,
        cos_centroid,
        l2_centroid,
        pair_mean,
        pair_std,
        pair_min,
        pair_max,
        gq_mean,
        gq_std,
        gq_min,
        gq_max,
    ], axis=1).astype(np.float32)

    return feats


def main():
    print("Loading Branch 3 v2 embeddings...", flush=True)
    global_emb = np.load(GLOBAL_PATH).astype(np.float32)
    quad_emb = np.load(QUAD_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH)

    if len(global_emb) != len(meta) or len(quad_emb) != len(meta):
        raise ValueError("Mismatch between embeddings and metadata length.")

    global_emb = l2_normalize(global_emb)
    quad_emb = l2_normalize(quad_emb)

    y = meta["label"].values
    paths = meta["path"].values
    train_idx, test_idx = get_or_create_global_path_split(
        paths=paths, labels=y, split_path=GLOBAL_SPLIT_JSON, test_size=0.2, random_state=42
    )

    g_train, g_test = global_emb[train_idx], global_emb[test_idx]
    q_train, q_test = quad_emb[train_idx], quad_emb[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Fitting real-image semantic reference...", flush=True)
    ref = fit_real_reference(g_train[y_train == 0], k=5)

    print("Computing semantic feature vectors...", flush=True)
    X_train = semantic_features(g_train, q_train, ref)
    X_test = semantic_features(g_test, q_test, ref)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=3000))
    ])

    print("Training Branch 3 v2 classifier...", flush=True)
    pipe.fit(X_train, y_train)

    print("Evaluating holdout set...", flush=True)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nHoldout Accuracy:", acc, flush=True)
    print("Holdout ROC-AUC:", auc, flush=True)
    print("\nConfusion Matrix [ [TN FP], [FN TP] ]:", flush=True)
    print(confusion_matrix(y_test, y_pred), flush=True)
    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    joblib.dump({"reference": ref, "classifier": pipe}, OUT_MODEL)
    print("\nSaved model:", OUT_MODEL, flush=True)

    print("\nSaving full Branch 3 v2 feature file...", flush=True)
    X_all = semantic_features(global_emb, quad_emb, ref)

    df = meta.copy()
    cols = [
        "b3_mahal",
        "b3_knn_mean",
        "b3_knn_min",
        "b3_cos_centroid",
        "b3_l2_centroid",
        "b3_pair_mean",
        "b3_pair_std",
        "b3_pair_min",
        "b3_pair_max",
        "b3_gq_mean",
        "b3_gq_std",
        "b3_gq_min",
        "b3_gq_max",
    ]
    for i, c in enumerate(cols):
        df[c] = X_all[:, i]

    df.to_parquet(OUT_FEATURES, index=False)
    print("Saved semantic features:", OUT_FEATURES, flush=True)


if __name__ == "__main__":
    main()
