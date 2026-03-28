import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

from scripts.project_paths import (
    BRANCH3_CLIP_EMBEDDINGS_NPY,
    BRANCH3_CLIP_META_CSV,
    BRANCH3_SEMANTIC_MODEL,
    BRANCH3_FEATURES_PARQUET,
)

EMB_PATH = BRANCH3_CLIP_EMBEDDINGS_NPY
META_PATH = BRANCH3_CLIP_META_CSV
OUT_MODEL = BRANCH3_SEMANTIC_MODEL
OUT_FEATURES = BRANCH3_FEATURES_PARQUET


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def fit_real_reference(real_emb, k=5):
    centroid = real_emb.mean(axis=0, keepdims=True)
    centroid = centroid / np.linalg.norm(centroid, axis=1, keepdims=True).clip(min=1e-12)

    lw = LedoitWolf()
    lw.fit(real_emb)

    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn_model.fit(real_emb)

    return {
        "centroid": centroid.astype(np.float32),
        "cov": lw,
        "knn": nn_model,
        "k": k,
    }


def semantic_distance_features(emb, ref):
    mah_sq = ref["cov"].mahalanobis(emb)
    mah = np.sqrt(np.maximum(mah_sq, 0.0))

    dists, _ = ref["knn"].kneighbors(emb, return_distance=True)
    knn_mean = dists.mean(axis=1)

    centroid = ref["centroid"]
    cos = (emb * centroid).sum(axis=1)

    feats = np.stack([mah, knn_mean, cos], axis=1).astype(np.float32)
    return feats


def main():
    print("Loading embeddings...", flush=True)
    emb = np.load(EMB_PATH)
    meta = pd.read_csv(META_PATH)

    if len(emb) != len(meta):
        raise ValueError("Mismatch between embeddings and metadata length.")

    emb = l2_normalize(emb.astype(np.float32))
    y = meta["label"].values

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )

    emb_train = emb[train_idx]
    emb_test = emb[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Fitting real-image reference on TRAIN real embeddings...", flush=True)
    real_train_emb = emb_train[y_train == 0]
    ref = fit_real_reference(real_train_emb, k=5)

    print("Computing semantic distance features...", flush=True)
    X_train = semantic_distance_features(emb_train, ref)
    X_test = semantic_distance_features(emb_test, ref)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=2000))
    ])

    print("Training Branch-3 classifier...", flush=True)
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

    bundle = {
        "reference": ref,
        "classifier": pipe,
    }
    joblib.dump(bundle, OUT_MODEL)
    print("\nSaved model:", OUT_MODEL, flush=True)

    print("\nSaving full semantic feature file for later fusion...", flush=True)
    ref_all = fit_real_reference(emb[y == 0], k=5)
    X_all = semantic_distance_features(emb, ref_all)

    full_df = meta.copy()
    full_df["b3_mahal"] = X_all[:, 0]
    full_df["b3_knn"] = X_all[:, 1]
    full_df["b3_cos"] = X_all[:, 2]
    full_df.to_parquet(OUT_FEATURES, index=False)

    print("Saved semantic features:", OUT_FEATURES, flush=True)


if __name__ == "__main__":
    main()