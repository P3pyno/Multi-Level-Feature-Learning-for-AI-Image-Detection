import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from scripts.project_paths import (
    BRANCH3_CLIP_V2_GLOBAL_NPY,
    BRANCH3_CLIP_V2_META_CSV,
    BRANCH3_RAW_CLIP_MODEL,
    BRANCH3_RAW_FEATURES_PARQUET,
)

EMB_PATH = BRANCH3_CLIP_V2_GLOBAL_NPY
META_PATH = BRANCH3_CLIP_V2_META_CSV
OUT_MODEL = BRANCH3_RAW_CLIP_MODEL
OUT_FEATURES = BRANCH3_RAW_FEATURES_PARQUET


def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def main():
    print("Loading raw CLIP embeddings...", flush=True)
    emb = np.load(EMB_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH)

    if len(emb) != len(meta):
        raise ValueError("Mismatch between embeddings and metadata.")

    emb = l2_normalize(emb)
    y = meta["label"].values

    print("Embedding shape:", emb.shape, flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        emb, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train size:", len(X_train), "| Test size:", len(X_test), flush=True)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=200,
            tol=1e-4,
            random_state=42,
            verbose=1
        ))
    ])

    print("Training raw CLIP Branch 3...", flush=True)
    pipe.fit(X_train, y_train)

    print("Evaluating holdout...", flush=True)
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

    joblib.dump(pipe, OUT_MODEL)
    print("\nSaved model:", OUT_MODEL, flush=True)

    print("\nSaving raw CLIP feature parquet for later fusion...", flush=True)
    df = meta.copy()
    for i in range(emb.shape[1]):
        df[f"b3_clip_{i:03d}"] = emb[:, i]
    df.to_parquet(OUT_FEATURES, index=False)
    print("Saved features:", OUT_FEATURES, flush=True)


if __name__ == "__main__":
    main()