import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from scripts.project_paths import (
    BRANCH1_FEATURES_CLEAN,
    BRANCH2_FEATURES_CSV,
    BRANCH2_CNN_FEATURES_PARQUET,
    BRANCH3_RAW_FEATURES_PARQUET,
    FUSION_V2_MODEL,
)


BRANCH1_PATH = BRANCH1_FEATURES_CLEAN
BRANCH2A_PATH = BRANCH2_FEATURES_CSV
BRANCH2B_PATH = BRANCH2_CNN_FEATURES_PARQUET
BRANCH3_PATH = BRANCH3_RAW_FEATURES_PARQUET

OUT_MODEL = FUSION_V2_MODEL


def prefix_features(df, prefix):
    feat_cols = [c for c in df.columns if c not in ["path", "label"]]
    return df.rename(columns={c: f"{prefix}{c}" for c in feat_cols})


def main():
    print("Loading Branch 1...", flush=True)
    b1 = pd.read_csv(BRANCH1_PATH)
    b1 = prefix_features(b1, "b1_")
    print("Branch 1 shape:", b1.shape, flush=True)

    print("Loading Branch 2A...", flush=True)
    b2a = pd.read_csv(BRANCH2A_PATH)
    b2a = prefix_features(b2a, "b2a_")
    print("Branch 2A shape:", b2a.shape, flush=True)

    print("Loading Branch 2B...", flush=True)
    b2b = pd.read_parquet(BRANCH2B_PATH)
    b2b = prefix_features(b2b, "b2b_")
    print("Branch 2B shape:", b2b.shape, flush=True)

    print("Loading Branch 3...", flush=True)
    b3 = pd.read_parquet(BRANCH3_PATH)
    b3 = prefix_features(b3, "b3_")
    print("Branch 3 shape:", b3.shape, flush=True)

    print("Merging all branches...", flush=True)
    df = b1.merge(b2a, on=["path", "label"], how="inner")
    df = df.merge(b2b, on=["path", "label"], how="inner")
    df = df.merge(b3, on=["path", "label"], how="inner")

    print("Merged shape:", df.shape, flush=True)

    X = df.drop(columns=["path", "label"])
    y = df["label"].values

    nan_total = int(X.isna().sum().sum())
    print("Total NaNs:", nan_total, flush=True)

    print("Splitting dataset...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}", flush=True)

    print("Training final fusion classifier...", flush=True)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=80,
            tol=1e-3,
            random_state=42,
            verbose=1
        ))
    ])

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

    joblib.dump(pipe, OUT_MODEL)
    print("\nSaved model:", OUT_MODEL, flush=True)


if __name__ == "__main__":
    main()