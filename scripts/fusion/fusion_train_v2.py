import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from scripts.project_paths import (
    BRANCH1_FEATURES_CLEAN_CSV,
    BRANCH2_FEATURES_CSV,
    BRANCH2_CNN_FEATURES_PARQUET,
    BRANCH3_FEATURES_V2_PARQUET,
    FUSION_V2_MODEL,
    GLOBAL_SPLIT_JSON,
)
from scripts.split_utils import get_or_create_global_path_split
from scripts.fusion.gating import fit_branch_gates, apply_branch_gates


BRANCH1_PATH = BRANCH1_FEATURES_CLEAN_CSV
BRANCH2A_PATH = BRANCH2_FEATURES_CSV
BRANCH2B_PATH = BRANCH2_CNN_FEATURES_PARQUET
BRANCH3_PATH = BRANCH3_FEATURES_V2_PARQUET
OUT_MODEL = FUSION_V2_MODEL
USE_BRANCH_GATING = True


def prefix_features(df, prefix):
    feat_cols = [c for c in df.columns if c not in ["path", "label"]]
    # If features already have the target prefix, keep names unchanged.
    if all(c.startswith(prefix) for c in feat_cols):
        return df
    return df.rename(columns={c: f"{prefix}{c}" for c in feat_cols})


def build_classifier(classifier_type):
    if classifier_type == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), activation="relu", max_iter=500, random_state=42)
    return SGDClassifier(loss="log_loss", max_iter=80, tol=1e-3, random_state=42, verbose=1)


def main(classifier_type="logreg"):
    print("Loading Branch 1...", flush=True)
    b1 = prefix_features(pd.read_csv(BRANCH1_PATH), "b1_")
    print("Branch 1 shape:", b1.shape, flush=True)

    print("Loading Branch 2A...", flush=True)
    b2a = prefix_features(pd.read_csv(BRANCH2A_PATH), "b2a_")
    print("Branch 2A shape:", b2a.shape, flush=True)

    print("Loading Branch 2B...", flush=True)
    b2b = prefix_features(pd.read_parquet(BRANCH2B_PATH), "b2b_")
    print("Branch 2B shape:", b2b.shape, flush=True)

    print("Loading Branch 3 semantic-v2 features...", flush=True)
    b3 = prefix_features(pd.read_parquet(BRANCH3_PATH), "b3_")
    print("Branch 3 shape:", b3.shape, flush=True)

    print("Merging all branches...", flush=True)
    df = b1.merge(b2a, on=["path", "label"], how="inner")
    df = df.merge(b2b, on=["path", "label"], how="inner")
    df = df.merge(b3, on=["path", "label"], how="inner")

    print("Merged shape:", df.shape, flush=True)

    X = df.drop(columns=["path", "label"])
    y = df["label"].values
    paths = df["path"].values

    nan_total = int(X.isna().sum().sum())
    print("Total NaNs:", nan_total, flush=True)

    print("Splitting dataset...", flush=True)
    train_idx, test_idx = get_or_create_global_path_split(
        paths=paths, labels=y, split_path=GLOBAL_SPLIT_JSON, test_size=0.2, random_state=42
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}", flush=True)

    gate_params = None
    if USE_BRANCH_GATING:
        print("Applying optional branch-level gating...", flush=True)
        gate_params = fit_branch_gates(X_train)
        X_train = apply_branch_gates(X_train, gate_params)
        X_test = apply_branch_gates(X_test, gate_params)

    print("Training final fusion classifier...", flush=True)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", build_classifier(classifier_type))
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

    joblib.dump({"model": pipe, "use_branch_gating": USE_BRANCH_GATING, "gate_params": gate_params}, OUT_MODEL)
    print("\nSaved model bundle:", OUT_MODEL, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["logreg", "mlp"], default="logreg")
    args = ap.parse_args()
    main(classifier_type=args.classifier)
