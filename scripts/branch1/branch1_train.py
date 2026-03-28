import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib

from scripts.project_paths import BRANCH1_FEATURES_CLEAN_CSV, BRANCH1_MODEL
from scripts.project_paths import GLOBAL_SPLIT_JSON
from scripts.split_utils import get_or_create_global_path_split


def load_data(path):
    path = str(path)
    print("Loading dataset:", path)

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    print("Loaded shape:", df.shape)

    if "label" not in df.columns:
        raise ValueError("Column 'label' not found!")

    if "path" not in df.columns:
        raise ValueError("Column 'path' not found. Global split requires image paths.")

    X = df.drop(columns=["path", "label"])

    y = df["label"].values
    paths = df["path"].values
    return X, y, paths


def train_and_eval(csv_path, out_model=BRANCH1_MODEL):
    X, y, paths = load_data(csv_path)

    print("\nSplitting dataset...")
    train_idx, test_idx = get_or_create_global_path_split(
        paths=paths,
        labels=y,
        split_path=GLOBAL_SPLIT_JSON,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Train size:", len(X_train), "| Test size:", len(X_test))

    print("\nBuilding pipeline...")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=8000,
                    verbose=1,
                ),
            ),
        ]
    )

    print("\nRunning 5-fold cross validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "accuracy"],
        n_jobs=-1,
    )

    print("\n5-fold CV ROC-AUC:")
    print("mean =", scores["test_roc_auc"].mean(), "std =", scores["test_roc_auc"].std())

    print("\n5-fold CV ACC:")
    print("mean =", scores["test_accuracy"].mean(), "std =", scores["test_accuracy"].std())

    print("\nTraining final model on full training set...")

    pipe.fit(X_train, y_train)

    print("\nEvaluating holdout set...")

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nHoldout Accuracy:", acc)
    print("Holdout ROC-AUC:", auc)

    print("\nConfusion Matrix [ [TN FP], [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nSaving model:", out_model)
    joblib.dump(pipe, out_model)

    print("Done.")


if __name__ == "__main__":
    csv_path = (
        sys.argv[1] if len(sys.argv) > 1 else BRANCH1_FEATURES_CLEAN_CSV
    )

    out_model = (
        sys.argv[2] if len(sys.argv) > 2 else BRANCH1_MODEL
    )

    train_and_eval(csv_path, out_model)
