import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from scripts.project_paths import BRANCH4_FEATURES_CSV, BRANCH4_MODEL, GLOBAL_SPLIT_JSON
from scripts.split_utils import get_or_create_global_path_split


def main(data_path=BRANCH4_FEATURES_CSV, out_model=BRANCH4_MODEL):
    print("Loading Branch4 dataset...", flush=True)
    df = pd.read_csv(data_path)
    print("Loaded shape:", df.shape, flush=True)

    X = df.drop(columns=["path", "label"])
    y = df["label"].values
    paths = df["path"].values

    train_idx, test_idx = get_or_create_global_path_split(
        paths=paths, labels=y, split_path=GLOBAL_SPLIT_JSON, test_size=0.2, random_state=42
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=2000)),
    ])

    print("Training Branch4 classifier...", flush=True)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("\nHoldout Accuracy:", accuracy_score(y_test, y_pred), flush=True)
    print("Holdout ROC-AUC:", roc_auc_score(y_test, y_prob), flush=True)
    print("\nConfusion Matrix [ [TN FP], [FN TP] ]:", flush=True)
    print(confusion_matrix(y_test, y_pred), flush=True)
    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, y_pred, digits=4), flush=True)

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_model)
    print("Saved model:", out_model, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(BRANCH4_FEATURES_CSV))
    ap.add_argument("--out", type=str, default=str(BRANCH4_MODEL))
    args = ap.parse_args()

    main(args.data, args.out)
