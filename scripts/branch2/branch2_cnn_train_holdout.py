import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from scripts.project_paths import BRANCH2_CNN_FEATURES_PARQUET, BRANCH2_CNN_SGD_MODEL

DATA_PATH = BRANCH2_CNN_FEATURES_PARQUET
OUT_MODEL = BRANCH2_CNN_SGD_MODEL

def main():
    print("Loading dataset...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    print("Loaded shape:", df.shape, flush=True)

    X = df.drop(columns=["path", "label"])
    y = df["label"].values

    print("Splitting dataset...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training classifier...", flush=True)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=50,
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