import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scripts.project_paths import PROJECT_ROOT

# =========================
# Output folder
# =========================
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Your results
# =========================
results = {
    "Branch1": {
        "accuracy": 0.7560408617641281,
        "roc_auc": 0.8295646508568572,
        "precision_real": 0.7505,
        "recall_real": 0.7179,
        "f1_real": 0.7338,
        "precision_fake": 0.7606,
        "recall_fake": 0.7897,
        "f1_fake": 0.7748,
        "macro_f1": 0.7543,
        "weighted_f1": 0.7556,
        "cm": np.array([[10271, 4036], [3415, 12820]]),
    },
    "Branch2-Hand": {
        "accuracy": 0.7245105101172157,
        "roc_auc": 0.7859383596051984,
        "precision_real": 0.7024,
        "recall_real": 0.7148,
        "f1_real": 0.7085,
        "precision_fake": 0.7447,
        "recall_fake": 0.7330,
        "f1_fake": 0.7388,
        "macro_f1": 0.7237,
        "weighted_f1": 0.7246,
        "cm": np.array([[10227, 4080], [4334, 11901]]),
    },
    "Branch2-CNN": {
        "accuracy": 0.8515159452557134,
        "roc_auc": 0.9545907100422218,
        "precision_real": 0.9705,
        "recall_real": 0.7044,
        "f1_real": 0.8163,
        "precision_fake": 0.7902,
        "recall_fake": 0.9812,
        "f1_fake": 0.8754,
        "macro_f1": 0.8459,
        "weighted_f1": 0.8477,
        "cm": np.array([[10078, 4229], [306, 15929]]),
    },
    "Branch3-Sem": {
        "accuracy": 0.8750245563486346,
        "roc_auc": 0.9411499999999999,
        "precision_real": 0.8047,
        "recall_real": 0.9682,
        "f1_real": 0.8789,
        "precision_fake": 0.9659,
        "recall_fake": 0.7929,
        "f1_fake": 0.8709,
        "macro_f1": 0.8749,
        "weighted_f1": 0.8746,
        "cm": np.array([[13852, 455], [3362, 12873]]),
    },
    "Branch3-Raw": {
        "accuracy": 0.9029533101957959,
        "roc_auc": 0.9626886776399499,
        "precision_real": 0.8998,
        "recall_real": 0.8922,
        "f1_real": 0.8960,
        "precision_fake": 0.9057,
        "recall_fake": 0.9125,
        "f1_fake": 0.9091,
        "macro_f1": 0.9025,
        "weighted_f1": 0.9029,
        "cm": np.array([[12764, 1543], [1421, 14814]]),
    },
    "Fusion-v1": {
        "accuracy": 0.8750245563486346,
        "roc_auc": 0.9659774487599557,
        "precision_real": 0.8047,
        "recall_real": 0.9682,
        "f1_real": 0.8789,
        "precision_fake": 0.9659,
        "recall_fake": 0.7929,
        "f1_fake": 0.8709,
        "macro_f1": 0.8749,
        "weighted_f1": 0.8746,
        "cm": np.array([[13852, 455], [3362, 12873]]),
    },
    "Fusion-v2": {
        "accuracy": 0.9386091284133324,
        "roc_auc": 0.9803789677064574,
        "precision_real": 0.9349,
        "recall_real": 0.9339,
        "f1_real": 0.9344,
        "precision_fake": 0.9418,
        "recall_fake": 0.9427,
        "f1_fake": 0.9423,
        "macro_f1": 0.9384,
        "weighted_f1": 0.9386,
        "cm": np.array([[13362, 945], [930, 15305]]),
    },
}

ORDER = [
    "Branch1",
    "Branch2-Hand",
    "Branch2-CNN",
    "Branch3-Sem",
    "Branch3-Raw",
    "Fusion-v1",
    "Fusion-v2",
]

# =========================
# Helpers
# =========================
def save_plot(name: str) -> None:
    out = PLOTS_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def annotate_bars(bars):
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# =========================
# 1. Accuracy / ROC-AUC comparison
# =========================
labels = ORDER
acc = [results[k]["accuracy"] for k in labels]
auc = [results[k]["roc_auc"] for k in labels]

x = np.arange(len(labels))
w = 0.38

plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - w / 2, acc, w, label="Accuracy")
bars2 = plt.bar(x + w / 2, auc, w, label="ROC-AUC")
plt.xticks(x, labels, rotation=25)
plt.ylim(0.65, 1.0)
plt.ylabel("Score")
plt.title("Branch and Fusion Performance Comparison")
plt.legend()
annotate_bars(bars1)
annotate_bars(bars2)
save_plot("01_accuracy_auc_comparison.png")

# =========================
# 2. F1 / Precision / Recall summary
# =========================
macro_f1 = [results[k]["macro_f1"] for k in labels]
weighted_f1 = [results[k]["weighted_f1"] for k in labels]
recall_fake = [results[k]["recall_fake"] for k in labels]

plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - w, macro_f1, w, label="Macro F1")
bars2 = plt.bar(x, weighted_f1, w, label="Weighted F1")
bars3 = plt.bar(x + w, recall_fake, w, label="Recall (Fake class)")
plt.xticks(x, labels, rotation=25)
plt.ylim(0.65, 1.0)
plt.ylabel("Score")
plt.title("F1 Scores and Fake-Class Recall")
plt.legend()
annotate_bars(bars1)
annotate_bars(bars2)
annotate_bars(bars3)
save_plot("02_f1_recall_summary.png")

# =========================
# 3. Real vs Fake class comparison
# =========================
real_f1 = [results[k]["f1_real"] for k in labels]
fake_f1 = [results[k]["f1_fake"] for k in labels]

plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - w / 2, real_f1, w, label="F1 (Real)")
bars2 = plt.bar(x + w / 2, fake_f1, w, label="F1 (Fake)")
plt.xticks(x, labels, rotation=25)
plt.ylim(0.65, 1.0)
plt.ylabel("F1-score")
plt.title("Per-Class F1 Comparison")
plt.legend()
annotate_bars(bars1)
annotate_bars(bars2)
save_plot("03_per_class_f1.png")

# =========================
# 4. Confusion matrices
# =========================
def plot_confusion_matrix(cm: np.ndarray, title: str, filename: str) -> None:
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred Real", "Pred Fake"])
    plt.yticks(tick_marks, ["True Real", "True Fake"])

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=11,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    save_plot(filename)


for name in labels:
    plot_confusion_matrix(
        results[name]["cm"],
        f"{name} Confusion Matrix",
        f"cm_{name.lower().replace('-', '_')}.png",
    )

# =========================
# 5. Thesis-style ablation plot
# =========================
ablation_labels = [
    "B1",
    "B2-Hand",
    "B2-CNN",
    "B3-Sem",
    "B3-Raw",
    "Fusion-v1",
    "Fusion-v2",
]
ablation_scores = [
    results["Branch1"]["accuracy"],
    results["Branch2-Hand"]["accuracy"],
    results["Branch2-CNN"]["accuracy"],
    results["Branch3-Sem"]["accuracy"],
    results["Branch3-Raw"]["accuracy"],
    results["Fusion-v1"]["accuracy"],
    results["Fusion-v2"]["accuracy"],
]

plt.figure(figsize=(11, 5.5))
bars = plt.bar(ablation_labels, ablation_scores)
plt.ylim(0.65, 1.0)
plt.ylabel("Accuracy")
plt.title("Ablation Study: Contribution of Each Branch and Fusion Strategy")
annotate_bars(bars)
save_plot("04_ablation_accuracy.png")

# =========================
# 6. ROC-AUC ranking plot
# =========================
sorted_items = sorted(results.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)
rank_labels = [k for k, _ in sorted_items]
rank_auc = [v["roc_auc"] for _, v in sorted_items]

plt.figure(figsize=(11, 5.5))
bars = plt.bar(rank_labels, rank_auc)
plt.xticks(rotation=25)
plt.ylim(0.75, 1.0)
plt.ylabel("ROC-AUC")
plt.title("Model Ranking by ROC-AUC")
annotate_bars(bars)
save_plot("05_auc_ranking.png")

print(f"\nAll plots saved in: {PLOTS_DIR}")