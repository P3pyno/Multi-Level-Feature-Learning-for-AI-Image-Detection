import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from tqdm import tqdm

OUT_ROOT = "parvesh_ai_vs_real"

ai_dir = os.path.join(OUT_ROOT, "ai")
real_dir = os.path.join(OUT_ROOT, "real")

os.makedirs(ai_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

print("Downloading dataset from HuggingFace...")
dataset = load_dataset("Parveshiiii/AI-vs-Real", split="train")

print(dataset)
print("Saving images locally...")

saved_ai = 0
saved_real = 0

for i, row in enumerate(tqdm(dataset)):
    img = row["image"]
    label = int(row["binary_label"])  # dataset: 0=AI, 1=Real

    if label == 0:
        save_path = os.path.join(ai_dir, f"{i:06d}.png")
        saved_ai += 1
    else:
        save_path = os.path.join(real_dir, f"{i:06d}.png")
        saved_real += 1

    img.save(save_path)

print("Done.")
print("Saved AI:", saved_ai)
print("Saved Real:", saved_real)
