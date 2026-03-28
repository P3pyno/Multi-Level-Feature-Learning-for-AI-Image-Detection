import os
from datasets import load_from_disk
from tqdm import tqdm

ARROW_DATASET_DIR = "/data/adam/datasets/AI-Generated-vs-Real-Images-Datasets/temp/train"

EXPORT_DIR = "/data/adam/datasets/AI-Generated-vs-Real-Images-Datasets/data"

ai_dir = os.path.join(EXPORT_DIR, "ai")
real_dir = os.path.join(EXPORT_DIR, "real")

os.makedirs(ai_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

dataset = load_from_disk(ARROW_DATASET_DIR)

print(dataset)
print(dataset.features)

saved = 0

for i, example in enumerate(tqdm(dataset, desc="Extracting")):
    label = int(example["label"])
    save_dir = real_dir if label == 1 else ai_dir

    example["image"].save(os.path.join(save_dir, f"{i}.png"))
    saved += 1

print(f"✅ Extracted {saved} images to {EXPORT_DIR}")
