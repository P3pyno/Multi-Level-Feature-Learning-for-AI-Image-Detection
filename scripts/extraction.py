import argparse
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from scripts.project_paths import DATA_DIR, TEMP_TRAIN_DIR


def main(arrow_dataset_dir, export_dir):
    arrow_dataset_dir = Path(arrow_dataset_dir)
    export_dir = Path(export_dir)

    ai_dir = export_dir / "ai"
    real_dir = export_dir / "real"

    ai_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(arrow_dataset_dir))

    print(dataset)
    print(dataset.features)

    saved = 0
    for i, example in enumerate(tqdm(dataset, desc="Extracting")):
        label = int(example["label"])
        # label convention in this dataset: 0=AI, 1=real
        save_dir = ai_dir if label == 0 else real_dir

        example["image"].save(save_dir / f"{i}.png")
        saved += 1

    print(f"✅ Extracted {saved} images to {export_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow-dataset", type=str, default=str(TEMP_TRAIN_DIR))
    ap.add_argument("--out", type=str, default=str(DATA_DIR))
    args = ap.parse_args()

    main(args.arrow_dataset, args.out)
