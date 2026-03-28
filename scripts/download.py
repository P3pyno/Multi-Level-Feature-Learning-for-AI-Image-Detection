import argparse
import os

from datasets import load_dataset

from scripts.project_paths import TEMP_DATASET_DICT_JSON


def main(dataset_name, out_dir, hf_endpoint=None):
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    ds = load_dataset(dataset_name)
    out_dir = str(out_dir)
    ds.save_to_disk(out_dir)
    print(f"Saved dataset to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Hemg/AI-Generated-vs-Real-Images-Datasets")
    ap.add_argument("--out", type=str, default=str(TEMP_DATASET_DICT_JSON.parent / "hf_dataset"))
    ap.add_argument("--hf-endpoint", type=str, default=None)
    args = ap.parse_args()

    main(args.dataset, args.out, hf_endpoint=args.hf_endpoint)
