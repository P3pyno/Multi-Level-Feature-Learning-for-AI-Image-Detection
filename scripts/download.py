import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
ds.save_to_disk("/data/adam/temp/AI-Generated-vs-Real-Images-Datasets/dataset")