import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def _normalize_path(p):
    return str(Path(p))


def get_or_create_global_path_split(
    paths,
    labels,
    split_path,
    test_size=0.2,
    random_state=42,
):
    """Create-once and reuse global split based on image path."""
    norm_paths = np.array([_normalize_path(p) for p in paths], dtype=object)
    labels = np.array(labels)
    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists():
        data = json.loads(split_path.read_text())
        train_set = set(data.get("train_paths", []))
        test_set = set(data.get("test_paths", []))
    else:
        unique_paths, unique_idx = np.unique(norm_paths, return_index=True)
        unique_labels = labels[unique_idx]

        train_paths, test_paths = train_test_split(
            unique_paths,
            test_size=test_size,
            random_state=random_state,
            stratify=unique_labels,
        )
        train_set = set(map(str, train_paths.tolist()))
        test_set = set(map(str, test_paths.tolist()))

        payload = {
            "test_size": test_size,
            "random_state": random_state,
            "num_unique_paths": int(len(unique_paths)),
            "train_paths": sorted(train_set),
            "test_paths": sorted(test_set),
        }
        split_path.write_text(json.dumps(payload, indent=2))

    train_idx = np.array([i for i, p in enumerate(norm_paths) if p in train_set], dtype=int)
    test_idx = np.array([i for i, p in enumerate(norm_paths) if p in test_set], dtype=int)

    overlap = set(norm_paths[train_idx]).intersection(set(norm_paths[test_idx]))
    if overlap:
        raise ValueError(f"Path split overlap detected: {len(overlap)} items")

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Global split produced empty train/test partition for this dataset.")

    return train_idx, test_idx
