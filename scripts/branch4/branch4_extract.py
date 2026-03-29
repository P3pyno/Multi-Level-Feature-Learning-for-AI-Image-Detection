import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from scripts.project_paths import DATA_DIR, BRANCH4_FEATURES_CSV

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root_dir):
    out = []
    for dp, _, fnames in os.walk(root_dir):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dp, f))
    return sorted(out)


def read_rgb(path, size=256):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def jpeg_reencode(img_uint8, quality=90):
    pil = Image.fromarray(img_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    return out


def perturbation_consistency_features(img_uint8):
    x = img_uint8.astype(np.float32) / 255.0

    jpg95 = jpeg_reencode(img_uint8, quality=95).astype(np.float32) / 255.0
    jpg75 = jpeg_reencode(img_uint8, quality=75).astype(np.float32) / 255.0

    up = cv2.resize(img_uint8, (320, 320), interpolation=cv2.INTER_CUBIC)
    down = cv2.resize(up, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    def diff_stats(a, b, prefix):
        d = (a - b).reshape(-1)
        return {
            f"{prefix}_l1": float(np.mean(np.abs(d))),
            f"{prefix}_l2": float(np.sqrt(np.mean(d * d))),
            f"{prefix}_max": float(np.max(np.abs(d))),
        }

    feats = {}
    feats.update(diff_stats(x, jpg95, "aug_jpg95"))
    feats.update(diff_stats(x, jpg75, "aug_jpg75"))
    feats.update(diff_stats(x, down, "aug_resize"))

    # cross-perturbation consistency
    feats.update(diff_stats(jpg95, jpg75, "cross_jpg95_75"))
    feats.update(diff_stats(jpg95, down, "cross_jpg95_resize"))
    feats.update(diff_stats(jpg75, down, "cross_jpg75_resize"))

    return feats


def main(data_root, out_path, size=256, limit=None):
    data_root = str(data_root)
    out_path = str(out_path)

    real_dir = os.path.join(data_root, "real")
    ai_dir = os.path.join(data_root, "ai")

    real_imgs = list_images(real_dir)
    ai_imgs = list_images(ai_dir)

    if limit is not None:
        real_imgs = real_imgs[:limit]
        ai_imgs = ai_imgs[:limit]

    rows = []
    for label, paths in [(0, real_imgs), (1, ai_imgs)]:
        for p in tqdm(paths, desc=f"Extract Branch4 label={label}"):
            try:
                img = read_rgb(p, size=size)
                feats = perturbation_consistency_features(img)
                feats["path"] = p
                feats["label"] = label
                rows.append(feats)
            except Exception as e:
                print(f"[WARN] {p}: {e}")

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Shape:", df.shape)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(DATA_DIR))
    ap.add_argument("--out", type=str, default=str(BRANCH4_FEATURES_CSV))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    main(args.data_root, args.out, size=args.size, limit=args.limit)
