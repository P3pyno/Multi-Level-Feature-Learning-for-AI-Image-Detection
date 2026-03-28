import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from scripts.project_paths import DATA_DIR, BRANCH2_FEATURES_CSV

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root_dir):
    out = []
    for dp, _, fnames in os.walk(root_dir):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dp, f))
    return sorted(out)

def read_gray(path, size=256):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

# ---- Handcrafted texture features ----

def lbp_hist(img, P=8, R=1):
    # convert to uint8 for stable LBP
    x = np.clip((img * 255.0).round(), 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(x, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return {f"lbp_{i:02d}": float(hist[i]) for i in range(len(hist))}

def glcm_feats(img, levels=16, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    # quantize to [0..levels-1]
    x = np.clip((img * (levels - 1)).round(), 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(x, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = {}
    for p in props:
        v = graycoprops(glcm, p)  # shape: (len(dist), len(angles))
        feats[f"glcm_{p}_mean"] = float(np.mean(v))
        feats[f"glcm_{p}_std"]  = float(np.std(v))
    return feats

def texture_entropy(img, n_bins=256):
    x = np.clip((img * (n_bins - 1)).round(), 0, n_bins - 1).astype(np.uint8)
    hist = np.bincount(x.ravel(), minlength=n_bins).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p))
    return {"tex_entropy": float(ent)}

def patch_self_similarity(img, patch=48, n_pairs=16, seed=0):
    # fast self-sim using cosine similarity between random patches
    rng = np.random.default_rng(seed)
    h, w = img.shape
    if h < patch or w < patch:
        return {"pss_cos_mean": 0.0, "pss_cos_std": 0.0}

    def sample_patch():
        y = int(rng.integers(0, h - patch + 1))
        x = int(rng.integers(0, w - patch + 1))
        p = img[y:y+patch, x:x+patch].astype(np.float32)
        p = p - p.mean()
        n = np.linalg.norm(p.ravel()) + 1e-8
        return (p.ravel() / n)

    sims = []
    for _ in range(n_pairs):
        a = sample_patch()
        b = sample_patch()
        sims.append(float(np.dot(a, b)))
    sims = np.array(sims, dtype=np.float32)
    return {"pss_cos_mean": float(np.mean(sims)), "pss_cos_std": float(np.std(sims))}

def extract_branch2_handcrafted(img):
    feats = {}
    feats.update(lbp_hist(img, P=8, R=1))
    feats.update(glcm_feats(img, levels=16))
    feats.update(texture_entropy(img))
    feats.update(patch_self_similarity(img, patch=48, n_pairs=16, seed=0))

    # safety: force finite
    for k, v in list(feats.items()):
        v = float(v) if np.isfinite(v) else 0.0
        feats[k] = v
    return feats

def main(data_root, out_path, size=256, limit=None):
    data_root = str(data_root)
    out_path = str(out_path)

    real_dir = os.path.join(data_root, "real")
    ai_dir   = os.path.join(data_root, "ai")

    real_imgs = list_images(real_dir)
    ai_imgs   = list_images(ai_dir)

    if limit is not None:
        real_imgs = real_imgs[:limit]
        ai_imgs   = ai_imgs[:limit]

    rows = []
    for label, paths in [(0, real_imgs), (1, ai_imgs)]:
        for p in tqdm(paths, desc=f"Extract label={label}"):
            try:
                img = read_gray(p, size=size)
                feats = extract_branch2_handcrafted(img)
                feats["path"] = p
                feats["label"] = label
                rows.append(feats)
            except Exception as e:
                print(f"[WARN] {p}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(DATA_DIR))
    ap.add_argument("--out", type=str, default=str(BRANCH2_FEATURES_CSV))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.data_root, args.out, size=args.size, limit=args.limit)