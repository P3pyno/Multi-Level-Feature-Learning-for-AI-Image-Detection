import os
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pywt
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift
from scipy.fftpack import dct

from scripts.project_paths import DATA_DIR, BRANCH1_FEATURES_CSV

# ----------------------------
# Utils
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root_dir):
    out = []
    for dp, _, fnames in os.walk(root_dir):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
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

# ----------------------------
# Branch-1 Feature: Noise residual via wavelet denoise
# ----------------------------
def wavelet_denoise(img, wavelet="db2", level=2):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    # Estimate noise sigma from HH1 (last level detail)
    cA, details = coeffs[0], coeffs[1:]
    (cH1, cV1, cD1) = details[-1]
    sigma = np.median(np.abs(cD1)) / 0.6745 + 1e-8
    # Universal threshold
    thr = sigma * math.sqrt(2.0 * math.log(img.size + 1))

    new_coeffs = [cA]
    for (cH, cV, cD) in details:
        new_coeffs.append((
            pywt.threshold(cH, thr, mode="soft"),
            pywt.threshold(cV, thr, mode="soft"),
            pywt.threshold(cD, thr, mode="soft"),
        ))
    den = pywt.waverec2(new_coeffs, wavelet=wavelet)
    den = np.clip(den, 0.0, 1.0)
    den = den[:img.shape[0], :img.shape[1]]
    return den

def noise_residual_features(img):
    den = wavelet_denoise(img, wavelet="db2", level=2)
    res = img - den
    r = res.flatten()

    feats = {}
    feats["nr_mean"] = float(np.mean(r))
    feats["nr_std"]  = float(np.std(r))
    feats["nr_skew"] = float(skew(r))
    feats["nr_kurt"] = float(kurtosis(r, fisher=True))
    feats["nr_energy"] = float(np.mean(r * r))
    feats["nr_abs_mean"] = float(np.mean(np.abs(r)))
    feats["nr_abs_q90"]  = float(np.quantile(np.abs(r), 0.90))
    feats["nr_abs_q99"]  = float(np.quantile(np.abs(r), 0.99))
    return feats

# ----------------------------
# Branch-1 Feature: FFT radial spectrum
# ----------------------------
def radial_profile(mag, n_bins=32):
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.astype(np.int32)

    r_max = int(r.max())
    # map radii into n_bins
    bin_edges = np.linspace(0, r_max + 1, n_bins + 1).astype(np.int32)
    prof = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        r0, r1 = bin_edges[i], bin_edges[i + 1]
        mask = (r >= r0) & (r < r1)
        if np.any(mask):
            prof[i] = float(np.mean(mag[mask]))
        else:
            prof[i] = 0.0

    s = prof.sum() + 1e-8
    return prof / s

def fft_radial_features(img, n_bins=32):
    F = fftshift(fft2(img))
    mag = np.abs(F).astype(np.float32)
    mag = np.log1p(mag)

    prof = radial_profile(mag, n_bins=n_bins)
    feats = {}
    for i, v in enumerate(prof):
        feats[f"fft_rad_{i:02d}"] = float(v)
    # extra summary features
    feats["fft_mag_mean"] = float(np.mean(mag))
    feats["fft_mag_std"]  = float(np.std(mag))
    return feats

# ----------------------------
# Branch-1 Feature: DCT coefficient statistics (8x8 blocks)
# ----------------------------
def dct2(block):
    # Orthonormal 2D DCT
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")

def dct_block_features(img, block=8):
    h, w = img.shape
    H = h - (h % block)
    W = w - (w % block)
    x = img[:H, :W]
    feats["dct_skew"]     = float(skew(coeffs))
    feats["dct_kurt"]     = float(kurtosis(coeffs, fisher=True))
    coeffs = []
    lowE = 0.0
    highE = 0.0
    totalE = 0.0

    for y in range(0, H, block):
        for x0 in range(0, W, block):
            b = x[y:y+block, x0:x0+block]
            C = dct2(b)
            # exclude DC for coefficient stats
            c = C.flatten()
            c_ac = np.delete(c, 0)
            coeffs.append(c_ac)

            # energy bands: low freq = top-left 4x4 (excluding DC), high = rest
            low = C[:4, :4].copy()
            low[0,0] = 0.0
            high = C.copy()
            high[:4, :4] = 0.0

            e_low  = float(np.sum(low * low))
            e_high = float(np.sum(high * high))
            e_tot  = float(np.sum(C * C))
            lowE += e_low
            highE += e_high
            totalE += e_tot

    coeffs = np.concatenate(coeffs, axis=0).astype(np.float32)
    feats = {}
    feats["dct_abs_mean"] = float(np.mean(np.abs(coeffs)))
    feats["dct_abs_std"]  = float(np.std(np.abs(coeffs)))
    feats["dct_skew"] = float(s) if np.isfinite(s) else 0.0
    feats["dct_kurt"] = float(k) if np.isfinite(k) else 0.0
    feats["dct_lowE_ratio"]  = float(lowE  / (totalE + 1e-8))
    feats["dct_highE_ratio"] = float(highE / (totalE + 1e-8))
    return feats

# ----------------------------
# Branch-1 Feature: Frequency asymmetry metrics
# ----------------------------
def freq_asymmetry_features(img):
    F = fftshift(fft2(img))
    mag = np.abs(F).astype(np.float32) + 1e-8
    mag = np.log1p(mag)

    h, w = mag.shape
    # Left-right asymmetry
    left = mag[:, :w//2]
    right = mag[:, w - w//2:]
    right_flipped = np.fliplr(right)
    lr = left - right_flipped
    # Up-down asymmetry
    top = mag[:h//2, :]
    bottom = mag[h - h//2:, :]
    bottom_flipped = np.flipud(bottom)
    ud = top - bottom_flipped

    feats = {}
    feats["asym_lr_l1"] = float(np.mean(np.abs(lr)))
    feats["asym_lr_l2"] = float(np.sqrt(np.mean(lr * lr)))
    feats["asym_ud_l1"] = float(np.mean(np.abs(ud)))
    feats["asym_ud_l2"] = float(np.sqrt(np.mean(ud * ud)))
    return feats

# ----------------------------
# Main extraction
# ----------------------------
def extract_branch1(img):
    feats = {}
    feats.update(noise_residual_features(img))
    feats.update(fft_radial_features(img, n_bins=32))
    feats.update(dct_block_features(img, block=8))
    feats.update(freq_asymmetry_features(img))
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
        ai_imgs = ai_imgs[:limit]

    rows = []
    for label, paths in [(0, real_imgs), (1, ai_imgs)]:
        for p in tqdm(paths, desc=f"Extract label={label}"):
            try:
                img = read_gray(p, size=size)
                feats = extract_branch1(img)
                feats["path"] = p
                feats["label"] = label
                rows.append(feats)
            except Exception as e:
                # skip unreadable/bad images
                print(f"[WARN] {p}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(DATA_DIR), help="Path containing ai/ and real/")
    ap.add_argument("--out", type=str, default=str(BRANCH1_FEATURES_CSV))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.data_root, args.out, size=args.size, limit=args.limit)