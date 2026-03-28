import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from scripts.auto_gpu import set_visible_gpus
from scripts.project_paths import (
    DATA_DIR,
    BRANCH3_CLIP_V2_GLOBAL_NPY,
    BRANCH3_CLIP_V2_QUADS_NPY,
    BRANCH3_CLIP_V2_META_CSV,
)
set_visible_gpus(memory_threshold_mb=500, util_threshold=10)

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root_dir):
    out = []
    for dp, _, fnames in os.walk(root_dir):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dp, f))
    return sorted(out)


def split_quadrants(img):
    w, h = img.size
    w2, h2 = w // 2, h // 2
    return [
        img.crop((0, 0, w2, h2)),       # TL
        img.crop((w2, 0, w, h2)),       # TR
        img.crop((0, h2, w2, h)),       # BL
        img.crop((w2, h2, w, h)),       # BR
    ]


class HFClipImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        z = self.model.get_image_features(pixel_values=pixel_values)
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return z


def main(data_root, out_prefix=None, batch_size=128, limit=None):
    data_root = str(data_root)

    device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"

    print("[branch3_v2] Loading CLIP...", flush=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    encoder = HFClipImageEncoder(model).to(device).eval()

    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"[branch3_v2] Using DataParallel on {torch.cuda.device_count()} visible GPUs", flush=True)
        encoder = nn.DataParallel(encoder)

    real_dir = os.path.join(data_root, "real")
    ai_dir = os.path.join(data_root, "ai")

    real_imgs = list_images(real_dir)
    ai_imgs = list_images(ai_dir)

    if limit is not None:
        real_imgs = real_imgs[:limit]
        ai_imgs = ai_imgs[:limit]

    meta_rows = []
    global_chunks = []
    quad_chunks = []

    def process_group(paths, label):
        nonlocal meta_rows, global_chunks, quad_chunks

        batch_global = []
        batch_quads = []
        batch_paths = []

        for p in tqdm(paths, desc=f"Extract CLIP-v2 label={label}"):
            try:
                img = Image.open(p).convert("RGB")
                quads = split_quadrants(img)

                batch_global.append(img)
                batch_quads.extend(quads)
                batch_paths.append(p)

                if len(batch_global) >= batch_size:
                    run_batch(batch_global, batch_quads, batch_paths, label)
                    batch_global, batch_quads, batch_paths = [], [], []

            except Exception as e:
                print(f"[WARN] {p}: {e}")

        if len(batch_global) > 0:
            run_batch(batch_global, batch_quads, batch_paths, label)

    def run_batch(global_imgs, quad_imgs, paths, label):
        nonlocal meta_rows, global_chunks, quad_chunks

        global_inputs = processor(images=global_imgs, return_tensors="pt")["pixel_values"].to(device)
        quad_inputs = processor(images=quad_imgs, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            z_global = encoder(global_inputs).detach().cpu().numpy().astype(np.float32)
            z_quad = encoder(quad_inputs).detach().cpu().numpy().astype(np.float32)

        z_quad = z_quad.reshape(len(paths), 4, -1)

        global_chunks.append(z_global)
        quad_chunks.append(z_quad)

        for p in paths:
            meta_rows.append((p, label))

    print(f"[branch3_v2] Device: {device}", flush=True)
    process_group(real_imgs, 0)
    process_group(ai_imgs, 1)

    global_emb = np.concatenate(global_chunks, axis=0)
    quad_emb = np.concatenate(quad_chunks, axis=0)
    meta = pd.DataFrame(meta_rows, columns=["path", "label"])

    np.save(BRANCH3_CLIP_V2_GLOBAL_NPY, global_emb)
    np.save(BRANCH3_CLIP_V2_QUADS_NPY, quad_emb)
    meta.to_csv(BRANCH3_CLIP_V2_META_CSV, index=False)

    print(f"[branch3_v2] Saved global embeddings: {BRANCH3_CLIP_V2_GLOBAL_NPY} shape={global_emb.shape}", flush=True)
    print(f"[branch3_v2] Saved quad embeddings:   {BRANCH3_CLIP_V2_QUADS_NPY} shape={quad_emb.shape}", flush=True)
    print(f"[branch3_v2] Saved meta:              {BRANCH3_CLIP_V2_META_CSV} shape={meta.shape}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(DATA_DIR))
    ap.add_argument("--out_prefix", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    main(
        data_root=args.data_root,
        out_prefix=args.out_prefix,
        batch_size=args.batch_size,
        limit=args.limit,
    )