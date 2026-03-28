import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from scripts.auto_gpu import set_visible_gpus
from scripts.project_paths import (
    DATA_DIR,
    BRANCH3_CLIP_EMBEDDINGS_NPY,
    BRANCH3_CLIP_META_CSV,
)

# IMPORTANT: set visible GPUs before importing torch
set_visible_gpus(memory_threshold_mb=500, util_threshold=10)

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root_dir):
    out = []
    for dp, _, fnames in os.walk(root_dir):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dp, f))
    return sorted(out)


class OpenClipImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z = self.model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return z


class HFClipImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        z = self.model.get_image_features(pixel_values=pixel_values)
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return z


def load_clip_backend():
    """
    Try open_clip first. If it fails, fall back to transformers CLIP.
    """
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        backend = "open_clip"
        encoder = OpenClipImageEncoder(model)
        return backend, encoder, preprocess, None
    except Exception as e:
        print("[branch3] open_clip failed:", e)
        print("[branch3] Falling back to transformers CLIP...")
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        backend = "transformers"
        encoder = HFClipImageEncoder(model)
        return backend, encoder, None, processor


def load_batch_open_clip(paths, preprocess):
    imgs = []
    valid = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
            valid.append(p)
        except Exception as e:
            print(f"[WARN] {p}: {e}")
    if len(imgs) == 0:
        return None, []
    x = torch.stack(imgs, dim=0)
    return x, valid


def load_batch_hf(paths, processor):
    imgs = []
    valid = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
            valid.append(p)
        except Exception as e:
            print(f"[WARN] {p}: {e}")
    if len(imgs) == 0:
        return None, []
    x = processor(images=imgs, return_tensors="pt")["pixel_values"]
    return x, valid


def main(data_root, out_prefix=None, batch_size=256, limit=None):
    data_root = str(data_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend, encoder, preprocess, processor = load_clip_backend()

    encoder = encoder.to(device)
    encoder.eval()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[branch3] Wrapping model in DataParallel over {torch.cuda.device_count()} visible GPUs")
        encoder = nn.DataParallel(encoder)

    real_dir = os.path.join(data_root, "real")
    ai_dir = os.path.join(data_root, "ai")

    real_imgs = list_images(real_dir)
    ai_imgs = list_images(ai_dir)

    if limit is not None:
        real_imgs = real_imgs[:limit]
        ai_imgs = ai_imgs[:limit]

    all_emb_chunks = []
    meta_rows = []

    def run_group(paths, label):
        nonlocal all_emb_chunks, meta_rows

        loader = load_batch_open_clip if backend == "open_clip" else load_batch_hf
        aux = preprocess if backend == "open_clip" else processor

        for i in tqdm(range(0, len(paths), batch_size), desc=f"Extract CLIP label={label}"):
            batch_paths = paths[i:i + batch_size]
            x, valid_paths = loader(batch_paths, aux)
            if x is None or len(valid_paths) == 0:
                continue

            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                z = encoder(x).detach().cpu().numpy().astype(np.float32)

            all_emb_chunks.append(z)
            for p in valid_paths:
                meta_rows.append((p, label))

    print(f"[branch3] Backend: {backend} | Device: {device}")
    run_group(real_imgs, 0)
    run_group(ai_imgs, 1)

    embeddings = np.concatenate(all_emb_chunks, axis=0)
    meta = pd.DataFrame(meta_rows, columns=["path", "label"])

    np.save(BRANCH3_CLIP_EMBEDDINGS_NPY, embeddings)
    meta.to_csv(BRANCH3_CLIP_META_CSV, index=False)

    print(f"[branch3] Saved embeddings: {BRANCH3_CLIP_EMBEDDINGS_NPY} shape={embeddings.shape}")
    print(f"[branch3] Saved metadata:   {BRANCH3_CLIP_META_CSV} shape={meta.shape}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(DATA_DIR))
    ap.add_argument("--out_prefix", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    main(
        data_root=args.data_root,
        out_prefix=args.out_prefix,
        batch_size=args.batch_size,
        limit=args.limit,
    )