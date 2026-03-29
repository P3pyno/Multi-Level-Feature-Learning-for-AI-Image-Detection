import argparse

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch

from scripts.branch1.branch1_extract import read_gray as b1_read_gray, extract_branch1
from scripts.branch2.branch2_extract import read_gray as b2_read_gray, extract_branch2_handcrafted
from scripts.branch2.branch2_cnn_extract import read_rgb, to_tensor, ResNetFeatureExtractor
from scripts.branch3.branch3_train import semantic_distance_features
from scripts.branch3.branch3_train_v2 import semantic_features as semantic_features_v2
from scripts.fusion.gating import apply_branch_gates
from scripts.project_paths import (
    BRANCH1_MODEL,
    BRANCH3_SEMANTIC_MODEL,
    BRANCH3_SEMANTIC_V2_MODEL,
    FUSION_MODEL,
    FUSION_V2_MODEL,
)


def split_quadrants(img):
    w, h = img.size
    w2, h2 = w // 2, h // 2
    return [
        img.crop((0, 0, w2, h2)),
        img.crop((w2, 0, w, h2)),
        img.crop((0, h2, w2, h)),
        img.crop((w2, h2, w, h)),
    ]


def _load_clip_image_encoder(device):
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        model = model.to(device)
        model.eval()

        def encode(img):
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z = model.encode_image(x)
                z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return z.detach().cpu().numpy().astype(np.float32)

        return encode
    except Exception:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model.eval()

        def encode(img):
            x = proc(images=[img], return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                z = model.get_image_features(pixel_values=x)
                z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return z.detach().cpu().numpy().astype(np.float32)

        return encode


def extract_branch2_cnn_single(image_path, arch="resnet50", size=224):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetFeatureExtractor(arch=arch).to(device)
    model.eval()

    img = read_rgb(image_path, size=size)
    x = to_tensor(img).to(device)
    with torch.no_grad():
        z2, z3, z4 = model(x)
    z = torch.cat([z2, z3, z4], dim=1).cpu().numpy()[0]
    return {f"b2b_cnn_{j:04d}": float(z[j]) for j in range(z.shape[0])}


def extract_branch3_semantic_single(image_path, b3_model_path):
    bundle = joblib.load(b3_model_path)
    ref = bundle["reference"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode = _load_clip_image_encoder(device)

    img = Image.open(image_path).convert("RGB")
    emb = encode(img)
    feats = semantic_distance_features(emb, ref)[0]

    return {
        "b3_mahal": float(feats[0]),
        "b3_knn": float(feats[1]),
        "b3_cos": float(feats[2]),
    }


def extract_branch3_semantic_v2_single(image_path, b3_model_path):
    bundle = joblib.load(b3_model_path)
    ref = bundle["reference"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode = _load_clip_image_encoder(device)

    img = Image.open(image_path).convert("RGB")
    global_emb = encode(img)

    quads = split_quadrants(img)
    quad_emb = np.stack([encode(q)[0] for q in quads], axis=0)[None, :, :]

    feats = semantic_features_v2(global_emb, quad_emb, ref)[0]
    return {
        "b3_mahal": float(feats[0]),
        "b3_knn_mean": float(feats[1]),
        "b3_knn_min": float(feats[2]),
        "b3_cos_centroid": float(feats[3]),
        "b3_l2_centroid": float(feats[4]),
        "b3_pair_mean": float(feats[5]),
        "b3_pair_std": float(feats[6]),
        "b3_pair_min": float(feats[7]),
        "b3_pair_max": float(feats[8]),
        "b3_gq_mean": float(feats[9]),
        "b3_gq_std": float(feats[10]),
        "b3_gq_min": float(feats[11]),
        "b3_gq_max": float(feats[12]),
    }


def predict_branch1(image_path, model_path=BRANCH1_MODEL, size=256):
    model = joblib.load(model_path)

    img = b1_read_gray(image_path, size=size)
    feats = extract_branch1(img)
    X = pd.DataFrame([feats])

    prob_ai = float(model.predict_proba(X)[0, 1])
    pred = int(prob_ai >= 0.5)

    return {
        "image_path": image_path,
        "pred_label": pred,
        "pred_class": "ai" if pred == 1 else "real",
        "prob_ai": prob_ai,
        "confidence": float(abs(prob_ai - 0.5) * 2.0),
    }


def predict_fusion_single(
    image_path,
    fusion_model_path=FUSION_MODEL,
    b3_model_path=BRANCH3_SEMANTIC_MODEL,
    fusion_version="v1",
):
    fusion_bundle = joblib.load(fusion_model_path)
    if isinstance(fusion_bundle, dict) and "model" in fusion_bundle:
        pipe = fusion_bundle["model"]
        use_gating = bool(fusion_bundle.get("use_branch_gating", False))
        gate_params = fusion_bundle.get("gate_params", None)
    else:
        pipe = fusion_bundle
        use_gating = False
        gate_params = None

    b1_img = b1_read_gray(image_path, size=256)
    b1 = {f"b1_{k}": v for k, v in extract_branch1(b1_img).items()}

    b2_img = b2_read_gray(image_path, size=256)
    b2a = {f"b2a_{k}": v for k, v in extract_branch2_handcrafted(b2_img).items()}

    b2b = extract_branch2_cnn_single(image_path, arch="resnet50", size=224)
    if fusion_version == "v2":
        b3 = extract_branch3_semantic_v2_single(image_path, b3_model_path)
    else:
        b3 = extract_branch3_semantic_single(image_path, b3_model_path)

    row = {}
    row.update(b1)
    row.update(b2a)
    row.update(b2b)
    row.update(b3)

    feat_cols = list(pipe.named_steps["imputer"].feature_names_in_)
    X = pd.DataFrame([[row.get(c, 0.0) for c in feat_cols]], columns=feat_cols)

    if use_gating and gate_params is not None:
        X = apply_branch_gates(X, gate_params)

    prob_ai = float(pipe.predict_proba(X)[0, 1])
    pred = int(prob_ai >= 0.5)

    return {
        "image_path": image_path,
        "pred_label": pred,
        "pred_class": "ai" if pred == 1 else "real",
        "prob_ai": prob_ai,
        "confidence": float(abs(prob_ai - 0.5) * 2.0),
    }


def main():
    ap = argparse.ArgumentParser(description="Single-image inference utility")
    ap.add_argument("image", type=str, help="Path to input image")
    ap.add_argument("--mode", choices=["branch1", "fusion"], default="fusion")
    ap.add_argument("--fusion-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--abstain-threshold", type=float, default=0.55, help="If max class probability is below this, output abstain.")
    ap.add_argument("--branch1-model", type=str, default=str(BRANCH1_MODEL))
    ap.add_argument("--fusion-model", type=str, default=None)
    ap.add_argument("--branch3-model", type=str, default=None)
    args = ap.parse_args()

    if args.mode == "branch1":
        out = predict_branch1(args.image, model_path=args.branch1_model)
    else:
        fusion_model = args.fusion_model or (str(FUSION_V2_MODEL) if args.fusion_version == "v2" else str(FUSION_MODEL))
        branch3_model = args.branch3_model or (str(BRANCH3_SEMANTIC_V2_MODEL) if args.fusion_version == "v2" else str(BRANCH3_SEMANTIC_MODEL))
        out = predict_fusion_single(
            args.image,
            fusion_model_path=fusion_model,
            b3_model_path=branch3_model,
            fusion_version=args.fusion_version,
        )

    max_prob = max(out["prob_ai"], 1.0 - out["prob_ai"])
    if max_prob < args.abstain_threshold:
        out["pred_label"] = -1
        out["pred_class"] = "abstain"
    out["abstain_threshold"] = args.abstain_threshold
    out["max_prob"] = float(max_prob)

    print(out)


if __name__ == "__main__":
    main()
