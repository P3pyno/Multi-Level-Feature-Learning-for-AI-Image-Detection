import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import models

from scripts.project_paths import DATA_DIR, BRANCH2_CNN_FEATURES_CSV

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root_dir):
    out = []
    for dp, _, fnames in os.walk(root_dir):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dp, f))
    return sorted(out)

def read_rgb(path, size=224):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def to_tensor(img_rgb):
    # img: HWC float32 [0,1]
    x = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0)  # 1x3xHxW
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    x = (x - mean) / std
    return x

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, arch="resnet50"):
        super().__init__()
        if arch == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif arch == "resnet34":
            m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # stem
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        # stages
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x)
        f2 = self.layer2(x)     # mid-level
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)    # high-level texture/semantics

        # global average pooling
        z2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)
        z3 = F.adaptive_avg_pool2d(f3, 1).flatten(1)
        z4 = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        return z2, z3, z4

def main(data_root, out_path, arch="resnet50", size=224, limit=None, batch_size=32):
    data_root = str(data_root)
    out_path = str(out_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetFeatureExtractor(arch=arch).to(device)
    model.eval()

    real_dir = os.path.join(data_root, "real")
    ai_dir   = os.path.join(data_root, "ai")
    real_imgs = list_images(real_dir)
    ai_imgs   = list_images(ai_dir)

    if limit is not None:
        real_imgs = real_imgs[:limit]
        ai_imgs   = ai_imgs[:limit]

    rows = []

    def process(paths, label):
        nonlocal rows
        batch_imgs = []
        batch_paths = []
        for p in tqdm(paths, desc=f"Extract CNN label={label}"):
            try:
                img = read_rgb(p, size=size)
                x = to_tensor(img)
                batch_imgs.append(x)
                batch_paths.append(p)
                if len(batch_imgs) >= batch_size:
                    run_batch(batch_imgs, batch_paths, label)
                    batch_imgs, batch_paths = [], []
            except Exception as e:
                print(f"[WARN] {p}: {e}")

        if len(batch_imgs) > 0:
            run_batch(batch_imgs, batch_paths, label)

    def run_batch(batch_imgs, batch_paths, label):
        nonlocal rows
        x = torch.cat(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            z2, z3, z4 = model(x)
        z = torch.cat([z2, z3, z4], dim=1).cpu().numpy()

        for i, p in enumerate(batch_paths):
            feats = {f"cnn_{j:04d}": float(z[i, j]) for j in range(z.shape[1])}
            feats["path"] = p
            feats["label"] = label
            rows.append(feats)

    process(real_imgs, 0)
    process(ai_imgs, 1)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("Shape:", df.shape)
    print("Columns:", len(df.columns))
    print("Device:", device, "Arch:", arch)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=str(DATA_DIR))
    ap.add_argument("--out", type=str, default=str(BRANCH2_CNN_FEATURES_CSV))
    ap.add_argument("--arch", type=str, default="resnet50", choices=["resnet18","resnet34","resnet50"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    main(args.data_root, args.out, arch=args.arch, size=args.size, limit=args.limit, batch_size=args.batch_size)