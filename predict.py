"""Vegetable classifier: frozen ResNet18 embeddings + (calibrated) logistic regression.

Usage:
    python predict.py --input path/to/folder --output preds.csv [--threshold 0.83]

Runs on CPU, no internet needed: all weights are loaded from ./weights.
"""
import argparse
import csv
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def pad_square(img, fill=(124, 116, 104)):
    # Fit the whole photo into a square (no cropping), same as in training
    w, h = img.size
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


# Same preprocessing as in training (see notebook)
TF = transforms.Compose([
    pad_square,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_rgb(path, max_side=320):
    img = Image.open(path)
    img.draft("RGB", (max_side, max_side))  # fast JPEG downscale while decoding
    img = ImageOps.exif_transpose(img)      # phone photos: respect EXIF rotation
    return img.convert("RGB")


def load_backbone(weights_path):
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Identity()
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return net.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder with images")
    ap.add_argument("--output", required=True, help="path to preds.csv")
    ap.add_argument("--weights-dir", default=str(HERE / "weights"))
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="confidence below this -> 'unknown' (0 = never unknown)")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    t_start = time.perf_counter()
    wdir = Path(args.weights_dir)
    net = load_backbone(wdir / "backbone_resnet18.pt")
    bundle = joblib.load(wdir / "classifier.joblib")
    clf, classes = bundle["model"], bundle["classes"]
    t_loaded = time.perf_counter()

    root = Path(args.input)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    rows = []

    for s in range(0, len(files), args.batch_size):
        batch = files[s:s + args.batch_size]
        tensors, names = [], []
        for p in batch:
            name = p.relative_to(root).as_posix()
            try:
                tensors.append(TF(load_rgb(p)))
                names.append(name)
            except Exception as e:  # broken file: do not crash the whole run
                print(f"WARNING: cannot read {name}: {e}")
                rows.append((name, "unknown", 0.0))
        if not tensors:
            continue
        with torch.inference_mode():
            emb = net(torch.stack(tensors)).numpy()
        probs = clf.predict_proba(emb)
        for name, pr in zip(names, probs):
            k = int(np.argmax(pr))
            conf = float(pr[k])
            label = classes[k] if conf >= args.threshold else "unknown"
            rows.append((name, label, round(conf, 4)))

    rows.sort(key=lambda r: r[0])
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "confidence"])
        w.writerows(rows)

    t_end = time.perf_counter()
    n = max(len(files), 1)
    print(f"images: {len(files)} -> {args.output}")
    print(f"model load: {t_loaded - t_start:.2f} s")
    print(f"per image (read + preprocess + predict, CPU): {1000 * (t_end - t_loaded) / n:.1f} ms")


if __name__ == "__main__":
    main()
