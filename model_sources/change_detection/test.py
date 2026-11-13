import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


METHOD_A_MEAN = np.array([0.406, 0.456, 0.485, 0.406, 0.456, 0.485], dtype=np.float32)
METHOD_A_STD = np.array([0.225, 0.224, 0.229, 0.225, 0.224, 0.229], dtype=np.float32)
METHOD_B_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
METHOD_B_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
BASE_SIZE = 256
BASE_PARAMS = 2500356
BASE_FLOPS = 2663153664


def _first_existing(root, names):
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def _normalize_pipeline_name(pipeline):
    name = str(pipeline).strip().lower()
    if name not in {"methoda", "methodb"}:
        raise ValueError("Unsupported data pipeline: {}. Use methodA or methodB.".format(pipeline))
    return name


def _apply_method_a(pre, post, size, preserve_ab_order):
    image = np.concatenate([pre, post], axis=2).astype(np.float32)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image / 255.0
    image = (image - METHOD_A_MEAN) / METHOD_A_STD
    if preserve_ab_order:
        return np.concatenate([image[:, :, 2::-1], image[:, :, 5:2:-1]], axis=2)
    return image[:, :, ::-1].copy()


def _apply_method_b(pre, post, size):
    pre = cv2.resize(pre, (size, size), interpolation=cv2.INTER_LINEAR)
    post = cv2.resize(post, (size, size), interpolation=cv2.INTER_LINEAR)
    pre = pre[:, :, ::-1].astype(np.float32) / 255.0
    post = post[:, :, ::-1].astype(np.float32) / 255.0
    pre = (pre - METHOD_B_MEAN) / METHOD_B_STD
    post = (post - METHOD_B_MEAN) / METHOD_B_STD
    return np.concatenate([pre, post], axis=2).copy()


def apply_preprocess(method, pre, post, size, preserve_ab_order):
    method = _normalize_pipeline_name(method)
    if method == "methodb":
        return _apply_method_b(pre, post, size)
    return _apply_method_a(pre, post, size, preserve_ab_order)


class ChangeDataset(Dataset):
    def __init__(self, root, split="test", size=256, preserve_ab_order=False, pipeline="methodA"):
        root = Path(root)
        split_root = root / split
        if split_root.is_dir():
            root = split_root

        self.pre_dir = _first_existing(root, ["A", "time1", "Time1", "T1", "t1"])
        self.post_dir = _first_existing(root, ["B", "time2", "Time2", "T2", "t2"])
        self.label_dir = _first_existing(root, ["label", "labels", "gt", "GT", "mask", "masks"])
        if self.pre_dir is None or self.post_dir is None or self.label_dir is None:
            raise FileNotFoundError("Missing time1/time2 or A/B and label folders under: {}".format(root))

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.files = sorted(
            [
                p.name
                for p in self.pre_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_exts and ":zone.identifier" not in p.name.lower()
            ]
        )
        if not self.files:
            raise FileNotFoundError("No test images found under: {}".format(self.pre_dir))

        self.size = int(size)
        self.preserve_ab_order = bool(preserve_ab_order)
        self.pipeline = _normalize_pipeline_name(pipeline)

    def __len__(self):
        return len(self.files)

    def _read_label(self, name):
        path = self.label_dir / name
        label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError("Could not read label: {}".format(path))
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        label = np.ceil(label.astype(np.float32) / 255.0).astype(np.int64)
        return torch.from_numpy(label)

    def __getitem__(self, index):
        name = self.files[index]
        pre_path = self.pre_dir / name
        post_path = self.post_dir / name
        pre = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
        post = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError("Could not read image: {}".format(pre_path))
        if post is None:
            raise FileNotFoundError("Could not read image: {}".format(post_path))

        image = apply_preprocess(self.pipeline, pre, post, self.size, self.preserve_ab_order)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = self._read_label(name)
        return image, label, name


def update_confusion(pred, target, counts):
    pred = pred.astype(bool)
    target = target.astype(bool)
    counts["tp"] += int((pred & target).sum())
    counts["fp"] += int((pred & ~target).sum())
    counts["fn"] += int((~pred & target).sum())
    counts["tn"] += int((~pred & ~target).sum())
    return counts


def metrics_from_confusion(counts):
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    eps = 1e-12

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)

    total = tp + fp + fn + tn
    po = oa
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total + eps)
    kappa = (po - pe) / (1.0 - pe + eps)
    fp_rate = fp / (fp + tn + eps)

    return {
        "F1": float(f1),
        "IoU": float(iou),
        "OA": float(oa),
        "Precision": float(precision),
        "Recall": float(recall),
        "Kappa": float(kappa),
        "FP_Rate": float(fp_rate),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def model_stats_for_size(size):
    scale = (float(size) * float(size)) / float(BASE_SIZE * BASE_SIZE)
    flops = int(round(float(BASE_FLOPS) * scale))
    return {
        "Params": int(BASE_PARAMS),
        "FLOPs": flops,
        "Params_M": round(float(BASE_PARAMS) / 1e6, 4),
        "FLOPs_G": round(float(flops) / 1e9, 4),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Inference/evaluation for the SeCoR inference-only release package.")
    parser.add_argument("--checkpoint", required=True, help="Path to TorchScript .pt checkpoint.")
    parser.add_argument("--data_root", required=True, help="Dataset root. Supports root/test/{A,B,label}.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for predictions and metrics.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--data_pipeline", default="methodA", help="Test preprocessing preset: methodA or methodB.")
    parser.add_argument("--preserve_ab_order", action="store_true")
    parser.add_argument("--no_save", action="store_true", help="Do not save per-image prediction masks.")
    return parser.parse_args()


def load_model(checkpoint, device):
    model = torch.jit.load(str(checkpoint), map_location=device)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    pred_dir = out_dir / "pred_masks"
    prob_dir = out_dir / "prob_npy"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save:
        pred_dir.mkdir(parents=True, exist_ok=True)
        prob_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device=device)
    dataset = ChangeDataset(
        args.data_root,
        split="test",
        size=args.size,
        preserve_ab_order=args.preserve_ab_order,
        pipeline=args.data_pipeline,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    with torch.no_grad():
        for batch_idx, (image, target, names) in enumerate(loader, start=1):
            image = image.to(device, non_blocking=True)
            pre = image[:, 0:3]
            post = image[:, 3:6]
            logits = model(pre, post)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            pred = (prob >= float(args.threshold)).astype(np.uint8)
            target_np = target.numpy().astype(np.uint8)

            for i, name in enumerate(names):
                pred_i = pred[i, 0]
                prob_i = prob[i, 0]
                gt_i = target_np[i]
                update_confusion(pred_i, gt_i, counts)
                if not args.no_save:
                    stem = Path(name).stem
                    cv2.imwrite(str(pred_dir / (stem + ".png")), pred_i * 255)
                    np.save(str(prob_dir / (stem + ".npy")), prob_i.astype(np.float32))

            if batch_idx % 20 == 0:
                print("[batch {}/{}]".format(batch_idx, len(loader)), flush=True)

    metrics = metrics_from_confusion(counts)
    model_stats = model_stats_for_size(args.size)
    result = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "threshold": float(args.threshold),
        "data_pipeline": args.data_pipeline,
        "model": model_stats,
        "metrics": metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(json.dumps({"model": model_stats, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
