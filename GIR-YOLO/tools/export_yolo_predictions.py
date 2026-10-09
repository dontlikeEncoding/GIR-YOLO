import argparse
import json
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split(data_cfg, split):
    root = Path(data_cfg.get("path", ""))
    split_path = Path(data_cfg[split])
    if split_path.is_absolute():
        return split_path
    return root / split_path


def yolo_label_to_xyxy(label_file, img_w, img_h):
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts[:5])
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2, int(cls)])
    return boxes


def image_to_label_path(image_path, data_root):
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        label = Path(*parts).with_suffix(".txt")
        return label
    return data_root / "labels" / p.parent.name / f"{p.stem}.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data)
    data_root = Path(data_cfg.get("path", ""))
    image_dir = resolve_split(data_cfg, args.split)
    image_paths = sorted(list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpeg")))
    print("images:", len(image_paths))
    print("weights:", args.weights)
    print("out:", args.out)

    model = YOLO(args.weights)

    records = []
    for i, img_path in enumerate(image_paths):
        if i % 200 == 0:
            print(i, "/", len(image_paths), img_path)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        label_path = image_to_label_path(img_path, data_root)
        gt = yolo_label_to_xyxy(label_path, w, h)

        pred = []
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            for box, score, cls in zip(xyxy, confs, clss):
                pred.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), int(cls)])

        records.append({
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "width": w,
            "height": h,
            "gt": gt,
            "pred": pred,
            "model": args.name
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.name,
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "records": records
    }, ensure_ascii=False), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
