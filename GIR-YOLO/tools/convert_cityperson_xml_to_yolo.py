# -*- coding: utf-8 -*-
"""
Convert CityPerson-style VOC XML annotations to the YOLO layout used by this repo.

Input:
    data/cityperson/
      train/*.png + train/*.xml
      val/*.png + val/*.xml
      test/*.png + test/*.xml

Output:
    data/cityperson_yolo/
      images/train/*.png
      labels/train/*.txt
      images/val/*.png
      labels/val/*.txt
      images/test/*.png
      labels/test/*.txt

YOLO label format:
    class_id x_center y_center width height
where coordinates are normalized to [0, 1].
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def parse_size(root, image_path):
    size = root.find("size")
    if size is not None:
        width_node = size.find("width")
        height_node = size.find("height")
        if width_node is not None and height_node is not None:
            width = int(float(width_node.text))
            height = int(float(height_node.text))
            if width > 0 and height > 0:
                return width, height

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("XML 中缺少有效 size 字段，请安装 Pillow 后重试。") from exc

    with Image.open(image_path) as img:
        return img.size


def clamp(value, low, high):
    return max(low, min(high, value))


def voc_box_to_yolo(box, width, height):
    xmin, ymin, xmax, ymax = box
    xmin = clamp(xmin, 0.0, float(width - 1))
    xmax = clamp(xmax, 0.0, float(width - 1))
    ymin = clamp(ymin, 0.0, float(height - 1))
    ymax = clamp(ymax, 0.0, float(height - 1))

    box_w = xmax - xmin
    box_h = ymax - ymin
    if box_w <= 0 or box_h <= 0:
        return None

    x_center = xmin + box_w / 2.0
    y_center = ymin + box_h / 2.0
    return (
        x_center / width,
        y_center / height,
        box_w / width,
        box_h / height,
    )


def convert_xml(xml_path, image_path, class_map, min_height):
    root = ET.parse(xml_path).getroot()
    width, height = parse_size(root, image_path)
    labels = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None:
            continue

        class_name = name_node.text.strip().lower()
        if class_name not in class_map:
            continue

        box_node = obj.find("bndbox")
        if box_node is None:
            continue

        try:
            xmin = float(box_node.find("xmin").text)
            ymin = float(box_node.find("ymin").text)
            xmax = float(box_node.find("xmax").text)
            ymax = float(box_node.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue

        if ymax - ymin < min_height:
            continue

        yolo_box = voc_box_to_yolo((xmin, ymin, xmax, ymax), width, height)
        if yolo_box is None:
            continue

        labels.append((class_map[class_name],) + yolo_box)

    return labels


def find_image(split_dir, stem):
    for ext in IMAGE_EXTS:
        image_path = split_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def convert_split(src_root, dst_root, split, class_map, min_height, copy_images):
    src_split = src_root / split
    image_dst_dir = dst_root / "images" / split
    label_dst_dir = dst_root / "labels" / split
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_split.exists():
        print(f"[跳过] {src_split} 不存在")
        return 0, 0, 0

    image_count = 0
    label_count = 0
    box_count = 0

    for xml_path in sorted(src_split.glob("*.xml")):
        image_path = find_image(src_split, xml_path.stem)
        if image_path is None:
            print(f"[警告] 找不到对应图片：{xml_path.name}")
            continue

        labels = convert_xml(xml_path, image_path, class_map, min_height)

        dst_image_path = image_dst_dir / image_path.name
        if copy_images:
            shutil.copy2(image_path, dst_image_path)
        elif not dst_image_path.exists():
            try:
                dst_image_path.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, dst_image_path)

        dst_label_path = label_dst_dir / f"{xml_path.stem}.txt"
        with dst_label_path.open("w", encoding="utf-8") as f:
            for label in labels:
                class_id, x_center, y_center, box_w, box_h = label
                f.write(
                    f"{class_id} {x_center:.8f} {y_center:.8f} "
                    f"{box_w:.8f} {box_h:.8f}\n"
                )

        image_count += 1
        label_count += 1
        box_count += len(labels)

    return image_count, label_count, box_count


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CityPerson VOC XML data to YOLO layout.")
    parser.add_argument("--src", default="data/cityperson", type=str, help="原始数据根目录")
    parser.add_argument("--dst", default="data/cityperson_yolo", type=str, help="输出 YOLO 数据根目录")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="需要转换的子集")
    parser.add_argument("--min_height", default=0.0, type=float, help="转换时保留的最小框高度，单位为像素")
    parser.add_argument("--copy_images", action="store_true", default=True, help="复制图片到输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    class_map = {
        "person": 0,
        "pedestrian": 0,
    }

    total_images = 0
    total_labels = 0
    total_boxes = 0
    for split in args.splits:
        image_count, label_count, box_count = convert_split(
            src_root=src_root,
            dst_root=dst_root,
            split=split,
            class_map=class_map,
            min_height=args.min_height,
            copy_images=args.copy_images,
        )
        total_images += image_count
        total_labels += label_count
        total_boxes += box_count
        print(f"[{split}] images={image_count}, labels={label_count}, boxes={box_count}")

    print(f"[完成] 输出目录：{dst_root}")
    print(f"[总计] images={total_images}, labels={total_labels}, boxes={total_boxes}")


if __name__ == "__main__":
    main()
