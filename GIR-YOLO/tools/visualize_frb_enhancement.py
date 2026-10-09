import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import create_model


def collect_images(image_dir, limit, seed):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [p for p in Path(image_dir).rglob('*') if p.is_file() and p.suffix.lower() in exts]
    images = sorted(images)
    if limit and len(images) > limit:
        random.seed(seed)
        images = sorted(random.sample(images, limit))
    return images


def load_model(args, device):
    model_args = argparse.Namespace(
        model=args.model,
        pretrained='',
        frb_num_basis=args.frb_num_basis,
        frb_hidden_channels=args.frb_hidden_channels,
        frb_strength=args.frb_strength,
        frb_gate_init=args.frb_gate_init,
    )
    model = create_model(model_args).to(device)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def tensor_to_rgb_image(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x * std + mean).clamp(0.0, 1.0)
    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    return (img * 255.0).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--model', default='lowlight_cspnet_v2')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_h', type=int, default=384)
    parser.add_argument('--input_w', type=int, default=768)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frb_num_basis', type=int, default=10)
    parser.add_argument('--frb_hidden_channels', type=int, default=16)
    parser.add_argument('--frb_strength', type=float, default=0.1)
    parser.add_argument('--frb_gate_init', type=float, default=-1.0)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args, device)
    if not hasattr(model, 'frbnet'):
        raise ValueError(f'Model {args.model} does not expose model.frbnet')

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = collect_images(args.image_dir, args.limit, args.seed)

    for image_path in images:
        image = Image.open(image_path).convert('RGB')
        resized = image.resize((args.input_w, args.input_h), Image.BILINEAR)
        x = transform(resized).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced = model.frbnet(x)

        original = np.array(resized)
        enhanced_img = tensor_to_rgb_image(enhanced)
        compare = np.concatenate([original, enhanced_img], axis=1)
        compare = cv2.cvtColor(compare, cv2.COLOR_RGB2BGR)

        rel = image_path.relative_to(args.image_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), compare)
        print(f'saved {out_path}')


if __name__ == '__main__':
    main()
