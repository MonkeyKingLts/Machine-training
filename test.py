#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

from config import parse_test_args
from models import CycleGAN
from utils import load_checkpoint, tensor_to_pil


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class SingleDomainDataset(Dataset):
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, folder: str | Path, image_size: int = 256):
        self.folder = Path(folder)
        self.paths = sorted(
            path
            for path in self.folder.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"目录为空或不存在: {self.folder}")
        self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return {"image": image, "path": str(path)}


def save_side_by_side(source: Image.Image, translated: Image.Image, path: Path) -> None:
    width, height = source.size
    canvas = Image.new("RGB", (width * 2, height))
    canvas.paste(source, (0, 0))
    canvas.paste(translated, (width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    cfg = parse_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = CycleGAN().to(device)
    load_checkpoint(model, None, cfg.checkpoint)
    model.eval()

    output_dir = Path(cfg.output_dir)
    dataroot = Path(cfg.dataroot)

    with torch.no_grad():
        if cfg.direction in ("AtoB", "both"):
            dataset_a = SingleDomainDataset(dataroot / cfg.domain_a, cfg.image_size)
            for index, sample in enumerate(tqdm(dataset_a, desc="夏季 → 冬季 (A→B)")):
                if index >= cfg.num_images:
                    break
                image = sample["image"].unsqueeze(0).to(device)
                translated = model.translate_a_to_b(image)
                save_side_by_side(
                    tensor_to_pil(image[0]),
                    tensor_to_pil(translated[0]),
                    output_dir / "AtoB" / f"{index:04d}.png",
                )

        if cfg.direction in ("BtoA", "both"):
            dataset_b = SingleDomainDataset(dataroot / cfg.domain_b, cfg.image_size)
            for index, sample in enumerate(tqdm(dataset_b, desc="冬季 → 夏季 (B→A)")):
                if index >= cfg.num_images:
                    break
                image = sample["image"].unsqueeze(0).to(device)
                translated = model.translate_b_to_a(image)
                save_side_by_side(
                    tensor_to_pil(image[0]),
                    tensor_to_pil(translated[0]),
                    output_dir / "BtoA" / f"{index:04d}.png",
                )

    print(f"推理结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
