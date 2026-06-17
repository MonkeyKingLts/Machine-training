from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size: int, is_train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)]
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


class UnpairedImageDataset(Dataset):
    """
    无配对双域图像数据集。

    目录结构::
        root/
          trainA/   # 域 A，例如夏季风景
          trainB/   # 域 B，例如冬季风景
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str | Path,
        domain_a: str = "trainA",
        domain_b: str = "trainB",
        image_size: int = 256,
        is_train: bool = True,
    ):
        self.root = Path(root)
        self.paths_a = self._collect_images(self.root / domain_a)
        self.paths_b = self._collect_images(self.root / domain_b)
        if not self.paths_a:
            raise FileNotFoundError(f"域 A 目录为空或不存在: {self.root / domain_a}")
        if not self.paths_b:
            raise FileNotFoundError(f"域 B 目录为空或不存在: {self.root / domain_b}")

        self.transform = build_transform(image_size, is_train)

    @classmethod
    def _collect_images(cls, folder: Path) -> list[Path]:
        if not folder.exists():
            return []
        return sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        )

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path_a = self.paths_a[index % len(self.paths_a)]
        path_b = self.paths_b[random.randint(0, len(self.paths_b) - 1)]
        image_a = self.transform(Image.open(path_a).convert("RGB"))
        image_b = self.transform(Image.open(path_b).convert("RGB"))
        return {"A": image_a, "B": image_b, "path_a": str(path_a), "path_b": str(path_b)}
