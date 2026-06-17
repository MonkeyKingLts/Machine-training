#!/usr/bin/env python3
"""生成简易合成演示数据，便于在无真实数据集时快速验证流程。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_landscape(seed: int, season: str, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)

    if season == "summer":
        sky = (120, 190, 255)
        ground = (60, 150, 45)
        accent = (255, 210, 80)
        snow = False
    else:
        sky = (170, 195, 220)
        ground = (210, 220, 230)
        accent = (120, 140, 160)
        snow = True

    draw.rectangle([0, 0, size, size // 2], fill=sky)
    draw.rectangle([0, size // 2, size, size], fill=ground)

    mountain_y = size // 2 + rng.integers(-10, 20)
    points = [(0, size)]
    for x in range(0, size + 1, size // 8):
        y = mountain_y - rng.integers(20, 80)
        points.append((x, y))
    points.append((size, size))
    draw.polygon(points, fill=accent)

    tree_count = rng.integers(4, 8)
    for _ in range(tree_count):
        x = int(rng.integers(20, size - 20))
        h = int(rng.integers(25, 55))
        trunk_color = (90, 60, 30) if not snow else (80, 70, 60)
        foliage = (30, 110, 40) if not snow else (230, 235, 240)
        draw.rectangle([x - 3, size // 2 + 40 - h, x + 3, size // 2 + 40], fill=trunk_color)
        draw.ellipse([x - 14, size // 2 + 20 - h, x + 14, size // 2 + 35], fill=foliage)

    if snow:
        for _ in range(120):
            x = int(rng.integers(0, size))
            y = int(rng.integers(0, size))
            draw.rectangle([x, y, x + 1, y + 1], fill=(255, 255, 255))

    image = image.filter(ImageFilter.GaussianBlur(radius=0.3))
    return image


def generate(output_dir: Path, count: int, size: int) -> None:
    for split, prefix in [("train", "train"), ("test", "test")]:
        for domain, season in [("A", "summer"), ("B", "winter")]:
            folder = output_dir / f"{prefix}{domain}"
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count if split == "train" else max(4, count // 4)):
                image = make_landscape(seed=index + (0 if season == "summer" else 1000), season=season, size=size)
                image.save(folder / f"{index:04d}.png")
    print(f"已生成演示数据: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CycleGAN 演示数据集")
    parser.add_argument("--output-dir", type=str, default="datasets/demo_summer_winter")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    generate(Path(args.output_dir), args.count, args.size)


if __name__ == "__main__":
    main()
