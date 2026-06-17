#!/usr/bin/env python3
"""下载 CycleGAN 官方 summer2winter_yosemite 数据集。"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip"
ZIP_PATH = "summer2winter_yosemite.zip"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"已保存到: {dest}")


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    print(f"正在解压到: {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    print("解压完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 summer2winter_yosemite 数据集")
    parser.add_argument("--output-dir", type=str, default="datasets/summer2winter_yosemite")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_path = output_dir.parent / ZIP_PATH

    if (output_dir / "trainA").exists() and (output_dir / "trainB").exists():
        print(f"数据集已存在: {output_dir}")
        return

    download(DATASET_URL, zip_path)
    extract_zip(zip_path, output_dir.parent)
    print(f"数据集就绪: {output_dir}")


if __name__ == "__main__":
    main()
