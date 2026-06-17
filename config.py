from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataroot: str
    name: str = "summer2winter"
    domain_a: str = "trainA"
    domain_b: str = "trainB"
    image_size: int = 256
    batch_size: int = 1
    num_workers: int = 2
    epochs: int = 200
    lr: float = 0.0002
    beta1: float = 0.5
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    use_identity: bool = True
    save_interval: int = 10
    sample_interval: int = 1
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"
    resume: str | None = None
    seed: int = 42


def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练 CycleGAN 进行双向风格转换")
    parser.add_argument("--dataroot", type=str, required=True, help="数据集根目录")
    parser.add_argument("--name", type=str, default="summer2winter", help="实验名称")
    parser.add_argument("--domain-a", type=str, default="trainA", help="域 A 子目录名")
    parser.add_argument("--domain-b", type=str, default="trainB", help="域 B 子目录名")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=0.5)
    parser.add_argument("--no-identity", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(
        dataroot=args.dataroot,
        name=args.name,
        domain_a=args.domain_a,
        domain_b=args.domain_b,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        use_identity=not args.no_identity,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        checkpoint_dir=args.checkpoint_dir,
        sample_dir=args.sample_dir,
        resume=args.resume,
        seed=args.seed,
    )


@dataclass
class TestConfig:
    dataroot: str
    checkpoint: str
    direction: str = "AtoB"
    domain_a: str = "testA"
    domain_b: str = "testB"
    image_size: int = 256
    output_dir: str = "results"
    num_images: int = 10


def parse_test_args() -> TestConfig:
    parser = argparse.ArgumentParser(description="使用 CycleGAN 进行风格转换推理")
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--direction",
        type=str,
        default="AtoB",
        choices=["AtoB", "BtoA", "both"],
        help="AtoB: 夏季→冬季, BtoA: 冬季→夏季, both: 双向",
    )
    parser.add_argument("--domain-a", type=str, default="testA")
    parser.add_argument("--domain-b", type=str, default="testB")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--num-images", type=int, default=10)
    args = parser.parse_args()
    return TestConfig(
        dataroot=args.dataroot,
        checkpoint=args.checkpoint,
        direction=args.direction,
        domain_a=args.domain_a,
        domain_b=args.domain_b,
        image_size=args.image_size,
        output_dir=args.output_dir,
        num_images=args.num_images,
    )
