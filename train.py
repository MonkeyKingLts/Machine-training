#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import parse_train_args
from datasets import UnpairedImageDataset
from models import CycleGAN
from utils import load_checkpoint, save_checkpoint, save_sample_grid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    cfg = parse_train_args()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset = UnpairedImageDataset(
        root=cfg.dataroot,
        domain_a=cfg.domain_a,
        domain_b=cfg.domain_b,
        image_size=cfg.image_size,
        is_train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = CycleGAN(
        lambda_cycle=cfg.lambda_cycle,
        lambda_identity=cfg.lambda_identity,
        use_identity=cfg.use_identity,
    ).to(device)

    optimizer_g = torch.optim.Adam(model.generator_parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optimizer_d = torch.optim.Adam(model.discriminator_parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optimizers = [optimizer_g, optimizer_d]

    start_epoch = 0
    if cfg.resume:
        start_epoch = load_checkpoint(model, optimizers, cfg.resume)
        print(f"从 epoch {start_epoch} 恢复训练")

    exp_dir = Path(cfg.checkpoint_dir) / cfg.name
    sample_dir = Path(cfg.sample_dir) / cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_g, epoch_d_a, epoch_d_b = 0.0, 0.0, 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch in progress:
            real_a = batch["A"].to(device)
            real_b = batch["B"].to(device)

            model.set_requires_grad([model.D_A, model.D_B], True)
            optimizer_d.zero_grad(set_to_none=True)
            loss_d_a, loss_d_b = model.forward_discriminators(real_a, real_b)
            (loss_d_a + loss_d_b).backward()
            optimizer_d.step()

            model.set_requires_grad([model.D_A, model.D_B], False)
            optimizer_g.zero_grad(set_to_none=True)
            losses = model.forward_generators(real_a, real_b)
            losses.g_total.backward()
            optimizer_g.step()

            epoch_g += losses.g_total.item()
            epoch_d_a += loss_d_a.item()
            epoch_d_b += loss_d_b.item()
            progress.set_postfix(
                G=f"{losses.g_total.item():.3f}",
                D_A=f"{loss_d_a.item():.3f}",
                D_B=f"{loss_d_b.item():.3f}",
            )

        n_batches = len(loader)
        print(
            f"[Epoch {epoch + 1}] "
            f"G={epoch_g / n_batches:.4f} "
            f"D_A={epoch_d_a / n_batches:.4f} "
            f"D_B={epoch_d_b / n_batches:.4f}"
        )

        if (epoch + 1) % cfg.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                sample = next(iter(loader))
                real_a = sample["A"].to(device)
                real_b = sample["B"].to(device)
                fake_b = model.translate_a_to_b(real_a)
                fake_a = model.translate_b_to_a(real_b)
                rec_a = model.translate_b_to_a(fake_b)
                rec_b = model.translate_a_to_b(fake_a)
                save_sample_grid(
                    real_a,
                    fake_b,
                    rec_a,
                    real_b,
                    fake_a,
                    rec_b,
                    sample_dir / f"epoch_{epoch + 1:04d}.png",
                )

        if (epoch + 1) % cfg.save_interval == 0 or epoch + 1 == cfg.epochs:
            save_checkpoint(model, optimizers, epoch + 1, exp_dir / f"epoch_{epoch + 1:04d}.pth")

    print(f"训练完成，检查点保存在: {exp_dir}")


if __name__ == "__main__":
    main()
