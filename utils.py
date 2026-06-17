from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 0.5 + 0.5


def save_sample_grid(
    real_a: torch.Tensor,
    fake_b: torch.Tensor,
    rec_a: torch.Tensor,
    real_b: torch.Tensor,
    fake_a: torch.Tensor,
    rec_b: torch.Tensor,
    output_path: str | Path,
) -> None:
    """保存 A→B→A 与 B→A→B 的对比网格图。"""
    rows = [
        denormalize(real_a),
        denormalize(fake_b),
        denormalize(rec_a),
        denormalize(real_b),
        denormalize(fake_a),
        denormalize(rec_b),
    ]
    grid = make_grid(torch.cat(rows, dim=0), nrow=real_a.size(0), padding=2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = denormalize(tensor.detach().cpu()).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def save_checkpoint(model, optimizers, epoch: int, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "G_A": model.G_A.state_dict(),
            "G_B": model.G_B.state_dict(),
            "D_A": model.D_A.state_dict(),
            "D_B": model.D_B.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
        },
        path,
    )


def load_checkpoint(model, optimizers, path: str | Path) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.G_A.load_state_dict(checkpoint["G_A"])
    model.G_B.load_state_dict(checkpoint["G_B"])
    model.D_A.load_state_dict(checkpoint["D_A"])
    model.D_B.load_state_dict(checkpoint["D_B"])
    if optimizers is not None and "optimizers" in checkpoint:
        for opt, state in zip(optimizers, checkpoint["optimizers"]):
            opt.load_state_dict(state)
    return int(checkpoint.get("epoch", 0))
