from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .networks import Discriminator, Generator, init_weights


@dataclass
class CycleGANLosses:
    g_total: Tensor
    g_adv: Tensor
    g_cycle: Tensor
    g_identity: Tensor
    d_a: Tensor
    d_b: Tensor


class CycleGAN(nn.Module):
    """CycleGAN 模型：G_A (A→B), G_B (B→A), D_A, D_B。"""

    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        use_identity: bool = True,
    ):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.use_identity = use_identity

        self.G_A = Generator()
        self.G_B = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()

        for net in (self.G_A, self.G_B, self.D_A, self.D_B):
            net.apply(init_weights)

        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

    def set_requires_grad(self, nets, requires_grad: bool) -> None:
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    @staticmethod
    def _gan_target(real_pred: Tensor, is_real: bool) -> Tensor:
        target_val = 1.0 if is_real else 0.0
        return torch.full_like(real_pred, target_val)

    def _gan_loss(self, pred: Tensor, is_real: bool) -> Tensor:
        target = self._gan_target(pred, is_real)
        return self.criterion_gan(pred, target)

    def forward_generators(self, real_a: Tensor, real_b: Tensor) -> CycleGANLosses:
        fake_b = self.G_A(real_a)
        fake_a = self.G_B(real_b)

        rec_a = self.G_B(fake_b)
        rec_b = self.G_A(fake_a)

        loss_g_adv = (
            self._gan_loss(self.D_B(fake_b), True) + self._gan_loss(self.D_A(fake_a), True)
        ) * 0.5

        loss_g_cycle = (
            self.criterion_cycle(rec_a, real_a) + self.criterion_cycle(rec_b, real_b)
        ) * self.lambda_cycle

        loss_g_identity = torch.tensor(0.0, device=real_a.device)
        if self.use_identity and self.lambda_identity > 0:
            idt_b = self.G_A(real_b)
            idt_a = self.G_B(real_a)
            loss_g_identity = (
                self.criterion_identity(idt_b, real_b) + self.criterion_identity(idt_a, real_a)
            ) * self.lambda_identity

        loss_g_total = loss_g_adv + loss_g_cycle + loss_g_identity

        return CycleGANLosses(
            g_total=loss_g_total,
            g_adv=loss_g_adv,
            g_cycle=loss_g_cycle,
            g_identity=loss_g_identity,
            d_a=torch.tensor(0.0, device=real_a.device),
            d_b=torch.tensor(0.0, device=real_a.device),
        )

    def forward_discriminators(self, real_a: Tensor, real_b: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            fake_b = self.G_A(real_a)
            fake_a = self.G_B(real_b)

        loss_d_a = 0.5 * (
            self._gan_loss(self.D_A(real_a), True)
            + self._gan_loss(self.D_A(fake_a.detach()), False)
        )
        loss_d_b = 0.5 * (
            self._gan_loss(self.D_B(real_b), True)
            + self._gan_loss(self.D_B(fake_b.detach()), False)
        )
        return loss_d_a, loss_d_b

    def translate_a_to_b(self, image_a: Tensor) -> Tensor:
        return self.G_A(image_a)

    def translate_b_to_a(self, image_b: Tensor) -> Tensor:
        return self.G_B(image_b)

    def generator_parameters(self):
        return itertools.chain(self.G_A.parameters(), self.G_B.parameters())

    def discriminator_parameters(self):
        return itertools.chain(self.D_A.parameters(), self.D_B.parameters())
