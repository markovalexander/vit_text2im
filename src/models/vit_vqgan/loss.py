# ------------------------------------------------------------------------------------------------------------
# Copied from enhancing-transformers by Thuan H. Nguyen (https://github.com/thuanz123/enhancing-transformers)
# ------------------------------------------------------------------------------------------------------------
from typing import Optional

import lpips
import torch
from torch import Tensor, nn

from src.data_types import StepType
from src.models.vit_vqgan.losses.op import conv2d_gradfix
from src.models.vit_vqgan.losses.style_discriminator import (
    StyleDiscriminator,
    hinge_d_loss,
    least_square_d_loss,
    vanilla_d_loss,
)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
      self,
      disc_start: int = 0,
      disc_loss: str = 'vanilla',
      codebook_weight: float = 1.0,
      loglaplace_weight: float = 1.0,
      loggaussian_weight: float = 1.0,
      perceptual_weight: float = 1.0,
      adversarial_weight: float = 1.0,
      use_adaptive_adv: bool = False,
      r1_gamma: float = 10,
      do_r1_every: int = 16,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "least_square"], f"Unknown GAN loss '{disc_loss}'."
        self.perceptual_loss = lpips.LPIPS(net="vgg", verbose=False)

        self.codebook_weight = codebook_weight
        self.loglaplace_weight = loglaplace_weight
        self.loggaussian_weight = loggaussian_weight
        self.perceptual_weight = perceptual_weight

        self.discriminator = StyleDiscriminator()
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "least_square":
            self.disc_loss = least_square_d_loss

        self.adversarial_weight = adversarial_weight
        self.use_adaptive_adv = use_adaptive_adv
        self.r1_gamma = r1_gamma
        self.do_r1_every = do_r1_every

    def calculate_adaptive_factor(self, nll_loss: torch.FloatTensor,
                                  g_loss: torch.FloatTensor, last_layer: nn.Module) -> torch.FloatTensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        adapt_factor = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        adapt_factor = adapt_factor.clamp(0.0, 1e4).detach()

        return adapt_factor

    def forward(
        self,
        codebook_loss: torch.FloatTensor,
        inputs: torch.FloatTensor,
        reconstructions: torch.FloatTensor,
        step_type: StepType,
        global_step: int,
        batch_idx: int,
        last_layer: Optional[nn.Module] = None,
    ) -> Tensor:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # now the GAN part
        if step_type == StepType.AUTOENCODER:
            # generator update
            loglaplace_loss = (reconstructions - inputs).abs().mean() * self.loglaplace_weight
            loggaussian_loss = (reconstructions - inputs).pow(2).mean() * self.loggaussian_weight
            perceptual_loss = self.perceptual_loss(2 * inputs - 1, 2 * reconstructions - 1).mean()

            nll_loss = loglaplace_loss + loggaussian_loss + perceptual_loss * self.perceptual_weight

            logits_fake = self.discriminator(reconstructions)
            g_loss = self.disc_loss(logits_fake)

            try:
                d_weight = self.adversarial_weight

                if self.use_adaptive_adv:
                    d_weight *= self.calculate_adaptive_factor(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = 1 if global_step >= self.discriminator_iter_start else 0
            loss = nll_loss + disc_factor * d_weight * g_loss + self.codebook_weight * codebook_loss

            return loss

        if step_type == StepType.DISCRIMINATOR:
            # second pass for discriminator update
            disc_factor = 1 if global_step >= self.discriminator_iter_start else 0
            do_r1 = self.training and bool(disc_factor) and batch_idx % self.do_r1_every == 0

            logits_real = self.discriminator(inputs.requires_grad_(do_r1))
            logits_fake = self.discriminator(reconstructions.detach())

            d_loss = disc_factor * self.disc_loss(logits_fake, logits_real)
            if do_r1:
                with conv2d_gradfix.no_weight_gradients():
                    gradients, = torch.autograd.grad(outputs=logits_real.sum(), inputs=inputs, create_graph=True)

                gradients_norm = gradients.square().sum([1,2,3]).mean()
                d_loss += self.r1_gamma * self.do_r1_every * gradients_norm / 2

            return d_loss
