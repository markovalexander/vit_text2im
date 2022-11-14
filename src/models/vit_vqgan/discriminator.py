from typing import List, Tuple

import torch
from einops import rearrange
from lpips import LPIPS
from torch import Tensor, nn

from src.data_types import StepType
from src.models.vit_vqgan.layers import CrossEmbedLayer, ResnetBlock


class Discriminator(nn.Module):
    def __init__(
        self,
        dims: List[int],
        channels: int = 3,
        groups: int = 8,
        cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
        last_kernel_size: int = 1,
    ):
        super().__init__()

        input_dim, *_, final_dim = dims


        layers = [
            nn.Sequential(
                CrossEmbedLayer(
                    channels, kernel_sizes=cross_embed_kernel_sizes, dim_out=input_dim, stride=1,
                ),
                nn.LeakyReLU(0.1)
            )
        ]

        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.GroupNorm(groups, dim_out),
                    ResnetBlock(dim_out, dim_out),
                )
        )

        self.layers = nn.ModuleList(layers)
        self.to_logits = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(final_dim, 1, last_kernel_size)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)


class LossNetwork(nn.Module):
    def __init__(
      self,
      dims: List[int],
      channels: int = 3,
      groups: int = 8,
      cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15),
      codebook_weight: float = 1.0,
      perceptual_weight: float = 1.0,
      use_grad_penalty: bool = False,
      gp_weight: float = 10.0,
      last_discr_kernel_size: int = 1,
      **kwargs,
    ):
        super().__init__()

        self.discriminator = Discriminator(
            dims, channels, groups, cross_embed_kernel_sizes, last_discr_kernel_size,
        )

        self.reconstruction_loss = nn.MSELoss()
        self.perceptual_loss = LPIPS(net="vgg", verbose=False)

        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.use_grad_penalty = use_grad_penalty
        self.gp_weight = gp_weight

    def forward(
      self,
      imgs: Tensor,
      quantizer_loss: Tensor,
      reconstructed_imgs: Tensor,
      step: StepType,
      last_decoder_layer_weights: Tensor,
    ) -> Tensor:

        if step.DISCRIMINATOR:
            reconstructed_imgs.detach_()
            imgs.requires_grad_()

            reconstructed_discr_logits = self.discriminator(reconstructed_imgs)
            real_discr_logits = self.discriminator(imgs)

            discr_loss = self.hinge_loss(real_discr_logits, reconstructed_discr_logits)
            if self.use_grad_penalty:
                loss = self.add_grad_penalty(discr_loss)

        if step.MODEL:
            reconstruction_loss = self.reconstruction_loss(imgs, reconstructed_imgs)
            perceptual_loss = self.perceptual_loss(2 * imgs - 1, 2 * reconstructed_imgs - 1).mean()
            gen_loss = self.gen_loss(self.discriminator(reconstructed_imgs))

            adaptive_weight = self.calculate_adaptive_weight(gen_loss, perceptual_loss, last_decoder_layer_weights)

            loss = reconstruction_loss
            loss = loss + self.perceptual_weight * perceptual_loss
            loss = loss + self.codebook_weight * quantizer_loss
            loss = loss + adaptive_weight * gen_loss

        return loss

    def discr_loss(self, real: Tensor, fake: Tensor) -> Tensor:
        return torch.mean(torch.relu(1 + fake) + torch.relu(1 - real))

    def gen_loss(self, img: Tensor) -> Tensor:
        return torch.mean(-img)

    def calculate_adaptive_weight(
      self,
      generator_loss: Tensor,
      perceptual_loss: Tensor,
      last_decoder_layer_weights: Tensor,
    ) -> Tensor:

        norm_grad_wrt_gen_loss = torch.autograd.grad(
            outputs=generator_loss,
            inputs=last_decoder_layer_weights,
            grad_outputs=torch.ones_like(generator_loss),
            retain_graph=True,
        )[0].detach().norm(p=2)
        norm_grad_wrt_perceptual_loss = torch.autograd.grad(
            outputs=perceptual_loss,
            inputs=last_decoder_layer_weights,
            grad_outputs=torch.ones_like(generator_loss),
            retain_graph=True,
        )[0].detach().norm(p=2)

        adaptive_weight = norm_grad_wrt_perceptual_loss / (norm_grad_wrt_gen_loss + 1e-8)
        adaptive_weight.clamp_(max=1e4)
        return adaptive_weight

    def grad_penalty(
      self,
      images: Tensor,
      logits: Tensor,
      discr_loss: Tensor,
    ) -> Tensor:
        gradients = torch.autograd.grad(
            outputs=logits,
            inputs=images,
            grad_outputs = torch.ones(logits.size(), device = images.device),
            create_graph = True,
            retain_graph = True,
            only_inputs = True,
        )[0]
        gradients = rearrange(gradients, 'b ... -> b (...)')
        gp = self.gp_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return discr_loss + gp
