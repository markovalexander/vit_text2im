import pytest
import torch

from src.data_types import QuantizerOutput, StepType
from src.models.vit_vqgan import ViTVQGAN
from src.models.vit_vqgan.discriminator import Discriminator
from src.models.vit_vqgan.quantizer import VectorQuantizer
from src.params import parse_params_from_config


@pytest.fixture(scope='module')
def discriminator():
    return Discriminator([32, 64, 128])


@pytest.fixture(scope='module')
def quantizer():
    return VectorQuantizer(16, 128)

# @pytest.mark.skip()
@torch.inference_mode()
def test_discriminator(discriminator):
    image_batch = torch.randn(4, 3, 32, 32)

    with torch.inference_mode():
        output = discriminator(image_batch)

    assert torch.isfinite(output).all()

    output_size = output.size()

    assert len(output_size) == 4
    assert output_size[0] == 4  # keeps batch
    assert output_size[1] == 1  # logit for every pixel


@torch.inference_mode()
# @pytest.mark.skip()
def test_quantizer(quantizer):
    image_batch = torch.randn(4, 3, 4, 4)

    output: QuantizerOutput = quantizer(image_batch)

    # codebook vectors check
    print(output.codebook_vectors.size())
    print(output.codebook_indices.size())

    assert len(output.codebook_vectors.size()) == 4
    assert output.codebook_vectors.size(0) == 4
    assert output.codebook_vectors.size(1) == 3
    assert torch.isfinite(output.codebook_vectors).all()

    # quantizer loss check
    assert output.loss.size() == torch.Size([])
    assert torch.isfinite(output.loss).all()

    # indices check
    indices = output.codebook_indices.view(12, -1)
    codebook = quantizer.norm_layer(quantizer.embedding.weight)

    for ind, vec in zip(indices, output.codebook_vectors.view(12, -1)):
        assert torch.isclose(vec, codebook[ind], atol=1e-5).all()


def test_vit_vqgan_from_params():
    params = parse_params_from_config('cifar10.yaml')
    model = ViTVQGAN(params.vit_params, params.quantizer_params, params.loss_params)
    assert model is not None

    image_batch = torch.randn(4, 3, 32, 32)

    output = model(image_batch, StepType.DISCRIMINATOR)
    assert output is not None
    assert torch.isfinite(output.loss).all()
    assert output.loss.size() == torch.Size([])

    output = model(image_batch, StepType.MODEL)
    assert output is not None
    assert torch.isfinite(output.loss).all()
    assert output.loss.size() == torch.Size([])
