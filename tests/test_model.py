import torch

from src.models.vit_vqgan import ViT_VQGAN
from src.params import ModelConfig
from src.utils import root_path


def test_model():
    config_path = root_path() / 'configs' / 'vqgan_test.yaml'

    with open(config_path, 'r') as file:
        params = ModelConfig.parse_raw(file.read())

    model = ViT_VQGAN(
        params.encoder_params,
        params.decoder_params,
        params.quantizer_params,
        params.loss_params,
    )
    assert model is not None

    test_tensor = torch.randn(1, 3, params.encoder_params.image_size, params.encoder_params.image_size)
    a = model(test_tensor, step_type=0, global_step=0, batch_idx=0)
    b = model(test_tensor, step_type=1, global_step=1, batch_idx=1)
    c = model(test_tensor, step_type=0, global_step=10_000, batch_idx=10_000)
    d = model(test_tensor, step_type=1, global_step=10_003, batch_idx=10_003)

    all_losses = torch.stack([a, b, c, d], )
    assert not torch.isinf(all_losses).any()
    assert not torch.isnan(all_losses).any()
