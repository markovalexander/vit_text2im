from typing import Optional

from pydantic_yaml import YamlModel


class ModelSettings(YamlModel):
    image_size: Optional[int] = None
    patch_size: Optional[int] = None
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    mlp_dim: Optional[int] = None
    channels: Optional[int] = 3


class VectorQuantizerSettings(YamlModel):
    codebook_dim: int
    codebook_size: int
    use_norm: bool = True
    use_straight_through: bool = True
    beta: float = 0.25


class LossSettings(YamlModel):
    disc_start: int = 0
    disc_loss: str = 'vanilla'
    codebook_weight: float = 1.0
    loglaplace_weight: float = 1.0
    loggaussian_weight: float = 1.0
    perceptual_weight: float = 1.0
    adversarial_weight: float = 1.0
    use_adaptive_adv: bool = False
    r1_gamma: float = 10
    do_r1_every: int = 16
    stylegan_size: int = 256


class ModelConfig(YamlModel):
    encoder_params: ModelSettings
    decoder_params: ModelSettings
    quantizer_params: VectorQuantizerSettings
    loss_params: LossSettings

    def __post_init__(self):
        assert self.encoder_params.image_size == self.decoder_params.image_size, \
            "Encoder end Decoder must work with the same image"
        assert self.encoder_params.image_size == self.loss_params.stylegan_size, \
            "StyleDiscriminator image size is not equal to Encoder's"
