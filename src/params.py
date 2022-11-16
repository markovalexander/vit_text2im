from pathlib import Path
from typing import Optional, Tuple

from pydantic_yaml import YamlModel

from src import root_path


class ModelSettings(YamlModel):
    hidden_dim: int
    input_channels: int = 3
    image_size: int = 32
    num_layers: int = 4
    patch_size: int= 16
    heads_channels: int = 512
    num_heads: int = 8
    fc_multiplier: int = 4


class ViTSettings(YamlModel):
    encoder: ModelSettings
    decoder: ModelSettings


class VectorQuantizerSettings(YamlModel):
    input_dim: int
    codebook_dim: Optional[int]
    codebook_size: int
    use_norm: bool = True
    use_straight_through: bool = True
    encode_images: bool = True
    beta: float = 0.25

    def __post_init__(self):
        self.codebook_dim = self.codebook_dim or self.input_dim

class LossSettings(YamlModel):
    discr_layers: int = 4
    channels: int = 3
    groups: int = 8
    cross_embed_kernel_sizes: Tuple[int] = (3, 7, 15)
    codebook_weight: float = 1.0
    perceptual_weight: float = 1.0
    use_grad_penalty: bool = False
    gp_weight: float = 10.0


class DataLoaderParams(YamlModel):
    root_path: Path
    name: str
    batch_size: int
    num_workers: int = 8

    def __post__init__(self):
        self.name = self.name.lower()

class TrainingParams(YamlModel):
    num_epochs: int = 1
    report_to_wandb: bool = False
    gradient_accumulation_steps: int = 1
    mixed_precision: str = 'no'
    log_steps: int = 100
    eval_steps: int = 500
    save_every: int = 5000
    save_dir: Path = 'checkpoints'

class ModelConfig(YamlModel):
    vit_params: ViTSettings
    quantizer_params: VectorQuantizerSettings
    loss_params: LossSettings
    training_params: TrainingParams = TrainingParams()
    data_params: DataLoaderParams

def parse_params_from_config(config_name: str) -> ModelConfig:
    config_path = root_path() / 'configs' / config_name

    with open(config_path, 'r') as file:
        config = ModelConfig.parse_raw(file.read())

    return config
