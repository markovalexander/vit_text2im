from pprint import pformat

import click

from src.models.vit_vqgan import ViTVQGAN
from src.params import parse_params_from_config
from src.trainer import Trainer


@click.command()
@click.option('--config', help='config file name', default='cifar10.yaml')
def train(config: str):
    params = parse_params_from_config(config)
    model = ViTVQGAN(params.vit_params, params.quantizer_params, params.loss_params)

    trainer = Trainer(model, params.data_params, params.training_params)
    trainer.print(pformat(params.dict()))

    trainer.train()

if __name__ == "__main__":
    train()
