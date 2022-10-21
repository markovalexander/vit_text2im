import torch
import typer
from accelerate import Accelerator

from src.data.imagenet import get_loaders
from src.models.vit_vqgan import Loss, ViT_VQGAN
from src.params import parse_params_from_config

app = typer.Typer()

@app.command('train')
def train(
  config: str = typer.Argument('vqgan.yaml', help="Name of .yaml file in configs folder"),
  lr: float = typer.Argument(4.5e-6, help="Base learning rate"),
):
    params = parse_params_from_config(config)
    model = ViT_VQGAN(params.encoder_params, params.decoder_params, params.quantizer_params)
    loss = Loss(params.loss_params)

    optimizer_model = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    optimizer_loss = torch.optim.AdamW(loss.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    train_loader, test_loader = get_loaders(
        params.data_params.root_path, params.data_params.batch_size, params.data_params.num_workers,
    )

    accelerator = Accelerator()
    train_loader, model, optimizer_model, loss, optimizer_loss = accelerator.prepare(
        train_loader, test_loader, model, optimizer_model, loss, optimizer_loss,
    )

