import torch
import typer
import wandb
from accelerate import Accelerator
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from src.data.imagenet import get_loaders
from src.data_types import StepType, ViTVQGANOutput
from src.models.vit_vqgan import Loss, ViT_VQGAN
from src.params import parse_params_from_config

app = typer.Typer()

@app.command('train')
def train(
  config: str = typer.Argument('vqgan.yaml', help="Name of .yaml file in configs folder"),
  lr: float = typer.Argument(4.5e-6, help="Base learning rate"),
):
    params = parse_params_from_config(config)
    accelerator = Accelerator(
        log_with=params.training_params.report_to,
        gradient_accumulation_steps=params.training_params.gradient_accumulation_steps,
        mixed_precision=params.training_params.mixed_precision,
        device_placement=True,
    )

    model = ViT_VQGAN(params.encoder_params, params.decoder_params, params.quantizer_params)
    loss = Loss(params.loss_params)

    optimizer_model = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    optimizer_loss = torch.optim.AdamW(loss.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    train_loader, test_loader = get_loaders(
        params.data_params.root_path, params.data_params.batch_size, params.data_params.num_workers,
    )
    test_images = next(iter(test_loader))['image']

    train_loader, model, optimizer_model, loss, optimizer_loss = accelerator.prepare(
        train_loader, model, optimizer_model, loss, optimizer_loss,
    )
    progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)

    accelerator.init_trackers('vit_vqgan', {'config_name': config, 'lr': lr, 'params': params.dict()})

    global_step = 0
    for epoch in range(params.training_params.num_epochs):
        model.train()
        loss.train()
        total_loss = 0
        for step_idx, batch in enumerate(progress_bar):
            with accelerator.accumulate(model), accelerator.autocast():
                global_step += 1
                step_type = StepType.from_global_step(global_step)

                model_output: ViTVQGANOutput = model(batch['image'])
                loss_value = loss(
                    model_output.quantizer_loss,
                    batch['image'],
                    model_output.reconstructed,
                    step_type,
                    global_step,
                    step_idx,
                    model.get_last_layer(),
                )
                total_loss += loss_value.detach().float()
                accelerator.backward(loss_value)

                if step_type.AUTOENCODER:
                    optimizer_model.step()
                    optimizer_model.zero_grad()
                else:
                    optimizer_loss.step()
                    optimizer_loss.zero_grad()
                accelerator.log({'train_loss': loss_value}, step=global_step)

        model.eval()
        with accelerator.main_process_first():
            output = model(test_images)
            reconstructed = output.reconstructed

            input_image = make_grid(test_images, nrow=2)
            reconstructed = make_grid(reconstructed, nrow=2)

            input_log = wandb.Image(
                input_image.detach().cpu().numpy().transpose((1, 2, 0)), caption='input image',
            )
            reconstructed_log = wandb.Image(
                reconstructed.detach().cpu().numpy().transpose((1, 2, 0)), caption='reconstructed',
            )
        accelerator.log({'input image': input_log, 'reconstructed image': reconstructed_log})

    accelerator.end_training()
