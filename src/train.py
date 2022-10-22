from pprint import pformat

import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from src.data.dummy import DummyDataset
from src.data.imagenet import get_loaders
from src.data_types import StepType, ViTVQGANOutput
from src.models.vit_vqgan import ViT_VQGAN
from src.params import parse_params_from_config


def fix_ddp_loss(loss, model):
    params_sum = 0
    for p in model.parameters():
        params_sum += torch.sum(p) * 0
    return loss + params_sum


def train(
  config: str = 'vqgan.yaml',
  lr: float = 4.5e-6,
):
    params = parse_params_from_config(config)
    accelerator = Accelerator(
        log_with='wandb' if params.training_params.report_to_wandb else None,
        gradient_accumulation_steps=params.training_params.gradient_accumulation_steps,
        mixed_precision=params.training_params.mixed_precision,
        device_placement=True,
    )

    accelerator.print(pformat(params.dict()))

    model = ViT_VQGAN(params.encoder_params, params.decoder_params, params.quantizer_params, params.loss_params)

    optimizer_model = torch.optim.AdamW(model.get_train_params(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    optimizer_loss = torch.optim.AdamW(model.get_loss_params(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    if params.data_params.name == 'imagenet':
        train_loader, test_loader = get_loaders(
            params.data_params.root_path,
            params.data_params.batch_size,
            params.data_params.num_workers,
        )
    else:
        train_loader = DataLoader(DummyDataset(), shuffle=True, batch_size=params.data_params.batch_size)
        test_loader = DataLoader(DummyDataset(), batch_size=params.data_params.batch_size)

    test_images = next(iter(test_loader))['image']

    train_loader, model, optimizer_model, optimizer_loss = accelerator.prepare(
        train_loader, model, optimizer_model, optimizer_loss,
    )
    progress_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)

    if params.training_params.report_to_wandb:
        accelerator.init_trackers('vit_vqgan', {'config_name': config, 'lr': lr, 'params': params.dict()})

    global_step = 0
    for _ in range(params.training_params.num_epochs):
        model.train()
        total_loss = 0
        for step_idx, batch in enumerate(progress_bar):
            with accelerator.accumulate(model), accelerator.autocast():
                global_step += 1
                step_type = StepType.from_global_step(global_step)

                model_output: ViTVQGANOutput = model(batch['image'], step_type, global_step, step_idx)

                loss = model_output.loss
                loss = fix_ddp_loss(loss, model)

                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if step_type.AUTOENCODER:
                    optimizer_model.step()
                    optimizer_model.zero_grad()
                else:
                    optimizer_loss.step()
                    optimizer_loss.zero_grad()

                if params.training_params.report_to_wandb:
                    accelerator.log(
                        {'train_loss': loss, 'quantizer_loss': model_output.quantizer_loss}, step=global_step,
                    )

        model.eval()
        if params.training_params.report_to_wandb:
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

    if params.training_params.report_to_wandb:
        accelerator.end_training()

if __name__ == "__main__":
    train('vqgan_test.yaml')
