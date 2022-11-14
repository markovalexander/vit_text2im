from pprint import pformat

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import wandb
from src.data.cifar import get_cifar_loaders
from src.data.dummy import DummyDataset
from src.data_types import StepType, ViTVQGANOutput
from src.models.vit_vqgan import ViTVQGAN
from src.params import parse_params_from_config


def train(
  config: str = 'cifar10.yaml',
  lr: float = 3e-4,
):
    params = parse_params_from_config(config)
    accelerator = Accelerator(
        log_with='wandb' if params.training_params.report_to_wandb else None,
        gradient_accumulation_steps=params.training_params.gradient_accumulation_steps,
        mixed_precision=params.training_params.mixed_precision,
        device_placement=True,
    )

    accelerator.print(pformat(params.dict()))

    model = ViTVQGAN(params.vit_params, params.quantizer_params, params.loss_params)

    optimizer_model = torch.optim.AdamW(model.get_model_params(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    optimizer_loss = torch.optim.AdamW(model.get_loss_params(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    if params.data_params.name.startswith('cifar'):
        train_loader, test_loader = get_cifar_loaders(
            name=params.data_params.name,
            path=params.data_params.root_path,
            batch_size=params.data_params.batch_size,
            n_workers=params.data_params.num_workers,
        )
    else:
        train_loader = DataLoader(DummyDataset(), shuffle=True, batch_size=params.data_params.batch_size)
        test_loader = DataLoader(DummyDataset(), batch_size=params.data_params.batch_size)


    train_loader, test_loader, model, optimizer_model, optimizer_loss = accelerator.prepare(
        train_loader, test_loader, model, optimizer_model, optimizer_loss,
    )

    test_images = next(iter(test_loader))[0]

    if params.training_params.report_to_wandb:
        accelerator.init_trackers(
            f'vit_vqgan_{params.data_params.name}',
            {'config_name': config, 'lr': lr, 'params': params.dict()},
        )

    test_images = accelerator.gather(test_images)
    with accelerator.main_process_first():
        input_image = make_grid(test_images, nrow=2)

        input_log = wandb.Image(
            input_image.detach().cpu().numpy().transpose((1, 2, 0)), caption='input image',
        )
        accelerator.log({'input image': input_log})

    global_step = 0
    num_epochs = params.training_params.num_epochs
    progress_bar = tqdm(total=len(train_loader) * num_epochs, disable=not accelerator.is_local_main_process)

    for _ in range(num_epochs):
        total_loss = 0
        for _, batch in enumerate(train_loader):
            model.train()
            with accelerator.accumulate(model), accelerator.autocast():
                images = batch[0]
                global_step += 1
                step_type = StepType.from_global_step(global_step)

                model_output: ViTVQGANOutput = model(images, step_type)

                loss = model_output.loss

                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if step_type.MODEL:
                    optimizer_model.step()
                    optimizer_model.zero_grad()
                else:
                    optimizer_loss.step()
                    optimizer_loss.zero_grad()

                if params.training_params.report_to_wandb and \
                    global_step % params.training_params.log_steps == 0:
                    accelerator.log(
                        {'train_loss': loss, 'quantizer_loss': model_output.quantizer_output.loss}, step=global_step,
                    )
            progress_bar.update(1)
            if global_step % params.training_params.save_every == 0:
                accelerator.save_state(params.training_params.save_dir.as_posix())

            if global_step % params.training_params.eval_steps == 0 and params.training_params.report_to_wandb:
                model.eval()
                with torch.inference_mode():
                    reconstructed = model(test_images).reconstructed

                reconsrtucted_images_gathered = accelerator.gather(reconstructed)
                reconstructed = make_grid(reconsrtucted_images_gathered, nrow=2)

                reconstructed_log = wandb.Image(
                    reconstructed.detach().cpu().numpy().transpose((1, 2, 0)), caption='reconstructed',
                )
                accelerator.log({'reconstructed image': reconstructed_log})

    if params.training_params.report_to_wandb:
        accelerator.end_training()

if __name__ == "__main__":
    train('cifar10.yaml')
