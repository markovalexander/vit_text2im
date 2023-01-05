from typing import Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import wandb
from src.data import DummyDataset, get_cifar_loaders
from src.data_types import StepType, ViTVQGANOutput
from src.models.vit_vqgan import ViTVQGAN
from src.params import DataLoaderParams, TrainingParams


def cycle(loader):
    while True:
        for data in loader:
            yield data


class Trainer:
    def __init__(
        self,
        model: ViTVQGAN,
        data_args: DataLoaderParams,
        training_args: TrainingParams,
    ) -> None:

        self.data_args = data_args
        self.training_args = training_args

        vae_optimizer = self._build_optimizer(
            model.get_model_params(),
            training_args.learning_rate,
            training_args.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-8,
        )
        discr_optimizer = self._build_optimizer(
            model.get_loss_params(),
            training_args.learning_rate,
            training_args.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-8,
        )

        train_loader, val_loader = get_cifar_loaders(
            data_args.name, data_args.batch_size, data_args.root_path,
        )

        self.accelerator = Accelerator(
            log_with='wandb' if training_args.report_to_wandb else None,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            mixed_precision=training_args.mixed_precision,
            device_placement=True,
        )
        self.model, self.vae_optimizer, self.discr_optimizer = self.accelerator.prepare(
            model, vae_optimizer, discr_optimizer,
        )

        self.train_loader, self.val_loader = self.accelerator.prepare(train_loader, val_loader)
        self.train_loader = cycle(train_loader)
        self.test_images = next(iter(self.val_loader))[0]
        self._prepare_logging()


    def train(self):
        progress_bar = tqdm(
            total=self.training_args.num_train_steps * self.training_args.gradient_accumulation_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        for global_step in range(1, self.training_args.num_train_steps + 1):
            self._train_step(global_step)
            progress_bar.update(1)

            if global_step % self.training_args.save_every == 0:
                self.accelerator.save_state(self.training_args.save_dir)

        if self.training_args.report_to_wandb:
            self.accelerator.end_training()

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def _train_step(self, global_step: int):
        self.model.train()

        with self.accelerator.accumulate(self.model), self.accelerator.autocast():
            model_loss, quantizer_loss = self._model_step(global_step)
            discr_loss = self._discr_step()

            if self.training_args.report_to_wandb and \
                global_step % self.training_args.log_steps == 0:
                self.accelerator.log(
                    {'train_loss': model_loss, 'quantizer_loss': quantizer_loss, 'discr_loss': discr_loss}, step=global_step,
                )

        if global_step % self.training_args.save_every == 0:
            self.accelerator.save_state(self.training_args.save_dir.as_posix())

        if global_step % self.training_args.eval_steps == 0 and self.training_args.report_to_wandb:
            self.model.eval()
            with torch.inference_mode():
                reconstructed = self.model(self.test_images).reconstructed

            reconsrtucted_images_gathered = self.accelerator.gather(reconstructed)
            reconstructed = make_grid(reconsrtucted_images_gathered, nrow=2)

            reconstructed_log = wandb.Image(
                reconstructed.detach().cpu().numpy().transpose((1, 2, 0)), caption='reconstructed',
            )
            self.accelerator.log({'reconstructed image': reconstructed_log})

    def _build_optimizer(
        self,
        params,
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float],
        eps: float,
    ) -> torch.optim.Optimizer:
        if weight_decay < 1e-9:
            return torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)

        else:
            params_with_decay, params_no_decay = [], []
            for p in params:
                if p.ndim < 2:
                    params_no_decay.append(p)
                else:
                    params_with_decay.append(p)

            params = [{'params': params_with_decay}, {'params': params_no_decay, 'weight_decay': 0}]

        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)

    def _build_data_loaders(self, data_params: DataLoaderParams) -> Tuple[DataLoader, DataLoader]:
        if data_params.name.startswith('cifar'):
            train_loader, val_loader = get_cifar_loaders(
                name=data_params.name,
                path=data_params.root_path,
                batch_size=data_params.batch_size,
                n_workers=data_params.num_workers,
            )
        else:
            train_loader = DataLoader(DummyDataset(), shuffle=True, batch_size=data_params.batch_size)
            val_loader = DataLoader(DummyDataset(), batch_size=data_params.batch_size)

        return train_loader, val_loader

    def _prepare_logging(self):
        if not self.training_args.report_to_wandb:
            return

        self.accelerator.init_trackers(
                f'vit_vqgan_{self.data_args.name}',
                {
                    'lr': self.training_args.learning_rate,
                    'training_params': self.training_args.dict(),
                    'data_params': self.data_args.dict(),
                },
            )

        test_images = self.accelerator.gather(self.test_images)
        with self.accelerator.main_process_first():
            input_image = make_grid(test_images, nrow=2)

            input_log = wandb.Image(
                input_image.detach().cpu().numpy().transpose((1, 2, 0)), caption='input image',
            )
            self.accelerator.log({'input image': input_log})

    def _model_step(self, global_step: int):
        model_loss = 0
        quantizer_loss = 0

        for _ in range(self.training_args.gradient_accumulation_steps):
            data = next(cycle(self.train_loader))[0].to(self.accelerator.device)
            step_type = StepType.MODEL
            model_output: ViTVQGANOutput = self.model(
                data, step_type, apply_grad_penalty=not (global_step % self.training_args.gradient_penalty_steps),
            )

            loss = self._fix_ddp_loss(model_output.loss)
            self.accelerator.backward(loss)
            model_loss += loss.detach()
            quantizer_loss += model_output.quantizer_output.loss.detach()

        self.vae_optimizer.step()
        self.vae_optimizer.zero_grad()
        return model_loss.item(), quantizer_loss.item() / self.training_args.gradient_accumulation_steps

    def _discr_step(self):
        discr_loss = 0

        for _ in range(self.training_args.gradient_accumulation_steps):
            data = next(cycle(self.train_loader))[0].to(self.accelerator.device)
            step_type = StepType.DISCRIMINATOR
            model_output: ViTVQGANOutput = self.model(data, step_type)

            loss = self._fix_ddp_loss(model_output.loss)
            self.accelerator.backward(loss)
            discr_loss += loss.detach()

        self.discr_optimizer.step()
        self.discr_optimizer.zero_grad()
        return discr_loss.item()

    def _fix_ddp_loss(self, loss):
        for p in self.model.parameters():
            loss += p.sum() * 0
        return loss
