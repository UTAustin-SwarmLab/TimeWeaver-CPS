from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import os
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple
import lightning as L

from timeseries_synthesis.models.load_models import (
    load_timeseries_denoiser,
)
from timeseries_synthesis.utils.basic_utils import get_dataset_config, get_denoiser_config
from timeseries_synthesis.utils.synthesis_utils import synthesis_via_diffusion
from timeseries_synthesis.utils.constrained_synthesis_helper_functions import extract_equality_constraints, obtain_constraint_violation
 

class ConstrainedTimeSeriesDiffusionModelTrainer(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.dataset_config = get_dataset_config(config=config)
        self.denoiser_config = get_denoiser_config(config=config)

        self.denoiser_model = load_timeseries_denoiser(config=config)

    def forward(self, batch: torch.Tensor) -> Tuple[Dict, torch.Tensor]:
        denoiser_input = self.denoiser_model.prepare_training_input(batch)
        noise_est = self.denoiser_model(denoiser_input)
        return denoiser_input, noise_est

    def calculate_loss(self, denoiser_input: Dict, noise_est: torch.Tensor) -> torch.Tensor:
        if self.config.denoiser_name[:3] == "sss":
            denoiser_loss = torch.nn.functional.mse_loss(noise_est, denoiser_input["noise"], reduction="mean")
            return denoiser_loss
        else:
            denoiser_loss = torch.nn.functional.mse_loss(noise_est, denoiser_input["noise"], reduction="sum")
            
        constraints_loss = self.compute_constraints_loss(denoiser_input, noise_est)
        # print(f"denoiser_loss: {denoiser_loss}, constraints_loss: {constraints_loss}")
        return denoiser_loss, constraints_loss
    
    def compute_constraints_loss(self, denoiser_input: Dict, noise_est: torch.Tensor) -> torch.Tensor:
        constraint_violation = obtain_constraint_violation(denoiser_input['noisy_sample'], noise_est, denoiser_input['current_alpha_bar'], denoiser_input['constraints'])
        # print(f"constraint_violation: {constraint_violation}")
        return 0.01 * torch.mean(constraint_violation)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.denoiser_model.parameters(),
            lr=self.config.training.learning_rate,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)
        denoiser_loss, constraints_loss = self.calculate_loss(denoiser_input, noise_est)
        self.log(
            "train_loss",
            denoiser_loss + constraints_loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_denoiser_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_constraints_loss",
            constraints_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return denoiser_loss + constraints_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)
        denoiser_loss, constraints_loss = self.calculate_loss(denoiser_input, noise_est)
        self.log(
            "val_loss",
            denoiser_loss + constraints_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_denoiser_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_constraints_loss",
            constraints_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return denoiser_loss + constraints_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)

        denoiser_loss = torch.nn.functional.mse_loss(noise_est, denoiser_input["noise"], reduction="mean")
        self.log(
            "test_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return denoiser_loss

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()