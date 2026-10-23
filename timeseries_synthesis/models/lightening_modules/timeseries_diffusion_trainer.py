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
 

class TimeSeriesDiffusionModelTrainer(L.LightningModule):
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
            return denoiser_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.denoiser_model.parameters(),
            lr=self.config.training.learning_rate,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)
        denoiser_loss = self.calculate_loss(denoiser_input, noise_est)
        self.log(
            "train_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return denoiser_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        denoiser_input, noise_est = self.forward(batch)
        denoiser_loss = self.calculate_loss(denoiser_input, noise_est)
        self.log(
            "val_loss",
            denoiser_loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return denoiser_loss

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