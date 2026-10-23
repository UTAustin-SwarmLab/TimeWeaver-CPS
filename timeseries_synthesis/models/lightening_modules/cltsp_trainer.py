from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple
import lightning as L

from timeseries_synthesis.models.load_models import (
    load_cltsp_model,
)

from timeseries_synthesis.utils.basic_utils import get_cltsp_config


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, config):
        super(SupConLoss, self).__init__()
        self.temperature = config.supervised_contrastive_learning.temperature
        self.contrast_mode = config.supervised_contrastive_learning.contrast_mode
        self.base_temperature = config.supervised_contrastive_learning.base_temperature
        self.device = config.device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature # 
            anchor_count = contrast_count # 2
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = (
            torch.exp(logits) * logits_mask + 1e-8
        )  # for stability, else the loss goes to NaN
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class CLTSPTrainer(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.cltsp_config = get_cltsp_config(config=config)
        self.cltsp_model = load_cltsp_model(config=config)
        self.sup_con_loss = SupConLoss(config=config)
        self.metadata_available = self.cltsp_model.metadata_available

    def forward(self, batch: torch.Tensor) -> Tuple[Dict, torch.Tensor]:
        input_ = self.cltsp_model.prepare_training_input(batch)
        if self.metadata_available:
            timeseries_emb, condition_emb = self.cltsp_model(input_)
            return input_, timeseries_emb, condition_emb
        else:
            timeseries_emb = self.cltsp_model(input_)
            return input_, timeseries_emb

    def calculate_mod_clip_loss(self, timeseries_emb, condition_emb) -> torch.Tensor:
        batch_size = timeseries_emb.shape[0]
        logits = (condition_emb @ timeseries_emb.T) / self.config.training.temperature
        target = torch.arange(batch_size).to(self.config.device)
        texts_loss = torch.nn.functional.cross_entropy(logits, target) 
        timeseries_loss = torch.nn.functional.cross_entropy(logits.T, target)
        loss = (timeseries_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

    def calculate_loss(self, loss_tuple) -> torch.Tensor:
        # print(timeseries_emb.shape, condition_emb.shape)

        if self.metadata_available:
            input, timeseries_emb, condition_emb = loss_tuple
        else:
            input, timeseries_emb = loss_tuple
            condition_emb = None

        if self.config.training.use_clip_loss:
            assert condition_emb is not None, "Condition embeddings not available"
            clip_loss = self.calculate_mod_clip_loss(timeseries_emb, condition_emb)
        else:
            clip_loss = 0

        num_samples_in_batch = int(
            timeseries_emb.shape[0] // self.cltsp_config.num_positive_samples
        )

        if self.config.training.use_timeseries_scl_loss:
            timeseries_emb_split = timeseries_emb.reshape(
                self.cltsp_config.num_positive_samples,
                num_samples_in_batch,
                -1,
            )  # (num_positive_samples, batch_size, embedding_dim)
            # print(timeseries_emb_split[0], timeseries_emb)
            # print(
            #     timeseries_emb_split[0].shape, timeseries_emb[:num_samples_in_batch].shape
            # )
            assert torch.all(
                timeseries_emb[:num_samples_in_batch] == timeseries_emb_split[0]
            )
            timeseries_emb_split_reshaped = torch.einsum(
                "ijk->jik", timeseries_emb_split
            )  # (batch_size, num_positive_samples, embedding_dim)
            assert torch.all(
                timeseries_emb[:num_samples_in_batch]
                == timeseries_emb_split_reshaped[:, 0, :]
            )
            supervised_contrastive_learning_loss_for_timeseries = self.sup_con_loss(
                timeseries_emb_split_reshaped
            )
        else:
            supervised_contrastive_learning_loss_for_timeseries = 0

        if self.config.training.use_condition_scl_loss:
            assert condition_emb is not None, "Condition embeddings not available"
            condition_emb_split = condition_emb.reshape(
                self.cltsp_config.num_positive_samples,
                num_samples_in_batch,
                -1,
            )  # (num_positive_samples, batch_size, embedding_dim)
            assert torch.all(
                condition_emb[:num_samples_in_batch] == condition_emb_split[0]
            )
            condition_emb_split_reshaped = torch.einsum(
                "ijk->jik", condition_emb_split
            )  # (batch_size, num_positive_samples, embedding_dim)
            assert torch.all(
                condition_emb[:num_samples_in_batch]
                == condition_emb_split_reshaped[:, 0, :]
            )
            supervised_contrastive_learning_loss_for_condition = self.sup_con_loss(
                condition_emb_split_reshaped
            )
        else:
            supervised_contrastive_learning_loss_for_condition = 0

        return (
            clip_loss,
            supervised_contrastive_learning_loss_for_timeseries,
            supervised_contrastive_learning_loss_for_condition,
        )
 
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.cltsp_model.parameters(),
            lr=self.config.training.learning_rate,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        loss_tuple = self.forward(batch)
        (
            clip_loss,
            supervised_contrastive_learning_loss_for_timeseries,
            supervised_contrastive_learning_loss_for_condition,
        ) = self.calculate_loss(loss_tuple)

        clip_factor = 1 if self.config.training.use_clip_loss else 0
        condition_scl_factor = 1 if self.config.training.use_condition_scl_loss else 0
        timeseries_scl_factor = 1 if self.config.training.use_timeseries_scl_loss else 0

        loss = (
            (
                timeseries_scl_factor
                * supervised_contrastive_learning_loss_for_timeseries
            )
            + (clip_factor * clip_loss)
            + (
                condition_scl_factor
                * supervised_contrastive_learning_loss_for_condition
            )
        )

        self.log(
            "train_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if self.config.training.use_clip_loss:
            self.log(
                "train_clip_loss",
                clip_loss,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        if self.config.training.use_timeseries_scl_loss:
            self.log(
                "train_scl_loss_ts",
                supervised_contrastive_learning_loss_for_timeseries,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        if self.config.training.use_condition_scl_loss:
            self.log(
                "train_scl_loss_cn",
                supervised_contrastive_learning_loss_for_condition,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        for key, value in batch.items():
            batch[key] = value.to(self.config.device)
        loss_tuple = self.forward(batch)

        (
            clip_loss,
            supervised_contrastive_learning_loss_for_timeseries,
            supervised_contrastive_learning_loss_for_condition,
        ) = self.calculate_loss(loss_tuple)

        clip_factor = 1 if self.config.training.use_clip_loss else 0
        condition_scl_factor = 1 if self.config.training.use_condition_scl_loss else 0
        timeseries_scl_factor = 1 if self.config.training.use_timeseries_scl_loss else 0

        loss = (
            (
                timeseries_scl_factor
                * supervised_contrastive_learning_loss_for_timeseries
            )
            + (clip_factor * clip_loss)
            + (
                condition_scl_factor
                * supervised_contrastive_learning_loss_for_condition
            )
        )

        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if self.config.training.use_clip_loss:
            self.log(
                "val_clip_loss",
                clip_loss,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        if self.config.training.use_timeseries_scl_loss:
            self.log(
                "val_scl_loss_ts",
                supervised_contrastive_learning_loss_for_timeseries,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        if self.config.training.use_condition_scl_loss:
            self.log(
                "val_scl_loss_cn",
                supervised_contrastive_learning_loss_for_condition,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss
