#!/usr/bin/env python3
import os
import argparse
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from timeseries_synthesis.utils.basic_utils import (
    edict2dict,
    seed_everything,
    import_from,
    get_dataloader_purpose,
    OKBLUE,
    OKYELLOW,
    ENDC,
)
from pmdarima import auto_arima

from timeseries_synthesis.models.cltsp_models.utils import ConditionEncoder
import omegaconf
import re

from statsmodels.tsa.statespace.sarimax import SARIMAX



class MetaToSARIMA(Dataset):
    """
    Dataset wrapping (discrete_conditions, continuous_conditions) -> sarima_params.

    - discrete_conditions: (N, T, D_discrete) expected to be one-hot / categorical features (left unscaled)
    - continuous_conditions: (N, T, D_cont) standardized using train statistics
    - y: (N, 7) SARIMA parameter targets
    """
    def __init__(self, disc: np.ndarray, cont: np.ndarray, y: np.ndarray, ts_start: np.ndarray):

        assert disc.shape[0] == cont.shape[0]
        self.disc = disc.astype(np.float32)
        self.cont = cont.astype(np.float32)
        self.y = []
        for idx in range(len(y)):
            self.y.append(np.array(list(y[idx].values())).astype(np.float32))
        self.y = np.array(self.y)
        self.ts_start = ts_start.astype(np.float32)
        # self.y = np.concatenate([self.ts_start, self.y], axis=1)
        print(self.y.shape)

        # Flatten time dimension and concatenate feature sets
        N, T, D_disc = self.disc.shape
        _, _, D_cont = self.cont.shape
        self.X = np.concatenate(
            [self.disc.reshape(N, T * D_disc), self.cont.reshape(N, T * D_cont)],
            axis=1
        )
        print(f"X shape: {self.X.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        disc_sep = torch.from_numpy(self.disc[idx])
        cont_sep = torch.from_numpy(self.cont[idx])
        ts_start = torch.from_numpy(self.ts_start[idx])
        return x, y, disc_sep, cont_sep, ts_start


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(2048, 1024, 256), out_dim: int = 7, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LitMeta2SARIMA(pl.LightningModule):
    def __init__(self, in_dim: int, out_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4,
                 hidden_dims=(2048, 1024, 256), dropout=0.1, device: str = "cuda", history_length: int = 48):
        super().__init__()
        self.device_str = device
        self.save_hyperparameters()

        # create a dummy cltsp_config and dataset_config
        cltsp_config = omegaconf.OmegaConf.create({
            "d_model": 128,
            "n_heads": 8,
            "d_keys": 8,
            "d_values": 8,
            "d_ff": 2048,
            "dropout": 0.1,
            "activation": "gelu",
            "num_positive_samples": 2,
            "num_encoder_layers": 8,
            "num_compression_layers": 1,
            "discrete_condition_embedding_dim": 128,
        })
        dataset_config = omegaconf.OmegaConf.create({   
            "num_channels": 1,
            "time_series_length": 96,
            "required_time_series_length": 96,
            "num_discrete_labels": 135,
            "num_discrete_conditions": 7,
            "num_continuous_labels": 4,
            "discrete_condition_embedding_dim": 128,
            "latent_dim": 48,
        })

        self.condition_encoder = ConditionEncoder(
            cltsp_config=cltsp_config,
            dataset_config=dataset_config,
            device=self.device_str,
        )

        self.history_encoder = MLP(history_length, hidden_dims=hidden_dims, out_dim=dataset_config.latent_dim, dropout=dropout)
        self.model = MLP(dataset_config.latent_dim * 2, hidden_dims=hidden_dims, out_dim=out_dim, dropout=dropout)
        self.crit = nn.MSELoss(reduction="mean")

    def forward(self, x, disc_sep, cont_sep, ts_start):
        emb = self.condition_encoder(disc_sep, cont_sep)
        history_emb = self.history_encoder(ts_start)
        emb = torch.cat([emb, history_emb], dim=-1)
        return self.model(emb)

    def _shared_step(self, batch, stage: str):
        x, y, disc_sep, cont_sep, ts_start = batch
        y_hat = self(x, disc_sep, cont_sep, ts_start)
        loss = self.crit(y_hat, y)

        # Log per-parameter MSE as well
        with torch.no_grad():
            per_param_mse = ((y_hat - y) ** 2).mean(dim=0)
            for i, v in enumerate(per_param_mse):
                self.log(f"{stage}_mse_param_{i}", v, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_mse", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

def main():
    parser = argparse.ArgumentParser(description="Train MLP: metadata -> SARIMA parameters (traffic dataset)")
    parser.add_argument("--traffic_dataset_dir", type=str, required=True)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--P", type=int, default=1)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--Q", type=int, default=1)
    parser.add_argument("--s", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=str, default="512,256,64",
                        help="Comma-separated hidden dims for the MLP.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(args.seed)

    # ---------- Load arrays ----------
    tdir = args.traffic_dataset_dir

    train_disc = np.load(os.path.join(tdir, "train_discrete_conditions.npy"))   # (N, T, D_disc)
    val_disc   = np.load(os.path.join(tdir, "val_discrete_conditions.npy"))
    test_disc  = np.load(os.path.join(tdir, "test_discrete_conditions.npy"))

    train_cont = np.load(os.path.join(tdir, "train_continuous_conditions.npy")) # (N, T, D_cont)
    val_cont   = np.load(os.path.join(tdir, "val_continuous_conditions.npy"))
    test_cont  = np.load(os.path.join(tdir, "test_continuous_conditions.npy"))

    train_ts = np.load(os.path.join(tdir, "train_timeseries.npy"))
    val_ts = np.load(os.path.join(tdir, "val_timeseries.npy"))
    test_ts = np.load(os.path.join(tdir, "test_timeseries.npy"))

    history_length = 48

    train_ts_start = train_ts[:, 0, :history_length]
    val_ts_start = val_ts[:, 0, :history_length]
    test_ts_start = test_ts[:, 0, :history_length]

    y_train = np.load(os.path.join(tdir, "standard_train_parameters.npy"), allow_pickle=True)  # (N, D)
    y_val   = np.load(os.path.join(tdir, "standard_val_parameters.npy"), allow_pickle=True)
    y_test  = np.load(os.path.join(tdir, "standard_test_parameters.npy"), allow_pickle=True)

    # ---------- Datasets / Dataloaders ----------
    train_ds = MetaToSARIMA(train_disc, train_cont, y_train, train_ts_start)
    val_ds   = MetaToSARIMA(val_disc,   val_cont,   y_val, val_ts_start)
    test_ds  = MetaToSARIMA(test_disc,  test_cont,  y_test, test_ts_start)

    in_dim = train_ds[0][0].numel()  # flattened feature length
    out_dim = train_ds[0][1].numel()
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(","))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    # ---------- Lightning Module ----------
    lit = LitMeta2SARIMA(
        in_dim=in_dim,
        out_dim=out_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        device=device,
        history_length=history_length,
    )

    # ---------- Callbacks ----------
    ckpt_cb = ModelCheckpoint(
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        filename="meta2sarima-{epoch:02d}-{val_mse:.6f}",
        save_last=True,
    )
    early_cb = EarlyStopping(
        monitor="val_mse",
        mode="min",
        patience=args.patience,
        verbose=True
    )

    # ---------- Trainer ----------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=False,
        log_every_n_steps=50,
        callbacks=[ckpt_cb, early_cb],
    )

    # ---------- Fit / Test ----------
    trainer.fit(lit, train_loader, val_loader)
    trainer.test(lit, test_loader, ckpt_path=ckpt_cb.best_model_path)

    # Helpful printouts
    print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")
    print(f"Input feature dimension: {in_dim}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # estimate the parameters for the test set
    test_parameters = []
    lit.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, y, disc_sep, cont_sep, ts_start = batch
            y_hat = lit(x, disc_sep, cont_sep, ts_start)
            test_parameters.append(y_hat.detach().cpu().numpy())
    test_parameters = np.concatenate(test_parameters, axis=0)
    np.save(os.path.join(tdir, "test_standard_parameters_predicted.npy"), test_parameters)
    print("saved the predicted parameters to the test set at: ", os.path.join(tdir, "test_standard_parameters_predicted.npy"))




if __name__ == "__main__":
    main()
