import os
import uuid
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra
import logging
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger

from timeseries_synthesis.utils.basic_utils import (
    edict2dict,
    seed_everything,
    import_from,
    get_dataloader_purpose,
    OKBLUE,
    OKYELLOW,
    ENDC,
)

global SHARED_UUID  # pylint: disable=global-statement
SHARED_UUID = None


@hydra.main(config_path="../../configs/", version_base="1.1")
def main(config: DictConfig):
    seed_everything(config.seed)

    mlf_logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        run_name=config.run_name,
        tracking_uri="file://" + config.mlflow_folder,
    )
    pl_dataloader = import_from(
        f"timeseries_synthesis.datasets.lightening_dataloaders.{config.dataloader_file}",
        f"{config.dataloader_model}",
    )(config)
    get_dataloader_purpose(pl_dataloader)

    print(OKYELLOW + "initializing the model to be trained" + ENDC)
    print(OKYELLOW + "model name is - " + config.model_name + ENDC)
    model_type = import_from(
        f"timeseries_synthesis.models.lightening_modules.{config.model_file}_trainer",
        config.model_name,
    )
    pl_model = model_type(config)
    print(OKYELLOW + "model initialized" + ENDC)

    log_dir = os.path.join(
        config.base_path,
        config.save_path,
        config.dataset_name,
        config.run_name,
        mlf_logger.version if mlf_logger.version else "other_gpu",
    )
    pl_model.log_dir = log_dir
    print(OKYELLOW + "the logging directory for this experiment is - " + log_dir + ENDC)

    save_path = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if config.should_compile_torch:
        pl_model = torch.compile(
            pl_model
        )  # compiles the model and *step (training/validation/prediction)
        torch._dynamo.config.log_level = logging.ERROR

    if (
        "gan" in config.model_file
        or "timeweaver" in config.model_file
        or config.store_intermediate_checkpoints
    ):
        print(
            OKBLUE + "setting up the checkpoint callback, will save frequently" + ENDC
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path,
            filename="model_checkpoint_epoch_{epoch:05d}",
            save_top_k=-1,  # Save only the best checkpoint based on a metric (optional)
            every_n_epochs=config.training.save_after_every_n_epochs,  # Save every n epochs
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path,
            monitor=config.save_key,
            filename="best_model",
            save_top_k=1,
            mode="min",
        )

    L.seed_everything(config.seed)

    print(
        OKBLUE + "loading pretrained weights for the model to be trained, if any" + ENDC
    )
    kwargs = {}
    if config.training.strategy != "None":
        kwargs["strategy"] = config.training.strategy
    if hasattr(config, "model_checkpoint_path"):
        if config.model_checkpoint_path != "":
            print(OKBLUE + "loading model from checkpoint" + ENDC)
            pl_model = model_type.load_from_checkpoint(
                config.model_checkpoint_path,
                config=config,
                scaler=(
                    pl_dataloader.scaler if hasattr(pl_dataloader, "scaler") else None
                ),
            )
            print(OKBLUE + "model loaded from checkpoint" + ENDC)
            pl_model.log_dir = log_dir
    torch.set_float32_matmul_precision("high")

    print(OKBLUE + "initializing the trainer" + ENDC)
    trainer = L.Trainer(
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=config.training.num_devices,
        max_epochs=config.training.max_epochs,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        log_every_n_steps=config.training.log_every_n_steps,
        default_root_dir=save_path,
        callbacks=[checkpoint_callback],
        logger=mlf_logger,
        **kwargs,
    )
    print(OKBLUE + "trainer initialized" + ENDC)
    
    # trainer.test(pl_model, dataloaders=pl_dataloader.val_dataloader())
    trainer.fit(
        pl_model, pl_dataloader.train_dataloader(), pl_dataloader.val_dataloader()
    )
    # trainer.test(pl_model, dataloaders=pl_dataloader.test_dataloader())


if __name__ == "__main__":
    main()
 