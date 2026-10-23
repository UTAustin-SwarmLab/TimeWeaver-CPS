import os
import torch
import numpy as np
import hydra
import lightning as L
import logging

from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
import warnings

warnings.filterwarnings("ignore")

from timeseries_synthesis.utils.basic_utils import (
    seed_everything,
    import_from,
    get_scaler,
    get_synthesizer,
    get_dataloader_purpose,
    get_dataset_config,
    load_pretrained_model,
    generate_synthesis_experiment_details,
    OKBLUE,
    OKYELLOW,
    ENDC,
)

from timeseries_synthesis.utils.constrained_synthesis_utils import *

import time 
# import sleep


def generate_constrained_synthetic_dataset(
    constraints_to_extract,
    dataloader,
    synthesizer,
    synthesis_experiment_details,
    train=False,
    val=False,
    test=False,
):
    time_list = []
    if train:
        split_name = "train"
    if val:
        split_name = "val"
    if test:
        split_name = "test"

    save_dir = synthesis_experiment_details["save_dir"]
    save_dir = os.path.join(save_dir, split_name)
    synthetic_samples_exist = (
        True
        if "synthetic_sample_loc" in list(synthesis_experiment_details.keys())
        else False
    )
    if synthetic_samples_exist:
        if synthesis_experiment_details["using_real_seed"]:
            real_sample_loc = synthesis_experiment_details["synthetic_sample_loc"]
            real_train_samples = np.load(
                os.path.join(real_sample_loc, "train_timeseries.npy")
            )
            real_val_samples = np.load(
                os.path.join(real_sample_loc, "val_timeseries.npy")
            )
            real_samples = np.concatenate([real_train_samples, real_val_samples], axis=0)
            #shuffle the real samples along the first axis
            num_samples = real_samples.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            real_samples = real_samples[indices]
            print(OKBLUE + "Using real train and val samples for projection" + ENDC)
            
        else:
            synthetic_sample_loc = os.path.join(
                synthesis_experiment_details["synthetic_sample_loc"], split_name
            )

    print(OKBLUE + "Generating synthetic samples" + ENDC)
    num_test_samples = 0
    synthesis_function = synthesis_experiment_details["synthesis_function"]

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch index: {batch_idx}")  
        # load the warm start solution for projection after synthesis
        synthesis_experiment_details["batch_idx"] = batch_idx

        if synthetic_samples_exist:
            if synthesis_experiment_details["using_real_seed"]:
                batch_size = batch["timeseries_full"].shape[0]
                synthetic_sample_ts = real_samples[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]
                synthesis_experiment_details["synthetic_sample"] = synthetic_sample_ts
            else:
                synthetic_sample_ts_loc = os.path.join(
                    synthetic_sample_loc, f"timeseries_0.npy"
                )
                synthetic_batch = np.load(synthetic_sample_ts_loc)
                num_samples_ = synthetic_batch.shape[0]
                random_idx = np.random.choice(num_samples_, 1)
                synthetic_sample = synthetic_batch[random_idx]
                synthesis_experiment_details["synthetic_sample"] = synthetic_sample
                print(synthetic_sample.shape)
                
                
                
            # remember, this works because the batch sizes are the same for the synthetic data generated without constraints.
            
        start = time.time()

        for key, value in batch.items():
            batch[key] = value.to(synthesizer.config.device)
        dataset_dict = synthesis_function(
            batch=batch,
            synthesizer=synthesizer,
            constraints_to_extract=constraints_to_extract,
            synthesis_config=synthesis_experiment_details,
        )
        
        end = time.time()
        
        print(f"Time taken for synthesis: {end-start}")
        
        time_list.append(end-start)
        
        if len(time_list) == 10:
            break
        
    print(f"Average time taken for synthesis: {np.mean(time_list)}")
    print(f"Standard deviation of time taken for synthesis: {np.std(time_list)}")


@hydra.main(
    config_path="../../configs/constrained_generation_configs", version_base="1.1"
)
def main(config: DictConfig):
    torch.multiprocessing.set_start_method("spawn")
    seed_everything(config.seed)
    config.dataset.batch_size = 1
    pl_dataloader = import_from(
        f"timeseries_synthesis.datasets.lightening_dataloaders.{config.dataloader_file}",
        f"{config.dataloader_model}",
    )(config)
    get_dataloader_purpose(pl_dataloader)

    # define model type
    synthesizer_wrapper_type = import_from(
        f"timeseries_synthesis.models.lightening_modules.{config.synthesizer_wrapper_model_file}_trainer",
        config.synthesizer_wrapper_model_name,
    )

    # load model with pretrained weights
    synthesizer_wrapper = load_pretrained_model(
        synthesizer_wrapper_type, config.synthesizer_wrapper_checkpoint_path, config
    )

    if config.should_compile_torch:
        synthesizer_wrapper = torch.compile(
            synthesizer_wrapper
        )  # compiles the model and *step (training/validation/prediction)
        torch._dynamo.config.log_level = logging.ERROR

    L.seed_everything(config.seed)

    torch.set_float32_matmul_precision("high")
    synthesizer_wrapper.eval()
    for parameter in synthesizer_wrapper.parameters():
        parameter.requires_grad = False

    synthesizer = get_synthesizer(config, synthesizer_wrapper)

    # assign config.denosier_checkpoint_path to denoiser_model.log_dir excluding the last two folders in the path
    save_dir = "/" + os.path.join(
        *config.synthesizer_wrapper_checkpoint_path.split("/")[:-2]
    )

    dataset_config = get_dataset_config(config)
    num_constraints = len(dataset_config.equality_constraints_to_extract)
    dataset_dir = dataset_config.log_dir

    print(
        OKBLUE
        + "The constraints that will be extracted are : "
        + str(dataset_config.equality_constraints_to_extract)
        + ENDC
    )

    synthesis_experiment_details = generate_synthesis_experiment_details(
        save_dir, config, num_constraints, dataset_dir
    )
    print(synthesis_experiment_details.keys())

    print(
        OKYELLOW
        + "All the results will be stored in this directory: "
        + str(synthesis_experiment_details["save_dir"])
        + ENDC
    )

    os.makedirs(synthesis_experiment_details["save_dir"], exist_ok=True)

    print(
        OKYELLOW
        + "Let us first generate the synthetic dataset for the test conditions"
        + ENDC
    )

    test_dataloader = pl_dataloader.test_dataloader()

    generate_constrained_synthetic_dataset(
        constraints_to_extract=dataset_config.equality_constraints_to_extract,
        dataloader=test_dataloader,
        synthesizer=synthesizer,
        synthesis_experiment_details=synthesis_experiment_details,
        test=True,
    )

    return None


if __name__ == "__main__":
    main()
