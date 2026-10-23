from typing import Dict
from easydict import EasyDict as edict
import numpy as np
import torch
import random
import os

from timeseries_synthesis.datasets.dataloaders.electricity_utils import (
    ElectricityTransformationUtils,
)
from timeseries_synthesis.datasets.dataloaders.ecg_utils import (
    ECGTransformationUtils,
)
from timeseries_synthesis.datasets.dataloaders.ecg_reduced_utils import (
    ECGReducedTransformationUtils,
)
from timeseries_synthesis.datasets.dataloaders.traffic_utils import (
    TrafficTransformationUtils,
)
from timeseries_synthesis.datasets.dataloaders.stocks_utils import (
    StocksTransformationUtils,
)

from timeseries_synthesis.utils.constrained_synthesis_utils import (
    synthesis_via_guided_diffusion,
    synthesis_via_guided_diffusion_modified,
    synthesis_via_projected_diffusion,
    projection_post_synthesis,
    synthesis_via_gan,
    synthesis_via_diffusion,
    synthesis_via_pdm_baseline,
    synthesis_via_diffts,
)
from timeseries_synthesis.utils.imputation_utils import imputation_via_projected_diffusion, imputation_via_pdm_baseline

def load_pretrained_model(model_type, model_checkpoint_path, config):
    print(OKBLUE + "loading model from checkpoint" + ENDC)
    model = model_type.load_from_checkpoint(
        model_checkpoint_path, config=config, scaler=None, strict=True
    )
    print(OKBLUE + "model loaded from checkpoint" + ENDC)
    return model

def generate_synthesis_experiment_details(save_dir, config, num_constraints, dataset_dir, eval_mode=False):
    synthesis_experiment_details = {}

    num_constraints_str = str(num_constraints) + "_constraints"
    if config.use_guidance:
        if config.use_fixed_value_projection:
            option = "guidance_with_fixed_value_projection"
            synthesis_experiment_details["use_fixed_value_projection"] = True
        else:
            option = "guidance" 
            synthesis_experiment_details["use_cop"] = False
        option += "_weight_%0.8f" % config.guidance_weight
        synthesis_experiment_details["guidance_weight"] = config.guidance_weight
        synthesis_experiment_details["synthesis_function"] = (
            synthesis_via_guided_diffusion_modified
        )

        save_dir = os.path.join(save_dir, num_constraints_str, option)
        
    elif config.use_diffts:
        synthesis_experiment_details["instance_weight"] = config.instance_weight
        option = "diffts_weight_%0.8f" % config.instance_weight
        synthesis_experiment_details["synthesis_function"] = synthesis_via_diffts
        save_dir = os.path.join(save_dir, num_constraints_str, option)
 
    elif config.use_projection:
        # we are using projection
        # there are two choices here:
        # 1. project during synthesis
        # 2. project after synthesis
        option = "projection"
        if config.project_during_synthesis:
            option += "_during_synthesis"
            # we have two choices here:
            # 1. strict projection
            # 2. penalty based projection
            if config.use_strict_projection:
                synthesis_experiment_details["projection_during_synthesis"] = "strict"
                synthesis_experiment_details["synthesis_function"] = synthesis_via_pdm_baseline 
                synthesis_experiment_details["imputation_function"] = imputation_via_pdm_baseline
                print(OKYELLOW + "Using PDM Baseline with %d projection steps" % config.num_projection_steps + ENDC)

                synthesis_experiment_details["num_projection_steps"] = config.num_projection_steps
                synthesis_experiment_details["num_denoising_steps"] = config.num_denoising_steps
                synthesis_experiment_details["init_weight"] = config.init_weight
                synthesis_experiment_details["p_value"] = config.p_value
                if config.use_prodigy:
                    # prodigy baseline
                    option += "_with_strict_projection_using_%d_projection_steps_and_%d_denoising_steps_init_weight_%0.8f_p_value_%0.8f" % (config.num_projection_steps, config.num_denoising_steps, config.init_weight, config.p_value)
                    synthesis_experiment_details["use_prodigy"] = True
                else:
                    # pdm baseline
                    option += "_with_strict_projection_using_%d_projection_steps_and_%d_denoising_steps" % (config.num_projection_steps, config.num_denoising_steps)
                    synthesis_experiment_details["use_prodigy"] = False
            if config.use_penalty_based_projection:
                synthesis_experiment_details["projection_during_synthesis"] = (
                    "penalty_based"
                )
                synthesis_experiment_details["synthesis_function"] = (
                    synthesis_via_projected_diffusion
                )
                synthesis_experiment_details["imputation_function"] = (
                    imputation_via_projected_diffusion
                )
                synthesis_experiment_details["gamma_choice"] = config.gamma_choice
                option += "_with_penalty_based_projection_using_%s" % config.gamma_choice
                # option += "_with_penalty_based_projection"
                
            # if compare_with_pdm is present, then add to the option
            # if config.compare_with_pdm:
            #     synthesis_experiment_details["selectively_denoise"] = True
            # else:
            #     synthesis_experiment_details["selectively_denoise"] = False

        if config.project_after_synthesis:
            option += "_after_synthesis"
            # we have two choices here:
            # 1. use discriminator
            # 2. do not use discriminator

            # we will use the existing synthetic data and post process it
            if config.use_real_seed:
                synthesis_experiment_details["using_real_seed"] = True
                synthesis_experiment_details["synthetic_sample_loc"] = dataset_dir
                option += "_with_real_seed"
            else:
                synthesis_experiment_details["using_real_seed"] = False
                synthesis_experiment_details["synthetic_sample_loc"] = os.path.join(
                    save_dir, "normal"
                )


            if config.use_discriminator:
                option += "_with_discriminator"
                # we have two choices here:
                # 1. use discriminator in projection objective
                # 2. use discriminator in refinement objective

                # load the discriminator
                synthesis_experiment_details["use_discriminator"] = True
                if not eval_mode:
                    discriminator_wrapper_model_file = (
                        config.discriminator_params.discriminator_wrapper_model_file
                    )
                    discriminator_wrapper_model_name = (
                        config.discriminator_params.discriminator_wrapper_model_name
                    )
                    discriminator_wrapper_checkpoint_path = (
                        config.discriminator_params.discriminator_wrapper_checkpoint_path
                    )
                    discriminator_wrapper_type = import_from(
                        f"timeseries_synthesis.models.lightening_modules.{discriminator_wrapper_model_file}_trainer",
                        discriminator_wrapper_model_name,
                    )
                    config.gan_name = config.discriminator_params.discriminator_name
                    discriminator_wrapper = load_pretrained_model(
                        discriminator_wrapper_type,
                        discriminator_wrapper_checkpoint_path,
                        config,
                    )
                    discriminator_wrapper.eval()
                    for parameter in discriminator_wrapper.parameters():
                        parameter.requires_grad = False
                    synthesis_experiment_details["discriminator_wrapper"] = (
                        discriminator_wrapper
                    )

                    if config.discriminator_params.use_gan_scaling:
                        print(OKYELLOW + "Using GAN scaling" + ENDC)
                        synthesis_experiment_details["discriminator_input_scaler"] = (
                            get_scaler(config)
                        )
                    else:
                        synthesis_experiment_details["discriminator_input_scaler"] = None
                
                synthesis_experiment_details["discriminator_weight"] = (
                    config.discriminator_params.discriminator_weight 
                )

                # Projection objective
                if config.in_projection_objective:
                    option += (
                        "_in_projection_objective_with_discriminator_weight_%0.8f"
                        % (config.discriminator_params.discriminator_weight)
                    )
                    synthesis_experiment_details["in_projection_objective"] = True
                    synthesis_experiment_details["in_refinement_objective"] = False

                # Refinement objective
                if config.in_refinement_objective:
                    option += "_in_refinement_objective"
                    synthesis_experiment_details["in_projection_objective"] = False
                    synthesis_experiment_details["in_refinement_objective"] = True
                    num_refinement_steps = (
                        config.discriminator_params.num_refinement_steps
                    )
                    synthesis_experiment_details["num_refinement_steps"] = (
                        num_refinement_steps
                    )
                    option += f"_using_{num_refinement_steps}_refinement_steps"

            else:
                option += "_without_discriminator"
                synthesis_experiment_details["use_discriminator"] = False
            synthesis_experiment_details["synthesis_function"] = (
                projection_post_synthesis
            )
            synthesis_experiment_details["imputation_function"] = (
                imputation_post_synthesis
            )

        save_dir = os.path.join(save_dir, num_constraints_str, option)
    else:
        option = "normal"
        synthesis_experiment_details["synthesis_function"] = synthesis_via_diffusion
        synthesis_experiment_details["we_using_gan"] = False
        save_dir = os.path.join(save_dir, option)

    synthesis_experiment_details["save_dir"] = save_dir
    return synthesis_experiment_details


def split_into_batches(tensor, batch_size):
    num_samples = tensor.shape[0]
    batches = []
    start_idx = 0
    end_idx = batch_size
    while end_idx <= num_samples:
        batch = tensor[start_idx:end_idx]
        batches.append(batch)
        start_idx = end_idx
        end_idx += batch_size
    batches.append(tensor[start_idx:])
    return batches


def get_synthesizer(config, synthesizer_wrapper):
    if "diffusion" in config.synthesizer_wrapper_model_file:
        synthesizer = synthesizer_wrapper.denoiser_model
    elif "gan" in config.synthesizer_wrapper_model_file:
        synthesizer = synthesizer_wrapper.synthesizer
    else:
        raise NotImplementedError
    return synthesizer


def get_scaler(config):
    if config.dataset_name == "electricity":
        print(OKYELLOW + "Loading the scaler" + ENDC)
        print(
            OKYELLOW
            + "The scaler will be loaded from: "
            + str(config.electricity_dataset.log_dir)
            + ENDC
        )
        print(OKYELLOW + "The scaler is ElectricityTransformationUtils" + ENDC)
        scaler = ElectricityTransformationUtils(config)
    elif config.dataset_name == "ecg":
        print(OKYELLOW + "Loading the scaler" + ENDC)
        print(
            OKYELLOW
            + "The scaler will be loaded from: "
            + str(config.ecg_dataset.log_dir)
            + ENDC
        )
        print(OKYELLOW + "The scaler is ECGTransformationUtils" + ENDC)
        scaler = ECGTransformationUtils(config)
    elif config.dataset_name == "ecg_reduced":
        print(OKYELLOW + "Loading the scaler" + ENDC)
        print(
            OKYELLOW
            + "The scaler will be loaded from: "
            + str(config.ecg_reduced_dataset.log_dir)
            + ENDC
        )
        print(OKYELLOW + "The scaler is ECGReducedTransformationUtils" + ENDC)
        scaler = ECGReducedTransformationUtils(config)
    elif config.dataset_name == "traffic":
        print(OKYELLOW + "Loading the scaler" + ENDC)
        print(
            OKYELLOW
            + "The scaler will be loaded from: "
            + str(config.traffic_dataset.log_dir)
            + ENDC
        )
        print(OKYELLOW + "The scaler is TrafficTransformationUtils" + ENDC)
        scaler = TrafficTransformationUtils(config)
    elif config.dataset_name == "stocks":
        print(OKYELLOW + "Loading the scaler" + ENDC)
        print(
            OKYELLOW
            + "The scaler will be loaded from: "
            + str(config.stocks_dataset.log_dir)
            + ENDC
        )
        print(OKYELLOW + "The scaler is StocksTransformationUtils" + ENDC)
        scaler = StocksTransformationUtils(config)
    else:
        scaler = None
    return scaler


def get_dataloader_purpose(pl_dataloader):
    print(OKYELLOW + "checking dataloader" + ENDC)
    for batch in pl_dataloader.test_dataloader():
        for key, val in batch.items():
            print(key, val.shape)
        break
    print(OKYELLOW + "dataloader check over" + ENDC)


def get_classifier_config(config):
    return config.classifier_config


def get_denoiser_config(config):
    if config.denoiser_name == "csdi_timeseries_denoiser_v1":
        denoiser_config = config.csdi_timeseries_denoiser_v1_config
    elif config.denoiser_name == "csdi_timeseries_denoiser_v4":
        denoiser_config = config.csdi_timeseries_denoiser_v4_config
    elif config.denoiser_name == "csdi_timeseries_denoiser_v5":
        denoiser_config = config.csdi_timeseries_denoiser_v5_config
    elif config.denoiser_name == "unet_timeseries_denoiser_v1":
        denoiser_config = config.unet_timeseries_denoiser_v1_config
    else:
        raise ValueError("denoiser name not recognized")
    return denoiser_config


def get_gan_config(config):
    if config.gan_name == "p2p":
        gan_config = config.p2p_config
    elif config.gan_name == "p2p_v1":
        gan_config = config.p2p_v1_config
    elif config.gan_name == "wavegan":
        gan_config = config.wavegan_config
    elif config.gan_name == "wavegan_v1":
        gan_config = config.wavegan_v1_config
    else:
        raise ValueError("gan name not recognized")
    return gan_config


def get_cltsp_config(config):
    if config.cltsp_name == "cltsp_v3":
        cltsp_config = config.cltsp_v3_config
    else:
        raise ValueError("cltsp name not recognized")
    return cltsp_config


def get_dataset_config(config):
    if config.dataset_name == "air_quality":
        dataset_config = config.air_quality_dataset
    elif config.dataset_name == "stocks":
        dataset_config = config.stocks_dataset
    elif config.dataset_name == "traffic":
        dataset_config = config.traffic_dataset
    elif config.dataset_name == "waveforms":
        dataset_config = config.waveforms_dataset
    else:
        raise ValueError("dataset name not recognized")
    return dataset_config


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def edict2dict(edict_obj: Dict):
    dict_obj = {}
    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals
    return dict_obj


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


OKRED = "\033[91m"
OKBLUE = "\033[94m"
ENDC = "\033[0m"
OKGREEN = "\033[92m"
OKYELLOW = "\033[93m"
