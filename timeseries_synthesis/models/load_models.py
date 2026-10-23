import os
from omegaconf import DictConfig
from typing import List, Dict, Tuple

import torch
from timeseries_synthesis.utils.basic_utils import OKBLUE, OKYELLOW, ENDC

from timeseries_synthesis.utils.basic_utils import (
    get_cltsp_config,
    get_dataset_config,
)

from timeseries_synthesis.models.cltsp_models.cltsp_v3 import CLTSP_v3

from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.csdi_denoiser_v1 import (
    CSDITSDenoiser_v1,
)
from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.csdi_denoiser_v4 import (
    CSDITSDenoiser_v4,
)
from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.csdi_denoiser_v5 import (
    CSDITSDenoiser_v5,
)
from timeseries_synthesis.models.diffusion_models.timeseries_diffusion_models.unet import (
    Unet1D,
)
from timeseries_synthesis.models.gan_models.wavegan import CondWaveGAN
from timeseries_synthesis.models.gan_models.wavegan_v1 import CondWaveGAN_v1


def load_timeseries_denoiser(config: DictConfig):
    if config.denoiser_name == "csdi_timeseries_denoiser_v1":
        denoiser_model = CSDITSDenoiser_v1(
            config=config,
        )
    elif config.denoiser_name == "csdi_timeseries_denoiser_v4":
        denoiser_model = CSDITSDenoiser_v4(
            config=config,
        )
    elif config.denoiser_name == "csdi_timeseries_denoiser_v5":
        denoiser_model = CSDITSDenoiser_v5(
            config=config,
        )
    elif config.denoiser_name == "unet_timeseries_denoiser_v1":
        denoiser_model = Unet1D(
            config=config,
        )
    else:
        raise ValueError("denoiser name not recognized")
    return denoiser_model


def load_timeseries_gan(config: DictConfig):
    if config.gan_name == "wavegan":
        gan_model = CondWaveGAN(config)
    elif config.gan_name == "wavegan_v1":
        gan_model = CondWaveGAN_v1(config)
    else:
        raise ValueError("gan name not recognized")
    return gan_model


def load_cltsp_model(config: DictConfig) -> torch.nn.Module:
    cltsp_config = get_cltsp_config(config=config)
    dataset_config = get_dataset_config(config=config)
    device = config.device

    if config.cltsp_name == "cltsp_v3":
        cltsp_model = CLTSP_v3(
            cltsp_config=cltsp_config,
            dataset_config=dataset_config,
            device=device,
        )
    else:
        raise ValueError("cltsp model name not recognized")
    return cltsp_model