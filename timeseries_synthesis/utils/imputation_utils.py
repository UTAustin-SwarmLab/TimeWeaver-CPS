import torch
import numpy as np
from tqdm import tqdm
from time import sleep
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")

from timeseries_synthesis.utils.constrained_synthesis_helper_functions import *
from timeseries_synthesis.utils.gaussian_diffusion import Diffusion_TS

def imputation_via_projected_diffusion(
    batch, synthesizer, constraints_to_extract, synthesis_config=None
):
    T, Alpha, Alpha_bar, Sigma = (
        synthesizer.diffusion_hyperparameters["T"],
        synthesizer.diffusion_hyperparameters["Alpha"],
        synthesizer.diffusion_hyperparameters["Alpha_bar"],
        synthesizer.diffusion_hyperparameters["Sigma"],
    )
    device = synthesizer.device
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)

    input_ = synthesizer.prepare_training_input(batch)
    discrete_cond_input = input_["discrete_cond_input"]
    continuous_cond_input = input_["continuous_cond_input"]

    sample = input_["sample"]
    constraints_to_extract = []
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)

    B = sample.shape[0]
    K = sample.shape[1]
    H = sample.shape[2]
    x = torch.randn_like(sample).to(device)
    warm_start_samples = sample.detach().cpu().numpy()

    # intitialize a random sample from 0 to 1
    rand_x = torch.zeros(B, K, H).to(device)
    rand_x[:, :, :48] = 1.0
    loc_to_fix = rand_x > 0.5
    loc_to_fix = loc_to_fix.cpu().numpy()

    
    with torch.no_grad():
        for t in tqdm(range(T - 1, -1, -1), total=T):
            sleep(0.001)
            # print(t)
            diffusion_steps = torch.LongTensor(
                [
                    t,
                ]
                * B
            ).to(device)

            # get the denoiser input
            synthesis_input = {
                "noisy_sample": x,
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps, 
            }

            # get the noise estimate
            epsilon_theta = synthesizer(synthesis_input)

            # get the clean sample estimate
            # print(Alpha_bar[t-1], Alpha_bar[t], t)
            x0_est = get_sample_est_from_noisy_sample(x, epsilon_theta, Alpha_bar[t])
            x0_est_numpy = x0_est.detach().cpu().numpy()

            projected_x0_est = x0_est_numpy.copy()
            projected_x0_est[loc_to_fix] = sample[loc_to_fix].detach().cpu().numpy()

            projected_x0_est = torch.tensor(projected_x0_est).to(device)

            control_param = ((1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])) * (
                1 - Alpha_bar[t] / Alpha_bar[t - 1]
            )  # DDPM
            if t > 0:
                noise = torch.randn_like(x).to(device)
                x = (
                    (Alpha_bar[t - 1] ** 0.5) * projected_x0_est
                    + (1.0 - Alpha_bar[t - 1] - control_param) ** 0.5 * epsilon_theta
                    + noise * (control_param**0.5)
                )
            else:
                x = projected_x0_est

    synthesized_timeseries = synthesizer.prepare_output(x)

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": batch["discrete_label_embedding"].detach().cpu().numpy(),
        "continuous_conditions": batch["continuous_label_embedding"]
        .detach()
        .cpu()
        .numpy(),
        "loc_to_fix": loc_to_fix,
        "true_timeseries": sample.detach().cpu().numpy(),
    }

    return dataset_dict

def imputation_via_pdm_baseline(
    batch, synthesizer, constraints_to_extract, synthesis_config=None
):
    print("Using the basic PDM baseline for imputation")
    T, Alpha, Alpha_bar, Sigma = (
        synthesizer.diffusion_hyperparameters["T"],
        synthesizer.diffusion_hyperparameters["Alpha"],
        synthesizer.diffusion_hyperparameters["Alpha_bar"],
        synthesizer.diffusion_hyperparameters["Sigma"],
    )
    device = synthesizer.device
    Alpha = Alpha.to(device)
    Alpha_bar = Alpha_bar.to(device)
    Sigma = Sigma.to(device)

    input_ = synthesizer.prepare_training_input(batch)
    discrete_cond_input = input_["discrete_cond_input"]
    continuous_cond_input = input_["continuous_cond_input"]

    sample = input_["sample"]
    constraints_to_extract = []
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)

    B = sample.shape[0]
    K = sample.shape[1]
    H = sample.shape[2]
    x = torch.randn_like(sample).to(device)

    # intitialize a random sample from 0 to 1
    rand_x = torch.rand(B, K, H).to(device)
    loc_to_fix = rand_x > 0.5
    loc_to_fix = loc_to_fix.cpu().numpy()    
    
    with torch.no_grad():
        for t in tqdm(range(T - 1, -1, -1), total=T):
            sleep(0.001)
            # print(t)
            diffusion_steps = torch.LongTensor(
                [
                    t,
                ]
                * B
            ).to(device)

            # get the denoiser input
            synthesis_input = {
                "noisy_sample": x,
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps,
            }

            # get the noise estimate
            epsilon_theta = synthesizer(synthesis_input)
            x = (
                x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
            ) / torch.sqrt(Alpha[t])
            noise = torch.randn_like(x).to(device)
            if t > 0:
                x = x + Sigma[t] * noise
                
            # perform projection to constrain x to the equality constraints
            # print("Projection step: ", proj_idx, " at time step: ", t)        
            projected_x_est = x.detach().cpu().numpy()
            projected_x_est[loc_to_fix] = sample[loc_to_fix].detach().cpu().numpy()
            projected_x_est = torch.tensor(projected_x_est).to(device)
             
            if synthesis_config["use_prodigy"]:
                gamma_init = synthesis_config["init_weight"]
                p_value = synthesis_config["p_value"]
                gamma = (1 - gamma_init) * (1-t/T)**p_value + gamma_init
            else:
                gamma = 1.0
            x = x + gamma * (projected_x_est - x) 

    synthesized_timeseries = synthesizer.prepare_output(x)

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": batch["discrete_label_embedding"].detach().cpu().numpy(),
        "continuous_conditions": batch["continuous_label_embedding"]
        .detach()
        .cpu()
        .numpy(),
        "loc_to_fix": loc_to_fix,
        "true_timeseries": sample.detach().cpu().numpy(),
    }

    return dataset_dict
