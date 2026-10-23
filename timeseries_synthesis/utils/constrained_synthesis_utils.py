import torch
import numpy as np
from tqdm import tqdm
from time import sleep
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated")

from timeseries_synthesis.utils.constrained_synthesis_helper_functions import *
from timeseries_synthesis.utils.gaussian_diffusion import Diffusion_TS

def synthesis_via_diffusion(
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
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)

    with torch.no_grad():
        for t in tqdm(range(T - 1, -1, -1), total=T):
            sleep(0.001)
            diffusion_steps = torch.LongTensor(
                [
                    t,
                ]
                * B
            ).to(device)
            synthesis_input = {
                "noisy_sample": x,
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps,
            }

            epsilon_theta = synthesizer(synthesis_input)
            x = (
                x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
            ) / torch.sqrt(Alpha[t])
            noise = torch.randn_like(x).to(device)
            if t > 0:
                x = x + Sigma[t] * noise

    synthesized_timeseries = synthesizer.prepare_output(x)

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": batch["discrete_label_embedding"].detach().cpu().numpy(),
        "continuous_conditions": batch["continuous_label_embedding"]
        .detach()
        .cpu()
        .numpy(),
    }

    return dataset_dict


def synthesis_via_gan(batch, synthesizer, constraints_to_extract, synthesis_config):
    input_ = synthesizer.prepare_training_input(batch)
    sample = input_["sample"]
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)
    synthesized = synthesizer.generator(
        x=input_["noise_for_generator"],
        y=input_["discrete_cond_input"],
        z=input_["continuous_cond_input"],
    )
    synthesized_timeseries = synthesizer.prepare_output(synthesized)

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": batch["discrete_label_embedding"].detach().cpu().numpy(),
        "continuous_conditions": batch["continuous_label_embedding"]
        .detach()
        .cpu()
        .numpy(),
    }

    return dataset_dict


def projection_post_synthesis(
    batch, synthesizer, constraints_to_extract, synthesis_config=None
):
    input_ = synthesizer.prepare_training_input(batch)
    sample = input_["sample"]
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)
    synthetic_sample = synthesis_config["synthetic_sample"]
    projected_timeseries = np.zeros_like(synthetic_sample)

    print(
        "Projecting the samples to the constraints using scipy with the discriminator in the objective function"
    )
    projected_timeseries = project_all_samples_to_equality_constraints_with_scipy_single_threaded(
        synthetic=synthetic_sample,
        warm_start=None,
        constraints=equality_constraints,
        discrete_conditions=batch["discrete_label_embedding"].detach().cpu().numpy(),
        continuous_conditions=batch["continuous_label_embedding"]
        .detach()
        .cpu()
        .numpy(),
        synthesis_config=synthesis_config,
    )
    print(
        "Finished projecting the samples to the constraints using scipy with the discriminator in the objective function"
    )

    dataset_dict = {
        "timeseries": projected_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": batch["discrete_label_embedding"].detach().cpu().numpy(),
        "continuous_conditions": batch["continuous_label_embedding"]
        .detach()
        .cpu()
        .numpy(),
    }
    return dataset_dict

def synthesis_via_projected_diffusion(
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
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)

    B = sample.shape[0]
    K = sample.shape[1]
    H = sample.shape[2]
    x = torch.randn_like(sample).to(device)
    warm_start_samples = sample.detach().cpu().numpy()
        
    differences_list = []
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

            # perform penalty-based projection
            if synthesis_config["gamma_choice"] == "lin":
                penalty_coefficient = Alpha_bar[t].item() * 1e5
            elif synthesis_config["gamma_choice"] == "quad":
                penalty_coefficient = Alpha_bar[t].item() ** 2 * 1e5
            else:
                penalty_coefficient = np.clip(np.exp(1 / (1 - Alpha_bar[t-1].item())), 0.1, 1e5)
            
            projected_x0_est = project_all_samples_to_equality_constraints(
                x0_est_numpy,
                equality_constraints,
                penalty_coefficient=penalty_coefficient,
                warm_start_samples=warm_start_samples,
                projection_method=synthesis_config["projection_during_synthesis"],
            )
            warm_start_samples = projected_x0_est
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
                Alpha_bar_prev = Alpha_bar[t-1]
                true_noisy_sample = (Alpha_bar_prev ** 0.5) * sample 
                diff = true_noisy_sample - x
                samplewise_diff = torch.mean(torch.square(diff), dim=(1, 2))
                differences_list.append({t: samplewise_diff})
                
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
        "differences": differences_list,
    }

    return dataset_dict

def synthesis_via_pdm_baseline(
    batch, synthesizer, constraints_to_extract, synthesis_config=None
):
    print("Using the basic PDM baseline")
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
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)

    B = sample.shape[0]
    K = sample.shape[1]
    H = sample.shape[2]
    x = torch.randn_like(sample).to(device)
    warm_start_samples = np.zeros((B, K, H))
    
    differences_list = []
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
            projected_x_est = project_all_samples_to_equality_constraints(
                x.detach().cpu().numpy(),
                equality_constraints,
                warm_start_samples=warm_start_samples,
                projection_method=synthesis_config["projection_during_synthesis"],
            )
            if synthesis_config["use_prodigy"]:
                gamma_init = synthesis_config["init_weight"]
                p_value = synthesis_config["p_value"]
                gamma = (1 - gamma_init) * (1-t/T)**p_value + gamma_init
            else:
                gamma = 1.0
            warm_start_samples = projected_x_est
            projected_x_est_tensor = torch.tensor(projected_x_est).to(device)
            x = x + gamma * (projected_x_est_tensor - x) 

    synthesized_timeseries = synthesizer.prepare_output(x)

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": batch["discrete_label_embedding"].detach().cpu().numpy(),
        "continuous_conditions": batch["continuous_label_embedding"]
        .detach()
        .cpu()
        .numpy(),
        "differences": differences_list,
    }

    return dataset_dict

def synthesis_via_guided_diffusion(
    batch, synthesizer, constraints_to_extract, synthesis_config
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
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)
 
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)

    guidance_weight = synthesis_config["guidance_weight"]
    with torch.no_grad():
        for t in tqdm(range(T - 1, -1, -1), total=T):
            sleep(0.001)
            diffusion_steps = torch.LongTensor(
                [
                    t,
                ]
                * B
            ).to(device)

            # enable gradient computation here
            torch.set_grad_enabled(True)
            x.requires_grad = True

            # get the denoiser input
            synthesis_input = {
                "noisy_sample": x,
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps,
            }

            # get the noise estimate
            epsilon_theta = synthesizer(synthesis_input)

            # get the constraint violation
            constraint_violation_batch = obtain_constraint_violation(
                x,
                epsilon_theta,
                Alpha_bar[t],
                equality_constraints,
            ) 

            # get the gradient of the constraint violation
            constraint_violation_sum = torch.sum(constraint_violation_batch)
            guidance_gradient = torch.autograd.grad(
                constraint_violation_sum, x, retain_graph=True 
            )[0]

            zeroify_gradient(x)
            x.requires_grad = False

            # disable gradient computation here
            torch.set_grad_enabled(False)

            # clip the gradient
            guidance_gradient = torch.clamp(guidance_gradient, -1000, 1000)

            epsilon_theta -= (
                guidance_weight * ((1 - Alpha_bar[t]) ** 0.5) * guidance_gradient
            )

            if t > 0:
                # print("t=0", Alpha_bar[t-1], Alpha_bar[t], t)
                x = (Alpha_bar[t - 1] ** 0.5) * get_sample_est_from_noisy_sample(
                    x, epsilon_theta, Alpha_bar[t]
                ) + (1.0 - Alpha_bar[t - 1]) ** 0.5 * epsilon_theta
            else:
                # print("t=0", Alpha_bar[t-1], Alpha_bar[t], t)
                x = get_sample_est_from_noisy_sample(x, epsilon_theta, Alpha_bar[t])

            if torch.isnan(x).any():
                raise ValueError("NAN detected in the synthesized timeseries")
            
            if synthesis_config["use_fixed_value_projection"]: 
                x = project_to_fixed_value_constraints(x, equality_constraints)

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = batch["discrete_label_embedding"].detach().cpu().numpy()
    continuous_conditions = batch["continuous_label_embedding"].detach().cpu().numpy()

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict 

def synthesis_via_guided_diffusion_modified(
    batch, synthesizer, constraints_to_extract, synthesis_config
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
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)
 
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)
    
    inp = torch.nn.Parameter(x)

    guidance_weight = synthesis_config["guidance_weight"]
    for t in tqdm(range(T - 1, -1, -1), total=T):
        sleep(0.001)
        diffusion_steps = torch.LongTensor(
            [
                t,
            ]
            * B
        ).to(device)

        # enable gradient computation here
        with torch.enable_grad():
            optimizer = torch.optim.Adagrad([inp], lr=1.0)
            optimizer.zero_grad()
            # get the denoiser input
            synthesis_input = {
                    "noisy_sample": inp,
                    "discrete_cond_input": discrete_cond_input,
                    "continuous_cond_input": continuous_cond_input,
                    "diffusion_step": diffusion_steps,
            }

            # get the noise estimate
            epsilon_theta = synthesizer(synthesis_input)

            # get the constraint violation
            constraint_violation_batch = obtain_constraint_violation(
                inp,
                epsilon_theta,
                Alpha_bar[t],
                equality_constraints,
            ) 

            # get the gradient of the constraint violation
            constraint_violation_sum = torch.sum(constraint_violation_batch)
        
            constraint_violation_sum.backward()
            
            # obtain the gradient
            guidance_gradient = inp.grad
        
            guidance_gradient = torch.clamp(guidance_gradient, -1000, 1000)
            
            epsilon_theta -= (
                guidance_weight * ((1 - Alpha_bar[t]) ** 0.5) * guidance_gradient
            ).detach()
        
            if t > 0:
                # print("t=0", Alpha_bar[t-1], Alpha_bar[t], t)
                x = (Alpha_bar[t - 1] ** 0.5) * get_sample_est_from_noisy_sample(
                    x, epsilon_theta, Alpha_bar[t]
                ) + (1.0 - Alpha_bar[t - 1]) ** 0.5 * epsilon_theta
            else:
                # print("t=0", Alpha_bar[t-1], Alpha_bar[t], t)
                x = get_sample_est_from_noisy_sample(x, epsilon_theta, Alpha_bar[t])

            if torch.isnan(x).any():
                raise ValueError("NAN detected in the synthesized timeseries")
                
            if synthesis_config["use_fixed_value_projection"]: 
                x = project_to_fixed_value_constraints(x, equality_constraints)
                
            inp = torch.nn.Parameter(x)

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = batch["discrete_label_embedding"].detach().cpu().numpy()
    continuous_conditions = batch["continuous_label_embedding"].detach().cpu().numpy()

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict 

def langevin_fn(
        synthesizer, synthesis_inp, mean, sigma, t, alpha_cumprod, alpha_next_cumprod, equality_constraints, coef=1.0, coef_=1.0, learning_rate=0.1
    ):
    
    if t < 200 * 0.05:
        K = 0 # no gradient update during the last 10 steps
    elif t > 200 * 0.9:
        K = 3 # 3 gradient updates during the first 10 steps
    elif t > 200 * 0.75:
        K = 2 # 2 gradient updates during the first 10-50 steps
        learning_rate = learning_rate * 0.5 # reduce learning rate by 0.5
    else:
        K = 1 # 1 gradient update during the first 50-190 steps
        learning_rate = learning_rate * 0.25 # reduce learning rate by 0.25
            
    sample = synthesis_inp["noisy_sample"] # x_{t-1}
    input_embs_param = torch.nn.Parameter(sample) # x_{t-1} with gradient updates
    assert torch.equal(input_embs_param.data, mean)

    updated_x_prev = sample
    with torch.enable_grad():
        for i in range(K):
            optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
            optimizer.zero_grad()
            synthesis_inp["noisy_sample"] = input_embs_param
            eps_theta = synthesizer(synthesis_inp) # pred_noise for x_{t-1}
            logp_term = ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum() # how close is the sample to the mean for x_{t-1}
                
            constraint_violation_batch = obtain_constraint_violation(input_embs_param, eps_theta, alpha_next_cumprod, equality_constraints) # constraint violation for x_{t-1} -> x_0
            constraint_violation_sum = torch.sum(constraint_violation_batch)
            # print("Constraint violation sum: ", constraint_violation_sum * coef_)
            # print("Logp term: ", logp_term.item())
            loss = logp_term + coef_ * constraint_violation_sum # loss for x_{t-1}
            loss.backward()
            optimizer.step() # updates input_embs_param or x_{t-1} based on the loss
            epsilon = torch.randn_like(input_embs_param.data)
            input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma * epsilon).detach()) # updates x_{t-1} with noise, but actually no noise is added
            
            updated_x_prev = input_embs_param.data

    assert torch.equal(input_embs_param.data, updated_x_prev)

    return input_embs_param.data
    

def synthesis_via_diffts(
    batch, synthesizer, constraints_to_extract, synthesis_config
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
    equality_constraints = extract_equality_constraints(sample, constraints_to_extract)
 
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)

    guidance_weight = synthesis_config["instance_weight"]
    for t in tqdm(range(T - 1, -1, -1), total=T):
        tnext = t-1
        sleep(0.001)
        diffusion_steps = torch.LongTensor(
            [
                t,
            ]
            * B
        ).to(device) # t
        next_diffusion_steps = torch.LongTensor(
            [
                tnext,
            ]
            * B
        ).to(device) # t-1

        # get the denoiser input
        synthesis_input = {
                "noisy_sample": x, # x_t
                "discrete_cond_input": discrete_cond_input,
                "continuous_cond_input": continuous_cond_input,
                "diffusion_step": diffusion_steps, # t
        }

        # get the noise estimate
        epsilon_theta = synthesizer(synthesis_input) # pred_noise at t
        x0_est = get_sample_est_from_noisy_sample(x, epsilon_theta, Alpha_bar[t]) # x_0(x_t, pred_noise at t)
        
        alpha_cumprod = Alpha_bar[t]
        alpha_next_cumprod = Alpha_bar[tnext]
        
        if tnext < 0:
            x = x0_est  
            # again, from their code. No gradient update after the last denoising step.
            continue
        
        sigma = 0 # from their code, deterministic DDIM sampling, same as Guided DiffTime
        c = (1.0 - alpha_next_cumprod) ** 0.5
        xprev_mean = x0_est * alpha_next_cumprod.sqrt() + c * epsilon_theta
        noise = torch.randn_like(xprev_mean).to(device)
        xprev_sample = xprev_mean + sigma * noise # deterministic sampling, x_{t-1}, same as xprev_mean
        
        synthesis_inp = {
            "noisy_sample": xprev_sample, # x_{t-1}
            "discrete_cond_input": discrete_cond_input, 
            "continuous_cond_input": continuous_cond_input, 
            "diffusion_step":next_diffusion_steps # t-1
        }
        xprev_sample_updated = langevin_fn(synthesizer=synthesizer, 
                                           synthesis_inp=synthesis_inp, # contains x_{t-1} and t-1
                                           mean=xprev_mean, # contains mean of x_{t-1}
                                           sigma=sigma, # 0
                                           t=t, # current time step
                                           alpha_cumprod=alpha_cumprod, # Alpha_bar[t] 
                                           alpha_next_cumprod=alpha_next_cumprod, # Alpha_bar[t-1]
                                           equality_constraints=equality_constraints, # equality constraints
                                           coef_ = guidance_weight # guidance weight
                                )
        
        
        x = xprev_sample_updated
        if torch.isnan(x).any():
            raise ValueError("NAN detected in the synthesized timeseries")

    synthesized_timeseries = synthesizer.prepare_output(x)
    discrete_conditions = batch["discrete_label_embedding"].detach().cpu().numpy()
    continuous_conditions = batch["continuous_label_embedding"].detach().cpu().numpy()

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "equality_constraints": equality_constraints,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict 
