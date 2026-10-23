import torch
from tqdm import tqdm
from time import sleep


def synthesis_via_diffusion(batch, synthesizer, similarity_guidance_dict=None):
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
    B = sample.shape[0]
    x = torch.randn_like(sample).to(device)

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
    discrete_conditions = batch["discrete_label_embedding"].detach().cpu().numpy()
    continuous_conditions = batch["continuous_label_embedding"].detach().cpu().numpy()

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict


def synthesis_via_gan(batch, synthesizer, similarity_guidance_dict=None):
    input_ = synthesizer.prepare_training_input(batch)
    synthesized = synthesizer.generator(
        x=input_["noise_for_generator"],
        y=input_["discrete_cond_input"],
        z=input_["continuous_cond_input"],
    )

    synthesized_timeseries = synthesizer.prepare_output(synthesized)
    discrete_conditions = batch["discrete_label_embedding"].detach().cpu().numpy()
    continuous_conditions = batch["continuous_label_embedding"].detach().cpu().numpy()

    dataset_dict = {
        "timeseries": synthesized_timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict

def forecast_via_decoders(batch, synthesizer):
    decoder_input = synthesizer.prepare_training_input(batch)
    forecast = synthesizer(decoder_input)
    discrete_conditions = batch["discrete_label_embedding"].detach().cpu().numpy()
    continuous_conditions = batch["continuous_label_embedding"].detach().cpu().numpy()
        
    # append history and forecast to form timeseries 
    history = decoder_input["history"]
    timeseries = torch.cat([history, forecast], dim=1)
    timeseries = timeseries.permute(0, 2, 1).cpu().detach().numpy()
    

    dataset_dict = {
        "timeseries": timeseries,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    return dataset_dict
