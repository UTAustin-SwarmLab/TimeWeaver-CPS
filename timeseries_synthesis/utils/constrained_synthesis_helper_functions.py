import os
import numpy as np
import torch
from time import time
import pickle as pkl
import cvxpy as cp
import multiprocessing as mp
from functools import partial
from scipy.optimize import minimize, LinearConstraint
import scipy.signal as signal

from timeseries_synthesis.utils.constraint_utils.mean_constraint_utils import (
    obtain_mean_constraint_violation,
    obtain_scipy_mean_constraint,
    obtain_mean_penalty_cvxpy,
)
from timeseries_synthesis.utils.constraint_utils.mean_change_constraint_utils import (
    obtain_mean_change_constraint_violation,
    obtain_scipy_mean_change_constraint,
    obtain_mean_change_penalty_cvxpy,
)
from timeseries_synthesis.utils.constraint_utils.value_at_constraint_utils import (
    obtain_value_at_constraint_violation,
    obtain_scipy_value_at_constraint,
    obtain_value_at_penalty_cvxpy,
) 
from timeseries_synthesis.utils.constraint_utils.argmax_constraint_utils import (
    obtain_argmax_constraint_violation,
    obtain_scipy_argmax_constraint,
    obtain_argmax_penalty_cvxpy,
)
from timeseries_synthesis.utils.constraint_utils.argmin_constraint_utils import (
    obtain_argmin_constraint_violation,
    obtain_scipy_argmin_constraint,
    obtain_argmin_penalty_cvxpy,
)
from timeseries_synthesis.utils.constraint_utils.ohlc_constraint_utils import (
    obtain_ohlc_constraint_violation,
    obtain_scipy_ohlc_constraint,
    obtain_ohlc_penalty_cvxpy,
    obtain_ohlc_penalty_scipy,
)
from timeseries_synthesis.utils.constraint_utils.peak_and_valley_constraint_utils import (
    obtain_peak_and_valley_constraint_violation,
    obtain_scipy_peak_and_valley_constraint,
    obtain_peak_and_valley_penalty_cvxpy,
)
from timeseries_synthesis.utils.constraint_utils.max_and_argmax_constraint_utils import (
    obtain_max_and_argmax_constraint_violation,
)
from timeseries_synthesis.utils.constraint_utils.min_and_argmin_constraint_utils import (
    obtain_min_and_argmin_constraint_violation,
)
from timeseries_synthesis.utils.constraint_utils.autocorr_constraint_utils import (
    obtain_autocorr_constraint_violation,
    obtain_scipy_autocorr_constraint,
    obtain_autocorr_penalty_scipy
)

ACCEPTED_TOERANCE = 1e-2

"""
Function to extract the required equality constraints from the batch of real samples
"""


def obtain_peak_valley_pairs(sample):
    horizon = len(sample)
    peak_locs = list(signal.find_peaks(sample)[0])
    valley_locs = list(signal.find_peaks(-sample)[0])
    zero_grad_loc = peak_locs + valley_locs + [0, horizon - 1]
    zero_grad_loc = np.sort(zero_grad_loc)
    peak_valley_pairs = []
    peak_valley_trends = []
    peak_valley_values = []
    for i in range(1, len(zero_grad_loc)):
        start = zero_grad_loc[i - 1]
        end = zero_grad_loc[i]
        trend = np.sign(sample[end] - sample[start])
        peak_valley_pairs.append((start, end))
        peak_valley_trends.append(trend)
        peak_valley_values.append((sample[start], sample[end]))
    return peak_valley_pairs, peak_valley_trends, peak_valley_values


def extract_equality_constraints(batch_samples, constraints_to_extract):
    # check if batch_samples is a tensor
    if isinstance(batch_samples, torch.Tensor):
        batch_samples_numpy = batch_samples.detach().cpu().numpy()
    else:
        batch_samples_numpy = batch_samples
    equality_constraints = {}
    B = batch_samples_numpy.shape[0]
    C = batch_samples_numpy.shape[1]
    T = batch_samples_numpy.shape[2]
    for constraint_name in constraints_to_extract:
        if constraint_name == "argmax":
            argmax_value = np.argmax(batch_samples_numpy, axis=-1)
            equality_constraints["argmax"] = argmax_value
        elif constraint_name == "max and argmax":
            max_value = np.max(batch_samples_numpy, axis=-1)
            argmax_value = np.argmax(batch_samples_numpy, axis=-1)
            constraint_dict = {"max": max_value, "argmax": argmax_value}
            equality_constraints["max and argmax"] = constraint_dict
        elif constraint_name == "argmin":
            argmin_value = np.argmin(batch_samples_numpy, axis=-1)
            equality_constraints["argmin"] = argmin_value
        elif constraint_name == "min and argmin":
            min_value = np.min(batch_samples_numpy, axis=-1)
            argmin_value = np.argmin(batch_samples_numpy, axis=-1)
            constraint_dict = {"min": min_value, "argmin": argmin_value}
            equality_constraints["min and argmin"] = constraint_dict
        elif constraint_name == "mean":
            mean_value = np.mean(batch_samples_numpy, axis=-1)
            equality_constraints["mean"] = mean_value
        elif constraint_name == "mean change":
            mean_change = np.mean(np.diff(batch_samples_numpy, axis=-1), axis=-1)
            equality_constraints["mean change"] = mean_change
        elif "val_at" in constraint_name:
            timestep = int(constraint_name.split("_")[-1]) - 1
            value_at_timestep = batch_samples_numpy[:, :, timestep]
            equality_constraints[constraint_name] = value_at_timestep
        elif constraint_name == "peak and valley":
            num_samples = batch_samples_numpy.shape[0]
            num_channels = batch_samples_numpy.shape[1]
            peak_valley_constraints = {}
            peak_and_valley_trend_constraint_matrix = np.zeros((B, C, T - 1, T))
            peak_and_valley_values_matrix = np.zeros((B, C, T))
            peak_and_valley_indicator_matrix = np.zeros((B, C, T))
            for sample_idx in range(num_samples):
                for channel_idx in range(num_channels):
                    timeseries = batch_samples_numpy[sample_idx, channel_idx]
                    peak_valley_pairs, peak_valley_trends, peak_valley_values = (
                        obtain_peak_valley_pairs(timeseries)
                    )
                    peak_valley_constraints[(sample_idx, channel_idx)] = (
                        peak_valley_pairs,
                        peak_valley_trends,
                        peak_valley_values,
                    )
                    for timestamp in range(T - 1):
                        current_val = timeseries[timestamp]
                        next_val = timeseries[timestamp + 1]
                        if next_val > current_val:
                            arr = np.zeros(T)
                            arr[timestamp] = -1
                            arr[timestamp + 1] = 1
                            peak_and_valley_trend_constraint_matrix[
                                sample_idx, channel_idx, timestamp
                            ] = arr
                        else:
                            arr = np.zeros(T)
                            arr[timestamp] = 1
                            arr[timestamp + 1] = -1
                            peak_and_valley_trend_constraint_matrix[
                                sample_idx, channel_idx, timestamp
                            ] = arr

                    for peak_valley_pair, peak_valley_value in zip(
                        peak_valley_pairs, peak_valley_values
                    ):
                        start_loc = peak_valley_pair[0]
                        end_loc = peak_valley_pair[1]
                        start_val = peak_valley_value[0]
                        end_val = peak_valley_value[1]

                        peak_and_valley_values_matrix[
                            sample_idx, channel_idx, start_loc
                        ] = start_val
                        peak_and_valley_values_matrix[
                            sample_idx, channel_idx, end_loc
                        ] = end_val
                        peak_and_valley_indicator_matrix[
                            sample_idx, channel_idx, start_loc
                        ] = 1
                        peak_and_valley_indicator_matrix[
                            sample_idx, channel_idx, end_loc
                        ] = 1

            peak_valley_constraints["trend_constraint_matrix"] = (
                peak_and_valley_trend_constraint_matrix
            )
            peak_valley_constraints["values_matrix"] = peak_and_valley_values_matrix
            peak_valley_constraints["indicator_matrix"] = (
                peak_and_valley_indicator_matrix
            )

            equality_constraints["peak and valley"] = peak_valley_constraints
        elif constraint_name == "ohlc":
            # the OHLC constraint needs the stocks scaler; set STOCKS_DATASET_DIR to the stocks data directory
            dataset_dir = os.environ.get("STOCKS_DATASET_DIR", "")
            dataset_dir_exists = os.path.exists(dataset_dir)
            if not dataset_dir_exists:
                raise ValueError("Stocks dataset directory not found. Set the STOCKS_DATASET_DIR environment variable to the stocks data directory (it must contain scaler.pkl).")
            scaler = pkl.load(open(f"{dataset_dir}/scaler.pkl", "rb"))
            open_mean = scaler.mean_[0]
            open_std = scaler.scale_[0]
            high_mean = scaler.mean_[1]
            high_std = scaler.scale_[1]
            low_mean = scaler.mean_[2]
            low_std = scaler.scale_[2]
            close_mean = scaler.mean_[3]
            close_std = scaler.scale_[3]
            ohlc_constraints = {
                "open_mean": open_mean,
                "high_mean": high_mean,
                "low_mean": low_mean,
                "close_mean": close_mean,
                "open_std": open_std,
                "high_std": high_std,
                "low_std": low_std,
                "close_std": close_std,
            }

            open_val = batch_samples_numpy[:, 0].flatten() * open_std + open_mean
            high_val = batch_samples_numpy[:, 1].flatten() * high_std + high_mean
            low_val = batch_samples_numpy[:, 2].flatten() * low_std + low_mean
            close_val = batch_samples_numpy[:, 3].flatten() * close_std + close_mean
            assert np.all(np.max(open_val - high_val) < 1e-2)
            assert np.all(np.max(close_val - high_val) < 1e-2)
            assert np.all(np.max(low_val - open_val) < 1e-2)
            assert np.all(np.max(low_val - close_val) < 1e-2)
            assert np.all(np.max(low_val - high_val) < 1e-2)
            """
            We don't need to worry about this discrepancy. 
            The stored scaled values are slightly different from the actual values (order of 1e-7).
            Therefore, when the values are scaled back, the discrepancy is slightly larger (order of 1e-2).
            But, we know the mean and std for each channel, and that is what we are using to enforce the constraints.
            """
            equality_constraints["ohlc"] = ohlc_constraints
        elif "autocorr" in constraint_name:
            lag = int(constraint_name.split("_")[-1])
            autocorr_values = np.zeros((B, C))
            for sample_idx in range(B):
                for channel_idx in range(C):
                    timeseries = batch_samples_numpy[sample_idx, channel_idx]
                    channel_mean = np.mean(timeseries)
                    channel_variance = np.var(timeseries)
                    autocorr_values[sample_idx, channel_idx] = np.mean(
                        (timeseries[:-lag] - channel_mean)
                        * (timeseries[lag:] - channel_mean)
                    ) / channel_variance
            equality_constraints[constraint_name] = autocorr_values
            
    return equality_constraints


"""
Function for guidance based projection
"""


def zeroify_gradient(tensor):
    if tensor.grad is not None:
        tensor.grad.zero_()
    return tensor

 
def obtain_constraint_violation(
    noisy_sample,
    noise_est,
    current_alpha_bar,
    constraints,
    unit_test_mode=False,
    verbose=False,
): 
    constraints_keys = list(constraints.keys())
    # assert noisy_sample.requires_grad

    if unit_test_mode:
        samples_batch_estimate = noisy_sample
    else:
        samples_batch_estimate = (1 / (current_alpha_bar**0.5 + 1e-8)) * (
            noisy_sample - noise_est * (1 - current_alpha_bar) ** 0.5
        )

    # now we obtain the constraint violation
    constraint_violation_batch = torch.zeros(samples_batch_estimate.shape[0]).to(
        samples_batch_estimate.device
    )  # B
    for constraint_key in constraints_keys:
        if constraint_key == "mean":
            mean_constraint = torch.tensor(constraints["mean"]).to(
                samples_batch_estimate.device
            )
            constraint_violation_batch += obtain_mean_constraint_violation(
                mean_constraint, samples_batch_estimate
            )
        elif constraint_key == "mean change":
            mean_change_constraint = torch.tensor(constraints["mean change"]).to(
                samples_batch_estimate.device
            )
            constraint_violation_batch += obtain_mean_change_constraint_violation(
                mean_change_constraint, samples_batch_estimate
            )
        elif "val_at" in constraint_key:
            loc = int(constraint_key.split("_")[-1]) - 1
            value_at_timestep_constraint = torch.tensor(constraints[constraint_key]).to(
                samples_batch_estimate.device
            )
            constraint_violation_batch += obtain_value_at_constraint_violation(
                value_at_timestep_constraint, samples_batch_estimate, loc
            )

        elif constraint_key == "argmax":
            argmax_constraint = torch.tensor(constraints["argmax"]).to(
                samples_batch_estimate.device
            )
            constraint_violation_batch += obtain_argmax_constraint_violation(
                argmax_constraint, samples_batch_estimate
            )

        elif constraint_key == "argmin":
            argmin_constraint = torch.tensor(constraints["argmin"]).to(
                samples_batch_estimate.device
            )
            constraint_violation_batch += obtain_argmin_constraint_violation(
                argmin_constraint, samples_batch_estimate
            )

        elif constraint_key == "max and argmax":
            argmax_constraint = torch.tensor(
                constraints["max and argmax"]["argmax"]
            ).to(samples_batch_estimate.device)
            max_constraint = torch.tensor(constraints["max and argmax"]["max"]).to(
                samples_batch_estimate.device
            )
            constraint_violation_batch += obtain_max_and_argmax_constraint_violation(
                argmax_constraint, max_constraint, samples_batch_estimate
            )

        elif constraint_key == "min and argmin":
            argmin_constraint = torch.tensor(
                constraints["min and argmin"]["argmin"]
            ).to(samples_batch_estimate.device)
            min_constraint = torch.tensor(constraints["min and argmin"]["min"]).to(
                samples_batch_estimate.device
            )
            constraint_violation_batch += obtain_min_and_argmin_constraint_violation(
                argmin_constraint, min_constraint, samples_batch_estimate
            )

        elif constraint_key == "peak and valley":
            peak_and_valley_constraint_dict = constraints["peak and valley"]
            constraint_violation_batch += obtain_peak_and_valley_constraint_violation(
                peak_and_valley_constraint_dict, samples_batch_estimate
            )

        elif constraint_key == "ohlc":
            ohlc_constraint_dict = constraints["ohlc"]
            constraint_violation_batch += obtain_ohlc_constraint_violation(
                ohlc_constraint_dict, samples_batch_estimate
            )
        
        elif "autocorr" in constraint_key:
            # print("here")
            lag = int(constraint_key.split("_")[-1])
            autocorr_constraint = torch.tensor(constraints[constraint_key]).to(
                samples_batch_estimate.device
            )
            autocorr_constraint_violation = obtain_autocorr_constraint_violation(
                autocorr_constraint, samples_batch_estimate, lag
            )
            constraint_violation_batch += autocorr_constraint_violation
            
            # print(f"autocorr_constraint_violation: {autocorr_constraint_violation}")

    return constraint_violation_batch


def project_to_fixed_value_constraints(noisy_sample, constraints):
    constraints_keys = list(constraints.keys())
    for constraint_key in constraints_keys:
        if "val_at" in constraint_key:
            loc = int(constraint_key.split("_")[-1]) - 1
            value_at_timestep_constraint = torch.tensor(constraints[constraint_key]).to(
                noisy_sample.device
            )
            noisy_sample[:, :, loc] = value_at_timestep_constraint

        elif constraint_key == "max and argmax":
            argmax_constraint = torch.tensor(
                constraints["max and argmax"]["argmax"]
            ).to(noisy_sample.device)
            indices_to_set = argmax_constraint.unsqueeze(-1)
            max_constraint = torch.tensor(constraints["max and argmax"]["max"]).to(
                noisy_sample.device
            )
            noisy_sample = noisy_sample.scatter(
                2, indices_to_set, max_constraint.unsqueeze(-1)
            )

        elif constraint_key == "min and argmin":
            argmin_constraint = torch.tensor(
                constraints["min and argmin"]["argmin"]
            ).to(noisy_sample.device)
            indices_to_set = argmin_constraint.unsqueeze(-1)
            min_constraint = torch.tensor(constraints["min and argmin"]["min"]).to(
                noisy_sample.device
            )
            noisy_sample = noisy_sample.scatter(
                2, indices_to_set, min_constraint.unsqueeze(-1)
            )

        elif constraint_key == "peak and valley":
            values_matrix = (
                torch.tensor(constraints["peak and valley"]["values_matrix"])
                .float()
                .to(noisy_sample.device, dtype=noisy_sample.dtype)
            )
            indicator_matrix = (
                torch.tensor(constraints["peak and valley"]["indicator_matrix"])
                .float()
                .to(noisy_sample.device, dtype=noisy_sample.dtype)
            )
            noisy_sample = noisy_sample * (1 - indicator_matrix) + values_matrix

    return noisy_sample


"""
Functions to implement Constrained Optimization for Time Series Generation
"""


def scale_discriminator_input(
    sample, discrete_conditions, continuous_conditions, discriminator_input_scaler
):
    input_dict = {
        "timeseries": sample,
        "discrete_conditions": discrete_conditions,
        "continuous_conditions": continuous_conditions,
    }

    scaled_input_dict = discriminator_input_scaler.convert_from_normal_to_gan(
        input_dict
    )
    return (
        scaled_input_dict["timeseries"],
        scaled_input_dict["discrete_conditions"],
        scaled_input_dict["continuous_conditions"],
    )


def scale_timeseries_gradient(scaled_gradient, discriminator_input_scaler):
    return discriminator_input_scaler.convert_gradient_from_gan_to_normal(
        scaled_gradient
    )


def get_discriminator_value(
    sample,
    discrete_condn,
    continuous_condn,
    num_channels,
    horizon,
    synthesis_config,
    verbose=True,
):
    # get the discriminator
    discriminator = synthesis_config["discriminator_wrapper"].synthesizer.discriminator
    discriminator.eval()
    # get the discriminator device
    device = synthesis_config["discriminator_wrapper"].device
    # get the discriminator_input_scaler
    discriminator_input_scaler = synthesis_config["discriminator_input_scaler"]

    # make the inputs to have batch size of 1
    reshaped_sample = np.expand_dims(sample.reshape(num_channels, horizon), axis=0)
    reshaped_discrete_condn = np.expand_dims(discrete_condn, axis=0)
    reshaped_continuous_condn = np.expand_dims(continuous_condn, axis=0)

    # scale the discriminator input if necessary
    if discriminator_input_scaler is not None:
        if verbose:
            print("Scaling discriminator input")
        reshaped_sample, reshaped_discrete_condn, reshaped_continuous_condn = (
            scale_discriminator_input(
                reshaped_sample,
                reshaped_discrete_condn,
                reshaped_continuous_condn,
                discriminator_input_scaler,
            )
        )

    # convert the inputs to tensors
    reshaped_sample_tensor = torch.tensor(reshaped_sample).float().to(device)
    reshaped_discrete_condn_tensor = (
        torch.tensor(reshaped_discrete_condn).float().to(device)
    )
    reshaped_continuous_condn_tensor = (
        torch.tensor(reshaped_continuous_condn).float().to(device)
    )

    # obtain the discriminator value
    D_real = discriminator(
        x=reshaped_sample_tensor,
        y=reshaped_discrete_condn_tensor,
        z=reshaped_continuous_condn_tensor,
    )

    discriminator_value = D_real.cpu().detach().numpy().sum()
    if verbose:
        print(f"Discriminator value: {discriminator_value}")
    return discriminator_value


def get_discriminator_gradient(
    sample,
    discrete_condn,
    continuous_condn,
    num_channels,
    horizon,
    synthesis_config,
):
    # get the discriminator
    discriminator = synthesis_config["discriminator_wrapper"].synthesizer.discriminator
    discriminator.eval()
    # get the discriminator device
    device = synthesis_config["discriminator_wrapper"].device
    # get the discriminator_input_scaler
    discriminator_input_scaler = synthesis_config["discriminator_input_scaler"]
 
    # make the inputs to have batch size of 1
    reshaped_sample = np.expand_dims(sample.reshape(num_channels, horizon), axis=0)
    reshaped_discrete_condn = np.expand_dims(discrete_condn, axis=0)
    reshaped_continuous_condn = np.expand_dims(continuous_condn, axis=0)

    # scale the discriminator input if necessary
    if discriminator_input_scaler is not None:
        reshaped_sample, reshaped_discrete_condn, reshaped_continuous_condn = (
            scale_discriminator_input(
                reshaped_sample,
                reshaped_discrete_condn,
                reshaped_continuous_condn,
                discriminator_input_scaler,
            )
        )

    # convert the inputs to tensors
    reshaped_sample_tensor = torch.tensor(reshaped_sample).float().to(device)
    reshaped_discrete_condn_tensor = (
        torch.tensor(reshaped_discrete_condn).float().to(device)
    )
    reshaped_continuous_condn_tensor = (
        torch.tensor(reshaped_continuous_condn).float().to(device)
    )

    reshaped_sample_tensor.requires_grad = True

    # obtain the discriminator value
    D_real = discriminator(
        x=reshaped_sample_tensor,
        y=reshaped_discrete_condn_tensor,
        z=reshaped_continuous_condn_tensor,
    )

    # obtain the gradient
    D_real.backward()
    scaled_gradient = reshaped_sample_tensor.grad.cpu().detach().numpy()

    # scale the gradient back to normal
    if discriminator_input_scaler is not None:
        gradient = scale_timeseries_gradient(
            scaled_gradient, discriminator_input_scaler
        )
    else:
        gradient = scaled_gradient

    # flatten the gradient
    gradient = gradient[0].flatten()

    return gradient


def get_constraints_list(
    sample_idx,
    num_channels,
    horizon,
    constraints_to_be_executed,
    tolerance,
    keep_feasible=False,
):
    constraints = []
    constraints_keys = list(constraints_to_be_executed.keys())
    num_elems = num_channels * horizon
    for channel_idx in range(num_channels):
        start_loc = channel_idx * horizon
        end_loc = (channel_idx + 1) * horizon
        for constraint_key in constraints_keys:

            if constraint_key == "argmin":
                argmin_loc = constraints_to_be_executed["argmin"][sample_idx][
                    channel_idx
                ]
                argmin_constraint = obtain_scipy_argmin_constraint(
                    argmin_loc, horizon, num_elems, start_loc, end_loc, keep_feasible
                )
                constraints.append(argmin_constraint)

            elif constraint_key == "argmax":
                argmax_loc = constraints_to_be_executed["argmax"][sample_idx][
                    channel_idx
                ]
                argmax_constraint = obtain_scipy_argmax_constraint(
                    argmax_loc, horizon, num_elems, start_loc, end_loc, keep_feasible
                )
                constraints.append(argmax_constraint)

            elif constraint_key == "min and argmin":
                min_val = constraints_to_be_executed["min and argmin"]["min"][
                    sample_idx
                ][channel_idx]
                argmin_loc = constraints_to_be_executed["min and argmin"]["argmin"][
                    sample_idx
                ][channel_idx]
                argmin_constraint = obtain_scipy_argmin_constraint(
                    argmin_loc, horizon, num_elems, start_loc, end_loc, keep_feasible
                )
                min_constraint = obtain_scipy_value_at_constraint(
                    min_val, num_elems, start_loc, argmin_loc, tolerance, keep_feasible
                )
                constraints.append(argmin_constraint)
                constraints.append(min_constraint)

            elif constraint_key == "max and argmax":
                max_val = constraints_to_be_executed["max and argmax"]["max"][
                    sample_idx
                ][channel_idx]
                argmax_loc = constraints_to_be_executed["max and argmax"]["argmax"][
                    sample_idx
                ][channel_idx]
                argmax_constraint = obtain_scipy_argmax_constraint(
                    argmax_loc, horizon, num_elems, start_loc, end_loc, keep_feasible
                )
                max_constraint = obtain_scipy_value_at_constraint(
                    max_val, num_elems, start_loc, argmax_loc, tolerance, keep_feasible
                )
                constraints.append(argmax_constraint)
                constraints.append(max_constraint)

            elif constraint_key == "mean":
                meanval = constraints_to_be_executed["mean"][sample_idx][channel_idx]
                mean_constraint = obtain_scipy_mean_constraint(
                    meanval,
                    horizon,
                    num_elems,
                    start_loc,
                    end_loc,
                    tolerance,
                    keep_feasible,
                )
                constraints.append(mean_constraint)

            elif "val_at" in constraint_key:
                loc = int(constraint_key.split("_")[-1]) - 1
                value_at_timestep = constraints_to_be_executed[constraint_key][
                    sample_idx
                ][channel_idx]
                val_at_constraint = obtain_scipy_value_at_constraint(
                    value_at_timestep,
                    num_elems,
                    start_loc,
                    loc,
                    tolerance,
                    keep_feasible,
                )
                constraints.append(val_at_constraint)

            elif constraint_key == "mean change":
                mean_change_val = constraints_to_be_executed["mean change"][sample_idx][
                    channel_idx
                ]
                mean_change_constraint = obtain_scipy_mean_change_constraint(
                    mean_change_val,
                    horizon,
                    num_elems,
                    start_loc,
                    end_loc,
                    tolerance,
                    keep_feasible,
                )
                constraints.append(mean_change_constraint)

            elif constraint_key == "peak and valley":
                num_elems = num_channels * horizon
                trend_constraint_matrix = constraints_to_be_executed[constraint_key][
                    "trend_constraint_matrix"
                ]
                values_matrix = constraints_to_be_executed[constraint_key][
                    "values_matrix"
                ]
                indicator_matrix = constraints_to_be_executed[constraint_key][
                    "indicator_matrix"
                ]
                channel_trend_constraint_matrix = trend_constraint_matrix[
                    sample_idx, channel_idx
                ]
                channel_values_matrix = values_matrix[sample_idx, channel_idx]
                channel_indicator_matrix = indicator_matrix[sample_idx, channel_idx]

                peak_valley_constraint, trend_constraint = (
                    obtain_scipy_peak_and_valley_constraint(
                        channel_trend_constraint_matrix,
                        channel_values_matrix,
                        channel_indicator_matrix,
                        horizon,
                        num_elems,
                        start_loc,
                        end_loc,
                        tolerance,
                        keep_feasible,
                    )
                )
                constraints.append(peak_valley_constraint)
                constraints.append(trend_constraint)
                
            elif "autocorr" in constraint_key:
                lag = int(constraint_key.split("_")[-1])
                autocorr_constraint_value = constraints_to_be_executed[constraint_key][sample_idx][channel_idx]
                autocorr_constraint = obtain_scipy_autocorr_constraint(
                    autocorr_constraint_value, start_loc, end_loc, lag, tolerance, keep_feasible
                )
                constraints.append(autocorr_constraint)

    if "ohlc" in constraints_keys:
        ohlc_constraint_dict = constraints_to_be_executed["ohlc"]
        (
            open_high_constraint,
            close_high_constraint,
            low_open_constraint,
            low_close_constraint,
        ) = obtain_scipy_ohlc_constraint(
            ohlc_constraint_dict, horizon, num_elems, keep_feasible
        )
        constraints.append(open_high_constraint)
        constraints.append(close_high_constraint)
        constraints.append(low_open_constraint)
        constraints.append(low_close_constraint)

    return constraints


def constrained_optimization_for_ts_generation(
    per_sample_projection_requirements, verbose=False
):
    synthetic_sample = per_sample_projection_requirements["synthetic_sample"]
    warm_start_sample = per_sample_projection_requirements["warm_start_sample"]
    sample_idx = per_sample_projection_requirements["sample_idx"]
    discrete_condn_input = per_sample_projection_requirements["discrete_condn_input"]
    continuous_condn_input = per_sample_projection_requirements[
        "continuous_condn_input"
    ]
    constraints_to_be_executed = per_sample_projection_requirements[
        "constraints_to_be_executed"
    ]
    synthesis_config = per_sample_projection_requirements["synthesis_config"]
    num_channels = synthetic_sample.shape[0]
    horizon = synthetic_sample.shape[1]

    print(f"Processing sample {sample_idx}")

    get_discriminator_value_for_sample = partial(
        get_discriminator_value,
        discrete_condn=discrete_condn_input,
        continuous_condn=continuous_condn_input,
        num_channels=num_channels,
        horizon=horizon,
        synthesis_config=synthesis_config,
        verbose=verbose,
    )
    get_discriminator_gradient_for_sample = partial(
        get_discriminator_gradient,
        discrete_condn=discrete_condn_input,
        continuous_condn=continuous_condn_input,
        num_channels=num_channels,
        horizon=horizon,
        synthesis_config=synthesis_config,
    )

    factor = synthesis_config["discriminator_weight"]

    if synthesis_config["using_real_seed"]:
        objective = lambda x: -np.mean(
            np.square(x - synthetic_sample.flatten())
        ) - factor * get_discriminator_value_for_sample(x)
        # this is the COP objective, we need to maximize the distance from the real sample while maximizing the discriminator value
        objective_gradient_fn = lambda x: -(2 / (num_channels * horizon)) * (
            x - synthetic_sample.flatten()
        ) - factor * get_discriminator_gradient_for_sample(x)
    else:
        objective = lambda x: np.mean(
            np.square(x - synthetic_sample.flatten())
        ) - factor * get_discriminator_value_for_sample(x)
        # this is the COP finetuning objective, we need to minimize the distance from the synthetic sample while maximizing the discriminator value
        objective_gradient_fn = lambda x: (2 / (num_channels * horizon)) * (
            x - synthetic_sample.flatten()
        ) - factor * get_discriminator_gradient_for_sample(x)

    tolerances = [ACCEPTED_TOERANCE / 2, ACCEPTED_TOERANCE]

    for tol in tolerances:
        constraints = get_constraints_list(
            sample_idx,
            num_channels,
            horizon,
            constraints_to_be_executed,
            tol,
        )
        assert len(constraints) > 0
        x0 = warm_start_sample.flatten()
        if synthesis_config["using_real_seed"]:
            # observed that things get out of control here
            # note that we cannot impose bounds the same way as given in the COP paper
            # as they will be violated by the constraints
            bounds = [(-10, 10)] * len(x0)
        else:
            bounds = None
        start_time = time()
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            jac=objective_gradient_fn,
            bounds=bounds,
        )
        end_time = time()
        print(f"Time taken for optimization: {end_time - start_time}")
        projected_sample = result.x.reshape(synthetic_sample.shape)
        if result.success:
            break
        else:
            warm_start_sample = projected_sample

    return (projected_sample, sample_idx)


def project_all_samples_to_equality_constraints_with_scipy(
    synthetic,
    warm_start,
    constraints,
    discrete_conditions,
    continuous_conditions,
    synthesis_config,
):
    per_sample_projection_requirements_list = []
    for sample_idx in range(synthetic.shape[0]):
        synthetic_sample = synthetic[sample_idx]
        discrete_condn_input = discrete_conditions[sample_idx]
        continuous_condn_input = continuous_conditions[sample_idx]
        per_sample_projection_requirements = {
            "synthetic_sample": synthetic_sample,
            "discrete_condn_input": discrete_condn_input,
            "continuous_condn_input": continuous_condn_input,
            "constraints_to_be_executed": constraints,
            "sample_idx": sample_idx,
            "synthesis_config": synthesis_config,
        }

        if warm_start is None:
            per_sample_projection_requirements["warm_start_sample"] = synthetic_sample
        else:
            per_sample_projection_requirements["warm_start_sample"] = warm_start[
                sample_idx
            ]

        per_sample_projection_requirements_list.append(
            per_sample_projection_requirements
        )
        
    if synthetic.shape[0] == 1:
        result = constrained_optimization_for_ts_generation(per_sample_projection_requirements_list[0])
        projected_timeseries = np.expand_dims(result[0], axis=0)
        return projected_timeseries
    else:
        pool = mp.Pool(int(mp.cpu_count() / 4))
        results = pool.map(
            constrained_optimization_for_ts_generation,
            per_sample_projection_requirements_list,
        )
        pool.close()
        # sort the results based on the second element of the tuple
        results = sorted(results, key=lambda x: x[1])
        projected_timeseries_list = [result[0] for result in results]

        projected_timeseries = np.stack(projected_timeseries_list)

        return projected_timeseries


def project_all_samples_to_equality_constraints_with_scipy_single_threaded(
    synthetic,
    warm_start,
    constraints,
    discrete_conditions,
    continuous_conditions,
    synthesis_config,
):
    projected_timeseries_list = []
    for sample_idx in range(synthetic.shape[0]):
        synthetic_sample = synthetic[sample_idx]
        discrete_condn_input = discrete_conditions[sample_idx]
        continuous_condn_input = continuous_conditions[sample_idx]
        per_sample_projection_requirements = {
            "synthetic_sample": synthetic_sample,
            "discrete_condn_input": discrete_condn_input,
            "continuous_condn_input": continuous_condn_input,
            "constraints_to_be_executed": constraints,
            "sample_idx": sample_idx,
            "synthesis_config": synthesis_config,
        }

        if warm_start is None:
            per_sample_projection_requirements["warm_start_sample"] = synthetic_sample
        else:
            per_sample_projection_requirements["warm_start_sample"] = warm_start[
                sample_idx
            ]

        result = constrained_optimization_for_ts_generation(per_sample_projection_requirements)
        projected_timeseries_list.append(result[0])


    projected_timeseries = np.stack(projected_timeseries_list)
    return projected_timeseries

"""
Function to project the sample to the convex equality constraints using CVXPY
"""

def project_sample_to_equality_constraints_cvxpy(
    per_sample_projection_input_dict,
):
    sample_idx = per_sample_projection_input_dict["sample_idx"]
    constraints_to_be_executed = per_sample_projection_input_dict[
        "constraints_to_be_executed"
    ]
    warm_start_sample = per_sample_projection_input_dict["warm_start_sample"]
    sample = per_sample_projection_input_dict["sample"]
    horizon = sample.shape[-1]
    num_channels = sample.shape[-2]
    constraints_keys = list(constraints_to_be_executed.keys())

    flattened_sample = sample.flatten()
    random_channel_idx = np.random.randint(num_channels)
    assert (
        sample[random_channel_idx]
        == flattened_sample[
            random_channel_idx * horizon : (random_channel_idx + 1) * horizon
        ]
    ).all()
    opt_var = cp.Variable(flattened_sample.shape)
    if warm_start_sample is not None:
        opt_var.value = warm_start_sample.flatten()
    else:
        opt_var.value = flattened_sample

    objective_function = cp.Minimize(cp.norm(flattened_sample - opt_var) ** 2.0)

    tolerance = 5e-3

    constraints = []
    for channel_idx in range(num_channels):
        start_idx = channel_idx * horizon
        end_idx = (channel_idx + 1) * horizon
        for constraint_key in constraints_keys:
            if constraint_key == "argmax":
                argmax = constraints_to_be_executed["argmax"][sample_idx][channel_idx]
                constraints.append(
                    opt_var[start_idx:end_idx] <= opt_var[start_idx + argmax]
                )
            elif constraint_key == "max and argmax":
                argmax = constraints_to_be_executed["max and argmax"]["argmax"][
                    sample_idx
                ][channel_idx]
                maxval = constraints_to_be_executed["max and argmax"]["max"][
                    sample_idx
                ][channel_idx]
                # constraints.append(
                #     opt_var[start_idx:end_idx] <= opt_var[start_idx + argmax]
                # )
                constraints.append(opt_var[start_idx + argmax] <= maxval + tolerance)
                constraints.append(opt_var[start_idx + argmax] >= maxval - tolerance)
            elif constraint_key == "argmin":
                argmin = constraints_to_be_executed["argmin"][sample_idx][channel_idx]
                constraints.append(
                    opt_var[start_idx:end_idx] >= opt_var[start_idx + argmin]
                )
            elif constraint_key == "min and argmin":
                argmin = constraints_to_be_executed["min and argmin"]["argmin"][
                    sample_idx
                ][channel_idx]
                minval = constraints_to_be_executed["min and argmin"]["min"][
                    sample_idx
                ][channel_idx]
                # constraints.append(
                #     opt_var[start_idx:end_idx] >= opt_var[start_idx + argmin]
                # )
                constraints.append(opt_var[start_idx + argmin] <= minval + tolerance)
                constraints.append(opt_var[start_idx + argmin] >= minval - tolerance)
            elif constraint_key == "mean":
                meanval = constraints_to_be_executed["mean"][sample_idx][channel_idx]
                constraints.append(
                    cp.mean(opt_var[start_idx:end_idx]) <= meanval + tolerance
                )
                constraints.append(
                    cp.mean(opt_var[start_idx:end_idx]) >= meanval - tolerance
                )
            elif constraint_key == "mean change":
                mean_change = constraints_to_be_executed["mean change"][sample_idx][
                    channel_idx
                ]
                constraints.append(
                    cp.mean(cp.diff(opt_var[start_idx:end_idx]))
                    <= mean_change + tolerance
                )
                constraints.append(
                    cp.mean(cp.diff(opt_var[start_idx:end_idx]))
                    >= mean_change - tolerance
                )
            elif "val_at" in constraint_key:
                loc = int(constraint_key.split("_")[-1]) - 1
                value_at_timestep = constraints_to_be_executed[constraint_key][
                    sample_idx
                ][channel_idx]
                constraints.append(
                    opt_var[start_idx + loc] <= value_at_timestep + tolerance
                )
                constraints.append(
                    opt_var[start_idx + loc] >= value_at_timestep - tolerance
                )

    if "ohlc" in constraints_keys:
        ohlc = constraints_to_be_executed["ohlc"]
        open_mean = ohlc["open_mean"]
        high_mean = ohlc["high_mean"]
        low_mean = ohlc["low_mean"]
        close_mean = ohlc["close_mean"]
        open_std = ohlc["open_std"]
        high_std = ohlc["high_std"]
        low_std = ohlc["low_std"]
        close_std = ohlc["close_std"]
        open_unscaled = opt_var[:horizon] * open_std + open_mean
        high_unscaled = opt_var[horizon : 2 * horizon] * high_std + high_mean
        low_unscaled = opt_var[2 * horizon : 3 * horizon] * low_std + low_mean
        close_unscaled = opt_var[3 * horizon : 4 * horizon] * close_std + close_mean

        constraints.append(open_unscaled <= high_unscaled)
        constraints.append(close_unscaled <= high_unscaled)
        constraints.append(low_unscaled <= open_unscaled)
        constraints.append(low_unscaled <= close_unscaled)
        # constraints.append(low_unscaled <= high_unscaled)
    

    problem = cp.Problem(objective_function, constraints)
    
    if len(constraints_keys) < 3:
        problem.solve(warm_start=True, verbose=False)
    else:
        try:
            problem.solve(cp.ECOS, warm_start=True, verbose=False)
        except:
            problem.solve(warm_start=True, verbose=False)
    
    
    # problem.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        
    sol = opt_var.value
    projected_sample = sol.reshape(sample.shape)
    projected_sample = projected_sample.astype(np.float32)
    return (projected_sample, sample_idx)

def project_all_samples_to_equality_constraints(
    sample_estimate_batch,
    constraints,
    warm_start_samples,
    penalty_coefficient=0,
    projection_method="strict",
):
    per_sample_projection_inputs_list = []
    for sample_idx in range(sample_estimate_batch.shape[0]):
        sample_estimate = sample_estimate_batch[sample_idx]
        per_sample_projection_input_dict = {
            "sample": sample_estimate,
            "constraints_to_be_executed": constraints,
            "sample_idx": sample_idx,
            "penalty_coefficient": penalty_coefficient,
            "warm_start_sample": warm_start_samples[sample_idx],
        }
        per_sample_projection_inputs_list.append(per_sample_projection_input_dict)
        
    if sample_estimate_batch.shape[0] == 1: 
        if projection_method == "strict":
            result = project_sample_to_equality_constraints_cvxpy(per_sample_projection_inputs_list[0])
        elif projection_method == "penalty_based":
            list_of_constraints = list(constraints.keys())
            if "autocorr_12" in list_of_constraints:
                result = project_sample_to_minimize_scipy_penalty(per_sample_projection_inputs_list[0])
            else:
                result = project_sample_to_minimize_penalty(per_sample_projection_inputs_list[0])
            
        return np.expand_dims(result[0], axis=0)
    
    else:
        pool = mp.Pool(int(mp.cpu_count() / 4))

        if projection_method == "strict":
            project_sample_to_equality_constraints_fn = project_sample_to_equality_constraints_cvxpy
        elif projection_method == "penalty_based":
            list_of_constraints = list(constraints.keys())  
            if "autocorr_12" in list_of_constraints:
                project_sample_to_equality_constraints_fn = project_sample_to_minimize_scipy_penalty
            else:
                project_sample_to_equality_constraints_fn = project_sample_to_minimize_penalty
        results = pool.map(
            project_sample_to_equality_constraints_fn,
            per_sample_projection_inputs_list,
        )
        pool.close()
        # sort the results based on the second element of the tuple
        results = sorted(results, key=lambda x: x[1])
        projected_timeseries_list = [result[0] for result in results]
        # constraint_violation_list = [result[2] for result in results]
        
        # if penalty_coefficient > 999:
        #     print(f"Constraint violation: {constraint_violation_list}")

        projected_timeseries = np.stack(projected_timeseries_list)

        return projected_timeseries


"""
Penalty based optimization
"""


def get_sample_est_from_noisy_sample(noisy_sample, noise_est, current_alpha_bar):
    sample_est = (1 / (current_alpha_bar**0.5 + 1e-8)) * (
        noisy_sample - noise_est * (1 - current_alpha_bar) ** 0.5
    )
    return sample_est


def obtain_max_and_argmax_penalty_cvxpy(
    opt_var, start_idx, argmax_constraint, max_constraint, tolerance
):
    return cp.maximum(
        cp.abs(opt_var[start_idx + argmax_constraint] - max_constraint) - tolerance, 0
    )


def obtain_min_and_argmin_penalty_cvxpy(
    opt_var, start_idx, argmin_constraint, min_constraint, tolerance
):
    return cp.maximum(
        cp.abs(opt_var[start_idx + argmin_constraint] - min_constraint) - tolerance, 0
    )


def obtain_penalty(opt_var, penalty_functions_to_use):
    total_penalty = 0
    for penalty_function in penalty_functions_to_use:
        penalty = penalty_function(opt_var)
        total_penalty += penalty
    return total_penalty


def project_sample_to_minimize_penalty(
    per_sample_projection_input_dict, solver_type=None
):
    sample = per_sample_projection_input_dict["sample"]
    sample_idx = per_sample_projection_input_dict["sample_idx"]
    constraints_to_be_executed = per_sample_projection_input_dict[
        "constraints_to_be_executed"
    ]
    penalty_coefficient = per_sample_projection_input_dict["penalty_coefficient"]
    warm_start_sample = per_sample_projection_input_dict["warm_start_sample"]
    constraints_keys = list(constraints_to_be_executed.keys())

    num_channels = sample.shape[0]
    horizon = sample.shape[1]
    flattened_sample = sample.flatten()
    projected_sample = np.copy(sample)

    tolerance = 5e-3
    penalty_functions_to_use = []
    constraints_keys = list(constraints_to_be_executed.keys())
    for channel_idx in range(num_channels):
        # Add penalty functions
        for constraint_key in constraints_keys:
            if constraint_key == "mean":
                start_idx = channel_idx * horizon
                end_idx = (channel_idx + 1) * horizon
                mean_constraint = constraints_to_be_executed["mean"][sample_idx][
                    channel_idx
                ]
                mean_penalty_function = partial(
                    obtain_mean_penalty_cvxpy,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    mean_constraint=mean_constraint,
                    tolerance=tolerance,
                )
                penalty_functions_to_use.append(mean_penalty_function)

            elif constraint_key == "mean change":
                start_idx = channel_idx * horizon
                end_idx = (channel_idx + 1) * horizon
                mean_change_constraint = constraints_to_be_executed["mean change"][
                    sample_idx
                ][channel_idx]
                mean_change_penalty_function = partial(
                    obtain_mean_change_penalty_cvxpy,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    mean_change_constraint=mean_change_constraint,
                    tolerance=tolerance,
                )
                penalty_functions_to_use.append(mean_change_penalty_function)

            elif constraint_key == "argmin":
                start_idx = channel_idx * horizon
                end_idx = (channel_idx + 1) * horizon
                argmin_constraint = constraints_to_be_executed["argmin"][sample_idx][
                    channel_idx
                ]
                argmin_penalty_function = partial(
                    obtain_argmin_penalty_cvxpy,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    argmin_constraint=argmin_constraint,
                )
                penalty_functions_to_use.append(argmin_penalty_function)

            elif constraint_key == "argmax":
                start_idx = channel_idx * horizon
                end_idx = (channel_idx + 1) * horizon
                argmax_constraint = constraints_to_be_executed["argmax"][sample_idx][
                    channel_idx
                ]
                argmax_penalty_function = partial(
                    obtain_argmax_penalty_cvxpy,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    argmax_constraint=argmax_constraint,
                )
                penalty_functions_to_use.append(argmax_penalty_function)

            elif "val_at" in constraint_key:
                loc = int(constraint_key.split("_")[-1]) - 1
                start_idx = channel_idx * horizon
                value_at_constraint = constraints_to_be_executed[constraint_key][
                    sample_idx
                ][channel_idx]
                value_at_penalty_function = partial(
                    obtain_value_at_penalty_cvxpy,
                    start_idx=start_idx,
                    loc=loc,
                    value_at_constraint=value_at_constraint,
                    tolerance=tolerance,
                )
                penalty_functions_to_use.append(value_at_penalty_function)

            elif constraint_key == "max and argmax":
                start_idx = channel_idx * horizon
                max_val_constraint = constraints_to_be_executed["max and argmax"][
                    "max"
                ][sample_idx][channel_idx]
                argmax_constraint = constraints_to_be_executed["max and argmax"][
                    "argmax"
                ][sample_idx][channel_idx]
                max_penalty_function = partial(
                    obtain_value_at_penalty_cvxpy,
                    start_idx=start_idx,
                    loc=argmax_constraint,
                    value_at_constraint=max_val_constraint,
                    tolerance=tolerance,
                )
                penalty_functions_to_use.append(max_penalty_function)

            elif constraint_key == "min and argmin":
                start_idx = channel_idx * horizon
                min_val_constraint = constraints_to_be_executed["min and argmin"][
                    "min"
                ][sample_idx][channel_idx]
                argmin_constraint = constraints_to_be_executed["min and argmin"][
                    "argmin"
                ][sample_idx][channel_idx]
                min_penalty_function = partial(
                    obtain_value_at_penalty_cvxpy,
                    start_idx=start_idx,
                    loc=argmin_constraint,
                    value_at_constraint=min_val_constraint,
                    tolerance=tolerance,
                )
                penalty_functions_to_use.append(min_penalty_function)

            elif constraint_key == "peak and valley":
                peak_and_valley_penalty_function = partial(
                    obtain_peak_and_valley_penalty_cvxpy,
                    sample_idx=sample_idx,
                    channel_idx=channel_idx,
                    constraints_to_be_executed=constraints_to_be_executed,
                    horizon=horizon,
                    tolerance=tolerance,
                )
                penalty_functions_to_use.append(peak_and_valley_penalty_function)

    if "ohlc" in constraints_keys:
        ohlc_penalty_function = partial(
            obtain_ohlc_penalty_cvxpy,
            constraints_to_be_executed=constraints_to_be_executed,
            horizon=horizon,
            tolerance=tolerance,
        )
        penalty_functions_to_use.append(ohlc_penalty_function)

    # Initialize the optimization problem with optimazation variable of size timeseries_channel
    opt_var = cp.Variable(flattened_sample.shape)

    # Add the objective function to the optimization problem
    penalty_fn = partial(
        obtain_penalty, penalty_functions_to_use=penalty_functions_to_use
    )

    objective_function = cp.Minimize(
        cp.norm(flattened_sample - opt_var) ** 2.0
        + penalty_coefficient * penalty_fn(opt_var)
    )

    if warm_start_sample is not None:
        opt_var.value = warm_start_sample.flatten()
    else:
        opt_var.value = flattened_sample
    problem = cp.Problem(objective_function)

    if len(constraints_keys) < 3:
        problem.solve(warm_start=True)
    else:
        try:
            problem.solve(cp.ECOS, warm_start=True)
        except:
            problem.solve(warm_start=True)

    projected_flattened_sample = opt_var.value
    projected_sample = projected_flattened_sample.reshape(sample.shape)
    projected_sample = projected_sample.astype(np.float32)

    return (projected_sample, sample_idx)


def project_sample_to_minimize_scipy_penalty(
    per_sample_projection_input_dict, solver_type=None
):
    
    sample = torch.tensor(per_sample_projection_input_dict["sample"])
    sample = sample.unsqueeze(0)
    
    sample_idx = per_sample_projection_input_dict["sample_idx"]
    constraints_to_be_executed = per_sample_projection_input_dict[
        "constraints_to_be_executed"
    ]
    penalty_coefficient = per_sample_projection_input_dict["penalty_coefficient"]
    warm_start_sample = per_sample_projection_input_dict["warm_start_sample"]
    warm_start_sample = np.expand_dims(warm_start_sample, axis=0)
    constraints_keys = list(constraints_to_be_executed.keys())

    tolerance = 5e-3
    constraints_keys = list(constraints_to_be_executed.keys())
    
    x = torch.nn.Parameter(torch.tensor(warm_start_sample))
    lr = 1e-4
    with torch.enable_grad():
        for iter in range(500):
            optimizer = torch.optim.Adagrad([x], lr=lr)
            optimizer.zero_grad()
            constraint_violation = 0
            for constraint_key in constraints_keys:
                if constraint_key == "ohlc":
                    ohlc_constraint_dict = constraints_to_be_executed["ohlc"]
                    constraint_violation_batch = obtain_ohlc_constraint_violation(
                        ohlc_constraint_dict, x
                    )
                    ohlc_constraint_violation = torch.sum(constraint_violation_batch)
                    constraint_violation += ohlc_constraint_violation
            
                elif "autocorr" in constraint_key:
                    # print("here")
                    lag = int(constraint_key.split("_")[-1])
                    autocorr_constraint = torch.tensor(constraints_to_be_executed[constraint_key][sample_idx]).unsqueeze(0)
                    constraint_violation_batch = obtain_autocorr_constraint_violation(
                        autocorr_constraint, x, lag
                    )
                    autocorr_constraint_violation = torch.sum(constraint_violation_batch)
                    constraint_violation += autocorr_constraint_violation
            
            penalty = torch.max(torch.tensor(0.0), constraint_violation - tolerance)
            objective = torch.mean((x - sample) ** 2) + penalty_coefficient * penalty
            
            objective.backward()
            optimizer.step()
            x = torch.nn.Parameter(x.data)
            
            # update the learning rate
            if iter % 10 == 0:
                lr = lr * 0.5
                
            if constraint_violation <= tolerance:
                # s = x.detach().numpy()[0]
                # open_mean = constraints_to_be_executed["ohlc"]["open_mean"]
                # high_mean = constraints_to_be_executed["ohlc"]["high_mean"]
                # low_mean = constraints_to_be_executed["ohlc"]["low_mean"]
                # close_mean = constraints_to_be_executed["ohlc"]["close_mean"]
                # open_std = constraints_to_be_executed["ohlc"]["open_std"]
                # high_std = constraints_to_be_executed["ohlc"]["high_std"]
                # low_std = constraints_to_be_executed["ohlc"]["low_std"]
                # close_std = constraints_to_be_executed["ohlc"]["close_std"]
                # open_unscaled = s[0] * open_std + open_mean
                # high_unscaled = s[1] * high_std + high_mean
                # low_unscaled = s[2] * low_std + low_mean
                # close_unscaled = s[3] * close_std + close_mean
        
                # oh_violation = np.max(open_unscaled - high_unscaled)
                # lo_violation = np.max(low_unscaled - open_unscaled)
                # lc_violation = np.max(low_unscaled - close_unscaled)
                # ch_violation = np.max(close_unscaled - high_unscaled)
        
                # constraint_violation_mag = 0
                # if oh_violation > ACCEPTED_TOERANCE:
                #     # print(f"constraint violation for oh: {oh_violation}")
                #     constraint_violation_mag += oh_violation
                # if lo_violation > ACCEPTED_TOERANCE:
                #     # print(f"constraint violation for lo: {lo_violation}")
                #     constraint_violation_mag += lo_violation
                # if lc_violation > ACCEPTED_TOERANCE:
                #     # print(f"constraint violation for lc: {lc_violation}")
                #     constraint_violation_mag += lc_violation
                # if ch_violation > ACCEPTED_TOERANCE:
                #     # print(f"constraint violation for ch: {ch_violation}")
                #     constraint_violation_mag += ch_violation
                # # print(f"ohlc constraint violation: {constraint_violation_mag}")
                
                # assert constraint_violation_mag <= ACCEPTED_TOERANCE
                break 
    # print(f"constraint violation: {constraint_violation}")
    projected_sample = x.detach().numpy()[0]

    return (projected_sample, sample_idx)   


def verify_constraint_satisfaction_for_sample(sample, constraints, sample_idx):
    num_channels = sample.shape[0]
    num_constraints = len(constraints.keys())

    constraint_violation_mag = 0
    constraint_violation_matrix = np.zeros((num_channels, num_constraints))
    for channel_idx in range(num_channels):
        timeseries_channel = sample[channel_idx]
        constraints_keys = list(constraints.keys())
        for constraint_id, constraint_key in enumerate(constraints_keys):
            if constraint_key == "argmax":
                argmax = constraints["argmax"][sample_idx][channel_idx]
                constraint_violation = np.max(
                    timeseries_channel - timeseries_channel[argmax]
                )
                if constraint_violation > ACCEPTED_TOERANCE:
                    # print(f"constraint violation for argmax: {constraint_violation}")
                    constraint_violation_mag += constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1
            elif constraint_key == "max and argmax":
                max_val = np.max(timeseries_channel)
                req_max_val = constraints["max and argmax"]["max"][sample_idx][
                    channel_idx
                ]
                constraint_violation = np.abs(max_val - req_max_val)
                if constraint_violation > ACCEPTED_TOERANCE:
                    # print(f"constraint violation for max and argmax: {constraint_violation}")
                    constraint_violation_mag += constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1
            elif constraint_key == "argmin":
                argmin = constraints["argmin"][sample_idx][channel_idx]
                constraint_violation = np.max(
                    timeseries_channel[argmin] - timeseries_channel
                )
                if constraint_violation > ACCEPTED_TOERANCE:
                    # print(f"constraint violation for argmin: {constraint_violation}")
                    constraint_violation_mag += constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1
            elif constraint_key == "min and argmin":
                min_val = np.min(timeseries_channel)
                req_min_val = constraints["min and argmin"]["min"][sample_idx][
                    channel_idx
                ]
                constraint_violation = np.abs(min_val - req_min_val)
                if constraint_violation > ACCEPTED_TOERANCE:
                    # print(f"constraint violation for min and argmin: {constraint_violation}")
                    constraint_violation_mag += constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1
            elif constraint_key == "mean":
                mean_val = np.mean(timeseries_channel)
                req_mean_val = constraints["mean"][sample_idx][channel_idx]
                constraint_violation = np.abs(mean_val - req_mean_val)
                if constraint_violation > ACCEPTED_TOERANCE:
                    # print(f"constraint violation for mean: {constraint_violation}")
                    constraint_violation_mag += constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1
            elif constraint_key == "mean change":
                mean_change_val = np.mean(np.diff(timeseries_channel))
                req_mean_change_val = constraints["mean change"][sample_idx][
                    channel_idx
                ]
                constraint_violation = np.abs(mean_change_val - req_mean_change_val)
                if constraint_violation > ACCEPTED_TOERANCE:
                    # print(f"constraint violation for mean change: {constraint_violation}")
                    constraint_violation_mag += constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1
            elif "val_at" in constraint_key:
                loc = int(constraint_key.split("_")[-1]) - 1
                act_val = timeseries_channel[loc]
                req_val = constraints[constraint_key][sample_idx][channel_idx]
                constraint_violation = np.abs(act_val - req_val)
                if constraint_violation > ACCEPTED_TOERANCE:
                    # print(
                    # f"constraint violation for {constraint_key}: {constraint_violation}"
                    # )
                    constraint_violation_mag += constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1
            elif "autocorr" in constraint_key:
                lag = int(constraint_key.split("_")[-1])
                autocorr_constraint_value = constraints[constraint_key][
                    sample_idx
                ][channel_idx]
                current_mean = np.mean(timeseries_channel)
                current_variance = np.var(timeseries_channel)
                autocorr_constraint = np.mean(
                    (timeseries_channel[:-lag] - current_mean)
                    * (timeseries_channel[lag:] - current_mean)
                ) / current_variance
                autocorr_constraint_violation = np.abs(
                    autocorr_constraint - autocorr_constraint_value
                )
                # print(autocorr_constraint_value, autocorr_constraint)
                if autocorr_constraint_violation > ACCEPTED_TOERANCE:
                    # print(f"constraint violation for autocorr: {autocorr_constraint_violation}")
                    constraint_violation_mag += autocorr_constraint_violation
                    constraint_violation_matrix[channel_idx, constraint_id] = 1      
            
            elif constraint_key == "peak and valley":
                peak_and_valley_locs = constraints[constraint_key][
                    (sample_idx, channel_idx)
                ][0]
                peak_valley_trends = constraints[constraint_key][
                    (sample_idx, channel_idx)
                ][1]
                peak_valley_values = constraints[constraint_key][
                    (sample_idx, channel_idx)
                ][2]

                num_peak_valley_pairs = len(peak_and_valley_locs)

                for i in range(num_peak_valley_pairs):
                    start = peak_and_valley_locs[i][0]
                    end = peak_and_valley_locs[i][1]
                    trend = peak_valley_trends[i]
                    start_val = peak_valley_values[i][0]
                    end_val = peak_valley_values[i][1]

                    constraint_violation = np.abs(start_val - timeseries_channel[start])
                    if constraint_violation > ACCEPTED_TOERANCE:
                        # print(f"constraint violation for peak and valley: {constraint_violation}")
                        constraint_violation_mag += constraint_violation
                        constraint_violation_matrix[channel_idx, constraint_id] = 1
                    constraint_violation = np.abs(end_val - timeseries_channel[end])
                    if constraint_violation > ACCEPTED_TOERANCE:
                        # print(f"constraint violation for peak and valley: {constraint_violation}")
                        constraint_violation_mag += constraint_violation
                        constraint_violation_matrix[channel_idx, constraint_id] = 1

                    if trend == 1:
                        for idx in range(start + 1, end - 1):
                            constraint_violation = np.max(
                                timeseries_channel[start:idx] - timeseries_channel[idx]
                            )
                            if constraint_violation > ACCEPTED_TOERANCE:
                                # print(f"constraint violation for peak and valley: {constraint_violation}")
                                constraint_violation_mag += constraint_violation
                                constraint_violation_matrix[
                                    channel_idx, constraint_id
                                ] = 1

                    else:
                        for idx in range(start + 1, end - 1):
                            constraint_violation = np.max(
                                timeseries_channel[start:idx] - timeseries_channel[idx]
                            )
                            if constraint_violation < -ACCEPTED_TOERANCE:
                                # print(f"constraint violation for peak and valley: {constraint_violation}")
                                constraint_violation_mag += constraint_violation
                                constraint_violation_matrix[
                                    channel_idx, constraint_id
                                ] = 1

    if "ohlc" in constraints_keys:
        open_mean = constraints["ohlc"]["open_mean"]
        high_mean = constraints["ohlc"]["high_mean"]
        low_mean = constraints["ohlc"]["low_mean"]
        close_mean = constraints["ohlc"]["close_mean"]
        open_std = constraints["ohlc"]["open_std"]
        high_std = constraints["ohlc"]["high_std"]
        low_std = constraints["ohlc"]["low_std"]
        close_std = constraints["ohlc"]["close_std"]
        open_unscaled = sample[0] * open_std + open_mean
        high_unscaled = sample[1] * high_std + high_mean
        low_unscaled = sample[2] * low_std + low_mean
        close_unscaled = sample[3] * close_std + close_mean
        
        oh_violation = np.max(open_unscaled - high_unscaled)
        lo_violation = np.max(low_unscaled - open_unscaled)
        lc_violation = np.max(low_unscaled - close_unscaled)
        ch_violation = np.max(close_unscaled - high_unscaled)
        
        if oh_violation > ACCEPTED_TOERANCE:
            # print(f"constraint violation for oh: {oh_violation}")
            constraint_violation_mag += oh_violation
            constraint_violation_matrix[0, constraint_id] = 1
        if lo_violation > ACCEPTED_TOERANCE:
            # print(f"constraint violation for lo: {lo_violation}")
            constraint_violation_mag += lo_violation
            constraint_violation_matrix[1, constraint_id] = 1
        if lc_violation > ACCEPTED_TOERANCE:
            # print(f"constraint violation for lc: {lc_violation}")
            constraint_violation_mag += lc_violation
            constraint_violation_matrix[2, constraint_id] = 1
        if ch_violation > ACCEPTED_TOERANCE:
            # print(f"constraint violation for ch: {ch_violation}")
            constraint_violation_mag += ch_violation
            constraint_violation_matrix[3, constraint_id] = 1

    return np.sum(constraint_violation_matrix) == 0, constraint_violation_mag
