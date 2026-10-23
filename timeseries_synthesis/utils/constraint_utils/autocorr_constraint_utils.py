import numpy as np 
import torch
import cvxpy as cp
from scipy.optimize import NonlinearConstraint
from functools import partial

def obtain_autocorr_constraint_violation(autocorr_constraint, samples_batch_estimate, lag):
    B = samples_batch_estimate.shape[0]
    C = samples_batch_estimate.shape[1]
    
    autocorr_values = torch.zeros((B, C)).to(samples_batch_estimate.device)
    for sample_idx in range(B):
        for channel_idx in range(C):
            timeseries = samples_batch_estimate[sample_idx, channel_idx]
            channel_mean = torch.mean(timeseries)
            channel_variance = torch.var(timeseries)
            autocorr_values[sample_idx, channel_idx] = torch.mean(
                (timeseries[:-lag] - channel_mean)
                * (timeseries[lag:] - channel_mean)
            ) / channel_variance
    autocorr_values = autocorr_values.to(samples_batch_estimate.device)
    autocorr_diff_vector = autocorr_values - autocorr_constraint
    autocorr_constraint_violation = torch.square(
        torch.linalg.vector_norm(autocorr_diff_vector, dim=-1)
    )
    assert autocorr_constraint_violation.shape == (samples_batch_estimate.shape[0],)
    return autocorr_constraint_violation

def compute_autocorrelation(x, lag, start_loc, end_loc):
    """
    Compute the autocorrelation of a signal x at a given lag
    """
    mean = np.mean(x[start_loc:end_loc])
    variance = np.var(x[start_loc:end_loc])
    autocorr = np.mean((x[start_loc:end_loc - lag] - mean) * (x[start_loc + lag:end_loc] - mean)) / variance
    return autocorr

def obtain_scipy_autocorr_constraint(autocorr_constraint, start_loc, end_loc, lag, tolerance, keep_feasible):
    autocorr_function = partial(compute_autocorrelation, lag=lag, start_loc=start_loc, end_loc=end_loc)
    constraint = NonlinearConstraint(
        autocorr_function,
        autocorr_constraint - tolerance,
        autocorr_constraint + tolerance,
        keep_feasible=keep_feasible
    )
    return constraint

def obtain_autocorr_penalty_scipy(x, autocorr_constraint, start_loc, end_loc, lag, tolerance):
    return np.maximum(np.abs(compute_autocorrelation(x, lag, start_loc, end_loc) - autocorr_constraint) - tolerance, 0)
    
# def obtain_argmax_penalty_cvxpy(opt_var, start_idx, end_idx, argmax_constraint):
#     return cp.maximum(cp.max(opt_var[start_idx:end_idx] - opt_var[start_idx + argmax_constraint]), 0)