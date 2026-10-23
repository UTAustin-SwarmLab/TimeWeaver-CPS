import numpy as np 
import torch
import cvxpy as cp
from scipy.optimize import LinearConstraint

def obtain_argmax_constraint_violation(argmax_constraint, samples_batch_estimate):
    indices_to_set = argmax_constraint.unsqueeze(-1)
    current_max_values = samples_batch_estimate.gather(2, indices_to_set)
    assert current_max_values.shape == (
        samples_batch_estimate.shape[0],
        samples_batch_estimate.shape[1],
        1,
    )
    # # pick a random sample 
    # num_samples = samples_batch_estimate.shape[0]
    # random_sample_idx = np.random.randint(num_samples)
    # # pick a random channel 
    # num_channels = samples_batch_estimate.shape[1]
    # random_channel_idx = np.random.randint(num_channels)
    # req_argmax_index = argmax_constraint[random_sample_idx, random_channel_idx]
    # current_argmax_val = samples_batch_estimate[random_sample_idx, random_channel_idx, req_argmax_index]
    # assert current_argmax_val == current_max_values[random_sample_idx, random_channel_idx, 0]
    # print(f"current_argmax_val: {current_argmax_val}")
    # print(f"current_max_values: {current_max_values[random_sample_idx, random_channel_idx, 0]}")
    
    current_max_val_diff_vector = samples_batch_estimate - current_max_values
    current_max_val_violation = torch.nn.ReLU()(current_max_val_diff_vector)
    assert current_max_val_violation.shape == samples_batch_estimate.shape
    per_sample_argmax_loss = torch.sum(current_max_val_violation, dim=(-1, -2))
    assert per_sample_argmax_loss.shape == (samples_batch_estimate.shape[0],)
    return per_sample_argmax_loss

def obtain_scipy_argmax_constraint(argmax_loc, horizon, num_elems, start_loc, end_loc, keep_feasible):
    A = np.zeros((horizon, num_elems))
    A[:, start_loc:end_loc] = np.eye(horizon)
    A[:, start_loc + argmax_loc] = -1
    A[argmax_loc, :] = 0
    ub = 1e-6
    argmax_constraint = LinearConstraint(A, ub=ub, keep_feasible=keep_feasible)
    return argmax_constraint

def obtain_argmax_penalty_cvxpy(opt_var, start_idx, end_idx, argmax_constraint):
    return cp.maximum(cp.max(opt_var[start_idx:end_idx] - opt_var[start_idx + argmax_constraint]), 0)