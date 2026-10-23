import numpy as np
import torch
import cvxpy as cp
from scipy.optimize import LinearConstraint

def obtain_argmin_constraint_violation(argmin_constraint, samples_batch_estimate):
    indices_to_set = argmin_constraint.unsqueeze(-1)
    current_min_values = samples_batch_estimate.gather(2, indices_to_set)
    assert current_min_values.shape == (
        samples_batch_estimate.shape[0],
        samples_batch_estimate.shape[1],
        1,
    )
    current_min_val_diff_vector = current_min_values - samples_batch_estimate
    current_min_val_violation = torch.nn.ReLU()(current_min_val_diff_vector)
    assert current_min_val_violation.shape == samples_batch_estimate.shape
    per_sample_argmin_loss = torch.sum(current_min_val_violation, dim=(-1, -2))
    assert per_sample_argmin_loss.shape == (samples_batch_estimate.shape[0],)
    return per_sample_argmin_loss

def obtain_scipy_argmin_constraint(argmin_loc, horizon, num_elems, start_loc, end_loc, keep_feasible):
    A = np.zeros((horizon, num_elems))
    A[:, start_loc:end_loc] = np.eye(horizon) # to focus on the channel of interest
    A[:, start_loc + argmin_loc] = -1
    A[argmin_loc, :] = 0
    lb = -1e-6
    argmin_constraint = LinearConstraint(A, lb=lb, keep_feasible=keep_feasible)
    return argmin_constraint

def obtain_argmin_penalty_cvxpy(opt_var, start_idx, end_idx, argmin_constraint):
    return cp.maximum(cp.max(opt_var[start_idx + argmin_constraint] - opt_var[start_idx:end_idx]), 0)