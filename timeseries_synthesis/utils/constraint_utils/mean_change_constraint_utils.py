import numpy as np 
import torch
import cvxpy as cp
from scipy.optimize import LinearConstraint

def obtain_mean_change_constraint_violation(mean_change_constraint, samples_batch_estimate):
    current_mean_change = torch.mean(torch.diff(samples_batch_estimate), dim=-1)
    mean_change_diff_vector = current_mean_change - mean_change_constraint
    mean_change_constraint_violation = torch.square(
        torch.linalg.vector_norm(mean_change_diff_vector, dim=-1)
    )
    assert mean_change_constraint_violation.shape == (samples_batch_estimate.shape[0],)
    return mean_change_constraint_violation

def obtain_scipy_mean_change_constraint(mean_change_val, horizon, num_elems, start_loc, end_loc, tolerance, keep_feasible):
    A = np.zeros((1, num_elems))
    A[0, start_loc] = -1 / (horizon - 1)
    A[0, end_loc - 1] = 1 / (horizon - 1)
    # This might not seem straightforward but it is correct
    # This is similar to
    # A[0, start_loc : end_loc - 1] = -1 / (horizon - 1)
    # A[0, start_loc + 1 : end_loc] = 1 / (horizon - 1)

    ub = mean_change_val + tolerance
    lb = mean_change_val - tolerance
    mean_change_constraint = LinearConstraint(A, lb=lb, ub=ub, keep_feasible=keep_feasible)
    return mean_change_constraint

def obtain_mean_change_penalty_cvxpy(opt_var, start_idx, end_idx, mean_change_constraint, tolerance):
    return cp.maximum(cp.abs(cp.mean(cp.diff(opt_var[start_idx:end_idx])) - mean_change_constraint) - tolerance, 0)