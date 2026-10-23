import numpy as np
import torch
import cvxpy as cp
from scipy.optimize import LinearConstraint

def obtain_value_at_constraint_violation(value_at_constraint, samples_batch_estimate, loc):
    current_value_at = samples_batch_estimate[:, :, loc]  # B X C
    value_at_diff_vector = current_value_at - value_at_constraint
    value_at_constraint_violation = torch.square(
        torch.linalg.vector_norm(value_at_diff_vector, dim=-1)
    )
    assert value_at_constraint_violation.shape == (samples_batch_estimate.shape[0],)
    return value_at_constraint_violation

def obtain_scipy_value_at_constraint(value_at_timestep, num_elems, start_loc, loc, tolerance, keep_feasible):
    A = np.zeros((1, num_elems))
    A[0, start_loc + loc] = 1             
    ub = value_at_timestep + tolerance
    lb = value_at_timestep - tolerance
    val_at_constraint = LinearConstraint(A, lb=lb, ub=ub, keep_feasible=keep_feasible)
    return val_at_constraint

def obtain_value_at_penalty_cvxpy(opt_var, start_idx, loc, value_at_constraint, tolerance):
    return cp.maximum(cp.abs(opt_var[start_idx + loc] - value_at_constraint) - tolerance, 0)