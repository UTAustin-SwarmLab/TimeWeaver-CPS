import numpy as np
import torch
import cvxpy as cp
from scipy.optimize import LinearConstraint

def obtain_mean_constraint_violation(mean_constraint, samples_batch_estimate):
    current_mean = torch.mean(samples_batch_estimate, dim=-1)  # B X C
    mean_diff_vector = current_mean - mean_constraint
    mean_constraint_violation = torch.square(
        torch.linalg.vector_norm(mean_diff_vector, dim=-1)
    )
    assert mean_constraint_violation.shape == (samples_batch_estimate.shape[0],)
    return mean_constraint_violation    

def obtain_scipy_mean_constraint(meanval, horizon, num_elems, start_loc, end_loc, tolerance, keep_feasible):
    A = np.zeros((1, num_elems))
    A[0, start_loc:end_loc] = 1 / horizon            
    ub = meanval + tolerance
    lb = meanval - tolerance
    mean_constraint = LinearConstraint(
        A, lb=lb, ub=ub, keep_feasible=keep_feasible
    )
    return mean_constraint
    
def obtain_mean_penalty_cvxpy(opt_var, start_idx, end_idx, mean_constraint, tolerance):
    return cp.maximum(cp.abs(cp.mean(opt_var[start_idx:end_idx]) - mean_constraint) - tolerance, 0)