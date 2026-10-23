import numpy as np 
import torch
import cvxpy as cp
from scipy.optimize import LinearConstraint

def obtain_peak_and_valley_constraint_violation(peak_and_valley_constraint_dict, samples_batch_estimate):
    trend_constraint_matrix = torch.tensor(peak_and_valley_constraint_dict["trend_constraint_matrix"]).to(samples_batch_estimate.device, dtype=samples_batch_estimate.dtype)
    values_matrix = torch.tensor(peak_and_valley_constraint_dict["values_matrix"]).float().to(samples_batch_estimate.device, dtype=samples_batch_estimate.dtype) 
    indicator_matrix = torch.tensor(peak_and_valley_constraint_dict["indicator_matrix"]).float().to(samples_batch_estimate.device, dtype=samples_batch_estimate.dtype)

    samples_batch_estimate_extended = samples_batch_estimate.unsqueeze(-1)
            
    assert (samples_batch_estimate_extended.shape[0] == trend_constraint_matrix.shape[0])
    assert (samples_batch_estimate_extended.shape[1] == trend_constraint_matrix.shape[1])
    assert (samples_batch_estimate_extended.shape[2] == trend_constraint_matrix.shape[3])
            
    trend_constraint_violation = torch.nn.ReLU()(-1 * torch.einsum("bcij,bcjk->bcik", trend_constraint_matrix, samples_batch_estimate_extended,))
    trend_constraint_violation = trend_constraint_violation.reshape(samples_batch_estimate.shape[0], -1)
    
    assert trend_constraint_violation.shape == (samples_batch_estimate.shape[0], samples_batch_estimate.shape[1] * trend_constraint_matrix.shape[2],)
            
    trend_constraint_violation = torch.sum(trend_constraint_violation, dim=-1)
            
    assert trend_constraint_violation.shape == (samples_batch_estimate.shape[0],)
            
    values_violation = samples_batch_estimate * indicator_matrix - values_matrix
    values_violation = values_violation.reshape(samples_batch_estimate.shape[0], -1)
            
    assert values_violation.shape == (samples_batch_estimate.shape[0], samples_batch_estimate.shape[1] * samples_batch_estimate.shape[2],)
    values_violation = torch.square(torch.linalg.vector_norm(values_violation, dim=-1))
            
    assert values_violation.shape == (samples_batch_estimate.shape[0],)

    peak_and_valley_constraint_violation_batch = trend_constraint_violation + values_violation
    
    return peak_and_valley_constraint_violation_batch

def obtain_scipy_peak_and_valley_constraint(channel_trend_constraint_matrix, channel_values_matrix, channel_indicator_matrix, horizon, num_elems, start_loc, end_loc, tolerance, keep_feasible,):
    locs = np.where(channel_indicator_matrix == 1)[0]
    num_locs = len(locs)
    A = np.zeros((num_locs, num_elems))
    lb = np.zeros(num_locs)
    ub = np.zeros(num_locs)
    for loc_idx in range(len(locs)):
        loc = locs[loc_idx]
        A[loc_idx, start_loc + loc] = 1
        lb[loc_idx] = channel_values_matrix[loc] - tolerance
        ub[loc_idx] = channel_values_matrix[loc] + tolerance
                
    peak_valley_constraint = LinearConstraint(A, lb=lb, ub=ub, keep_feasible=keep_feasible)

    B = np.zeros((horizon-1, num_elems))
    B[:, start_loc:end_loc] = channel_trend_constraint_matrix
    trend_constraint = LinearConstraint(B, lb=0, keep_feasible=keep_feasible)

    return peak_valley_constraint, trend_constraint

def obtain_peak_and_valley_penalty_cvxpy(opt_var, sample_idx, channel_idx, constraints_to_be_executed, horizon, tolerance):
    start = channel_idx * horizon
    peak_and_valley_locs = constraints_to_be_executed["peak and valley"][(sample_idx, channel_idx)][0]
    peak_valley_trends = constraints_to_be_executed["peak and valley"][(sample_idx, channel_idx)][1]
    peak_valley_values = constraints_to_be_executed["peak and valley"][(sample_idx, channel_idx)][2]
    
    peak_and_valley_violation = 0
    for i in range(len(peak_and_valley_locs)):
        pv_start = peak_and_valley_locs[i][0]
        pv_end = peak_and_valley_locs[i][1]
        trend = peak_valley_trends[i]
        start_val = peak_valley_values[i][0]
        end_val = peak_valley_values[i][1]
                
        start_val_violation = cp.maximum(cp.abs(opt_var[start + pv_start] - start_val) - tolerance, 0) 
        end_val_violation = cp.maximum(cp.abs(opt_var[start + pv_end] - end_val) - tolerance, 0)
        
        peak_and_valley_violation += start_val_violation + end_val_violation
        
        if trend == 1:
            trend_penalty = cp.max(-cp.diff(opt_var[start + pv_start : start + pv_end + 1]))
            trend_penalty = cp.maximum(trend_penalty, 0)
        else:
            trend_penalty = cp.max(cp.diff(opt_var[start + pv_start : start + pv_end + 1]))
            trend_penalty = cp.maximum(trend_penalty, 0)
        
        peak_and_valley_violation += trend_penalty
        
    return peak_and_valley_violation