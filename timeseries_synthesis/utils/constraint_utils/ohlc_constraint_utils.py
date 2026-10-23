import numpy as np
import torch
import cvxpy as cp
from scipy.optimize import LinearConstraint

def obtain_ohlc_constraint_violation(ohlc_dict, samples_batch_estimate):
    open_mean = ohlc_dict["open_mean"]
    high_mean = ohlc_dict["high_mean"]
    low_mean = ohlc_dict["low_mean"]
    close_mean = ohlc_dict["close_mean"]
    open_std = ohlc_dict["open_std"]
    high_std = ohlc_dict["high_std"]
    low_std = ohlc_dict["low_std"]
    close_std = ohlc_dict["close_std"]
    
    open_val = samples_batch_estimate[:, 0] * open_std + open_mean
    high_val = samples_batch_estimate[:, 1] * high_std + high_mean
    low_val = samples_batch_estimate[:, 2] * low_std + low_mean
    close_val = samples_batch_estimate[:, 3] * close_std + close_mean
    
    open_high_penalty_vector = torch.nn.ReLU()(open_val - high_val)
    close_high_penalty_vector = torch.nn.ReLU()(close_val - high_val)
    low_open_penalty_vector = torch.nn.ReLU()(low_val - open_val)
    low_close_penalty_vector = torch.nn.ReLU()(low_val - close_val)
    low_high_penalty_vector = torch.nn.ReLU()(low_val - high_val)
    
    ohlc_penalty_vector = torch.concatenate(
        [
            open_high_penalty_vector,
            close_high_penalty_vector,
            low_open_penalty_vector,
            low_close_penalty_vector,
            low_high_penalty_vector,
        ],
        dim=-1,
    )
    
    assert ohlc_penalty_vector.shape == (
        samples_batch_estimate.shape[0],
        5 * samples_batch_estimate.shape[2],
    )
    per_sample_ohlc_penalty_vector = torch.sum(ohlc_penalty_vector, dim=-1)
    assert per_sample_ohlc_penalty_vector.shape == (
        samples_batch_estimate.shape[0],
    )
    
    return per_sample_ohlc_penalty_vector

def obtain_scipy_ohlc_constraint(ohlc_constraint_dict, horizon, num_elems, keep_feasible):
    open_mean = ohlc_constraint_dict["open_mean"]
    high_mean = ohlc_constraint_dict["high_mean"]
    low_mean = ohlc_constraint_dict["low_mean"]
    close_mean = ohlc_constraint_dict["close_mean"]
    open_std = ohlc_constraint_dict["open_std"]
    high_std = ohlc_constraint_dict["high_std"]
    low_std = ohlc_constraint_dict["low_std"]
    close_std = ohlc_constraint_dict["close_std"]

    open_indicator_matrix = np.zeros((horizon, num_elems))
    open_indicator_matrix[:, 0:horizon] = np.eye(horizon)
    open_indicator_matrix = open_indicator_matrix * open_std
        
    high_indicator_matrix = np.zeros((horizon, num_elems))
    high_indicator_matrix[:, horizon : 2 * horizon] = np.eye(horizon)
    high_indicator_matrix = high_indicator_matrix * high_std
        
    low_indicator_matrix = np.zeros((horizon, num_elems))
    low_indicator_matrix[:, 2 * horizon : 3 * horizon] = np.eye(horizon)
    low_indicator_matrix = low_indicator_matrix * low_std
        
    close_indicator_matrix = np.zeros((horizon, num_elems))
    close_indicator_matrix[:, 3 * horizon : 4 * horizon] = np.eye(horizon)
    close_indicator_matrix = close_indicator_matrix * close_std
        
    open_high_constraint_matrix = open_indicator_matrix - high_indicator_matrix
    open_high_upper_bound = high_mean - open_mean
    open_high_constraint = LinearConstraint(open_high_constraint_matrix, ub=open_high_upper_bound, keep_feasible=keep_feasible,)
        
    close_high_constraint_matrix = close_indicator_matrix - high_indicator_matrix
    close_high_upper_bound = high_mean - close_mean
    close_high_constraint = LinearConstraint(close_high_constraint_matrix, ub=close_high_upper_bound, keep_feasible=keep_feasible,)
        
    open_low_constraint_matrix = low_indicator_matrix - open_indicator_matrix
    open_low_upper_bound = open_mean - low_mean
    open_low_constraint = LinearConstraint(open_low_constraint_matrix, ub=open_low_upper_bound, keep_feasible=keep_feasible,)
        
    close_low_constraint_matrix = low_indicator_matrix - close_indicator_matrix
    close_low_upper_bound = close_mean - low_mean
    close_low_constraint = LinearConstraint(close_low_constraint_matrix, ub=close_low_upper_bound, keep_feasible=keep_feasible,)
    
    return open_high_constraint, close_high_constraint, open_low_constraint, close_low_constraint

def obtain_ohlc_penalty_cvxpy(opt_var, constraints_to_be_executed, horizon, tolerance):
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

    ohlc_violation = cp.maximum(cp.max(open_unscaled - high_unscaled), 0) 
    ohlc_violation += cp.maximum(cp.max(low_unscaled - open_unscaled), 0)
    ohlc_violation += cp.maximum(cp.max(low_unscaled - close_unscaled), 0)
    ohlc_violation += cp.maximum(cp.max(close_unscaled - high_unscaled), 0)
    
    return ohlc_violation

def obtain_ohlc_penalty_scipy(opt_var, constraints_to_be_executed, horizon, tolerance):
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

    ohlc_violation = np.maximum(np.max(open_unscaled - high_unscaled), 0) 
    ohlc_violation += np.maximum(np.max(low_unscaled - open_unscaled), 0)
    ohlc_violation += np.maximum(np.max(low_unscaled - close_unscaled), 0)
    ohlc_violation += np.maximum(np.max(close_unscaled - high_unscaled), 0)
    
    return ohlc_violation