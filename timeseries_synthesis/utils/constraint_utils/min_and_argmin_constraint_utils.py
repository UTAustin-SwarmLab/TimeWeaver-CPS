import numpy as np 
import torch

def obtain_min_and_argmin_constraint_violation(argmin_constraint, min_constraint, samples_batch_estimate):
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
    current_min_values = current_min_values.squeeze(-1)
    assert current_min_values.shape == (
        samples_batch_estimate.shape[0],
        samples_batch_estimate.shape[1],
    )
    current_min_val_diff_vector = current_min_values - min_constraint
    current_min_val_violation = torch.square(
        torch.linalg.vector_norm(current_min_val_diff_vector, dim=-1)
    )
    assert current_min_val_violation.shape == (samples_batch_estimate.shape[0],)
    return per_sample_argmin_loss + current_min_val_violation