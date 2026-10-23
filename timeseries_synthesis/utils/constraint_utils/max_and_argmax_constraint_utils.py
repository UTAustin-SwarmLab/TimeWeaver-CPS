import numpy as np 
import torch

def obtain_max_and_argmax_constraint_violation(argmax_constraint, max_constraint, samples_batch_estimate):
    indices_to_set = argmax_constraint.unsqueeze(-1)
    current_max_values = samples_batch_estimate.gather(2, indices_to_set)
    assert current_max_values.shape == (
        samples_batch_estimate.shape[0],
        samples_batch_estimate.shape[1],
        1,
    )
    current_max_val_diff_vector = samples_batch_estimate - current_max_values
    current_max_val_violation = torch.nn.ReLU()(current_max_val_diff_vector)
    assert current_max_val_violation.shape == samples_batch_estimate.shape
    per_sample_argmax_loss = torch.sum(current_max_val_violation, dim=(-1, -2))
    assert per_sample_argmax_loss.shape == (samples_batch_estimate.shape[0],)
    current_max_values = current_max_values.squeeze(-1)
    current_max_val_diff_vector = current_max_values - max_constraint
    current_max_val_violation = torch.square(
        torch.linalg.vector_norm(current_max_val_diff_vector, dim=-1)
    )
    assert current_max_val_violation.shape == (samples_batch_estimate.shape[0],)
    return per_sample_argmax_loss + current_max_val_violation