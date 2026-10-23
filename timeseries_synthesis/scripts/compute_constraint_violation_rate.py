from timeseries_synthesis.metrics.distance_utils import dtw_metric
import os
import numpy as np
import hydra


from omegaconf import DictConfig

from timeseries_synthesis.utils.basic_utils import (
    generate_synthesis_experiment_details,
    get_dataset_config,
    OKYELLOW,
    ENDC,
)

from timeseries_synthesis.utils.constrained_synthesis_helper_functions import (
    verify_constraint_satisfaction_for_sample
)

@hydra.main(config_path="../../configs/constrained_generation_configs", version_base="1.1")
def main(config: DictConfig):
    dataset_config = get_dataset_config(config)
    num_constraints = len(dataset_config.equality_constraints_to_extract)
    gt_data_dir = dataset_config.log_dir
    test_timeseries = np.load(os.path.join(gt_data_dir, "test_timeseries.npy"))

    synthetic_data_dir = generate_synthesis_experiment_details(config.synthetic_dataset_dir, config, num_constraints, gt_data_dir, eval_mode=True)['save_dir']
    print(
        OKYELLOW
        + "All the results will be stored in this directory: "
        + str(synthetic_data_dir)
        + ENDC
    )
    synthetic_test_timeseries_dir = os.path.join(synthetic_data_dir, "test")
    num_synthetic_test_timeseries_files = len(
        [
            f
            for f in os.listdir(synthetic_test_timeseries_dir)
            if f.startswith("timeseries")
        ]
    )
    synthetic_test_timeseries_list = [
        np.load(os.path.join(synthetic_test_timeseries_dir, f"timeseries_{i}.npy"))
        for i in range(num_synthetic_test_timeseries_files)
    ]
    constraints_list = [
        np.load(os.path.join(synthetic_test_timeseries_dir, f"constraints_{i}.npy"), allow_pickle=True).item()
        for i in range(num_synthetic_test_timeseries_files)
    ]
    
    num_elements = len(synthetic_test_timeseries_list)
    
    constraint_satisfaction_rate = 0
    avg_constraint_violation_magnitude = []
    for idx in range(num_elements):
        synthetic_timeseries_batch = synthetic_test_timeseries_list[idx]
        constraints_batch = constraints_list[idx]
        for i in range(len(synthetic_timeseries_batch)):
            synthetic_timeseries = synthetic_timeseries_batch[i]
            constraint_satisfied, constraint_violation_magnitude = verify_constraint_satisfaction_for_sample(synthetic_timeseries, constraints_batch, i)
            if constraint_satisfied:
                constraint_satisfaction_rate += 1
            avg_constraint_violation_magnitude.append(constraint_violation_magnitude)

    num_samples = test_timeseries.shape[0]
    constraint_satisfaction_rate /= num_samples
    print(f"Constraint violation rate: {1-constraint_satisfaction_rate}")
    
    avg_constraint_violation_magnitude = np.mean(avg_constraint_violation_magnitude)
    print(f"Average constraint violation magnitude: {avg_constraint_violation_magnitude}")

    return 0


if __name__ == "__main__":
    main()
