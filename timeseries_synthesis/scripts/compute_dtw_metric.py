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
from scipy.ndimage import uniform_filter1d

def ssim_1d(signal1, signal2, window_size=7, C1=1e-4, C2=9e-4):
    # Mean of signal1 and signal2
    mu1 = uniform_filter1d(signal1, size=window_size)
    mu2 = uniform_filter1d(signal2, size=window_size)

    # Variances of signal1 and signal2
    sigma1_sq = uniform_filter1d(signal1**2, size=window_size) - mu1**2
    sigma2_sq = uniform_filter1d(signal2**2, size=window_size) - mu2**2

    # Covariance of signal1 and signal2
    sigma12 = uniform_filter1d(signal1 * signal2, size=window_size) - mu1 * mu2

    # Compute SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim = numerator / denominator
    return np.mean(ssim)

@hydra.main(config_path="../../configs/constrained_generation_configs", version_base="1.1")
def main(config: DictConfig):
    dataset_config = get_dataset_config(config)
    num_constraints = len(dataset_config.equality_constraints_to_extract)
    gt_data_dir = dataset_config.log_dir
    test_timeseries = np.load(os.path.join(gt_data_dir, "test_timeseries.npy"))
    test_timeseries = test_timeseries.astype(np.float32)

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
    synthetic_test_timeseries = np.concatenate(synthetic_test_timeseries_list, axis=0)

    print(test_timeseries.shape, synthetic_test_timeseries.shape)

    horizon = test_timeseries.shape[-1]
    test_timeseries_flattened = test_timeseries.reshape(-1, horizon)
    synthetic_test_timeseries_flattened = synthetic_test_timeseries.reshape(-1, horizon)

    dtw_list = []
    ssim_list = []
    for idx in range(test_timeseries_flattened.shape[0]):
        dtw_score = dtw_metric(
            test_timeseries_flattened[idx], synthetic_test_timeseries_flattened[idx]
        )
        assert dtw_score >= 0
        dtw_list.append(dtw_score)
        
        max_val = max(np.max(test_timeseries_flattened[idx]), np.max(synthetic_test_timeseries_flattened[idx]))
        min_val = min(np.min(test_timeseries_flattened[idx]), np.min(synthetic_test_timeseries_flattened[idx]))
        x = (test_timeseries_flattened[idx] - min_val) / (max_val - min_val)
        y = (synthetic_test_timeseries_flattened[idx] - min_val) / (max_val - min_val)
        ssim_score = ssim_1d(x, y)
        
        # ssim_score = ssim_1d(
        #     test_timeseries_flattened[idx], synthetic_test_timeseries_flattened[idx]
        # ) 
        ssim_list.append(ssim_score)

    print(f"{OKYELLOW}Mean DTW Metric: {np.mean(dtw_list)}{ENDC}")
    print(f"{OKYELLOW}Std DTW Metric: {np.std(dtw_list)}{ENDC}")

    print(f"{OKYELLOW}Mean SSIM Metric: {np.mean(ssim_list)}{ENDC}")
    print(f"{OKYELLOW}Std SSIM Metric: {np.std(ssim_list)}{ENDC}")


    return 0


if __name__ == "__main__":
    main()
