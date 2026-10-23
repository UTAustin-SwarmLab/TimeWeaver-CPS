import os 
import argparse
import numpy as np 
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pmdarima import auto_arima
from time import time

def extract_sarima_parameters(ts):
    data = pd.Series(ts)
    start_time = time()
    model = auto_arima(
        data,
        seasonal=True,
        m=24,                # 12-month seasonality
        stepwise=True,       # fast search
        trace=False,         # disable trace for parallel execution
        suppress_warnings=True
    )
    end_time = time()
    extraction_time = end_time - start_time
    params_dict = model.params().to_dict()
    return params_dict, extraction_time


def process_timeseries(args):
    """Process a single timeseries and return its index and parameters"""
    idx, ts = args
    params_dict, extraction_time = extract_sarima_parameters(ts)
    return idx, params_dict, extraction_time


def extract_parameters_parallel(timeseries_array, num_processes=None, desc="Processing"):
    """Extract SARIMA parameters in parallel using multi-processing"""
    num_series = timeseries_array.shape[0]
    parameters = [None] * num_series
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks - pass actual timeseries data for efficiency
        future_to_idx = {
            executor.submit(process_timeseries, (idx, timeseries_array[idx, 0])): idx 
            for idx in range(num_series)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=num_series, desc=desc) as pbar:
            for future in as_completed(future_to_idx):
                idx, params_dict, extraction_time = future.result()
                print(f"Extraction time for timeseries {idx}: {extraction_time} seconds")
                parameters[idx] = params_dict
                pbar.update(1)
    
    return parameters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SARIMA parameters for the traffic dataset")
    parser.add_argument("--traffic_dataset_dir", type=str, required=True)
    traffic_dataset_dir = parser.parse_args().traffic_dataset_dir

    train_timeseries = np.load(os.path.join(traffic_dataset_dir, "train_timeseries.npy"))
    val_timeseries = np.load(os.path.join(traffic_dataset_dir, "val_timeseries.npy"))
    test_timeseries = np.load(os.path.join(traffic_dataset_dir, "test_timeseries.npy"))

    # Extract parameters in parallel using multi-processing
    test_parameters = extract_parameters_parallel(
        test_timeseries, 
        num_processes=1,  # Number of worker processes
        desc="Processing test timeseries"
    )

    val_parameters = extract_parameters_parallel(
        val_timeseries,
        num_processes=1,
        desc="Processing val timeseries"
    )

    train_parameters = extract_parameters_parallel(
        train_timeseries,
        num_processes=1,
        desc="Processing train timeseries"
    )

    possible_unique_parameter_keys = list() 
    for params in test_parameters:
        possible_unique_parameter_keys.extend(list(params.keys()))
    for params in val_parameters:
        possible_unique_parameter_keys.extend(list(params.keys()))
    for params in train_parameters:
        possible_unique_parameter_keys.extend(list(params.keys()))
    possible_unique_parameter_keys = list(set(possible_unique_parameter_keys))

    standard_test_parameters = []
    standard_val_parameters = []
    standard_train_parameters = []
    for params in test_parameters:
        standard_params = {key: 0.0 for key in possible_unique_parameter_keys}
        for key, value in params.items():
            standard_params[key] = value
        standard_test_parameters.append(standard_params)
    for params in val_parameters:
        standard_params = {key: 0.0 for key in possible_unique_parameter_keys}
        for key, value in params.items():
            standard_params[key] = value
        standard_val_parameters.append(standard_params)
    for params in train_parameters:
        standard_params = {key: 0.0 for key in possible_unique_parameter_keys}
        for key, value in params.items():
            standard_params[key] = value
        standard_train_parameters.append(standard_params)

    np.save(os.path.join(traffic_dataset_dir, "standard_test_parameters.npy"), np.array(standard_test_parameters))
    np.save(os.path.join(traffic_dataset_dir, "standard_val_parameters.npy"), np.array(standard_val_parameters))
    np.save(os.path.join(traffic_dataset_dir, "standard_train_parameters.npy"), np.array(standard_train_parameters))
    print("saved the standard parameters to the test set at: ", os.path.join(traffic_dataset_dir, "standard_test_parameters.npy"))
    print("saved the standard parameters to the val set at: ", os.path.join(traffic_dataset_dir, "standard_val_parameters.npy"))
    print("saved the standard parameters to the train set at: ", os.path.join(traffic_dataset_dir, "standard_train_parameters.npy"))