import numpy as np 
import os 
import sys

synthetic_data_root_dir = sys.argv[1]

synthetic_data_test_dir = os.path.join(synthetic_data_root_dir, "test")
synthetic_data_train_dir = os.path.join(synthetic_data_root_dir, "train")
synthetic_data_val_dir = os.path.join(synthetic_data_root_dir, "val")

num_synthetic_test_timeseries_files = len([f for f in os.listdir(synthetic_data_test_dir) if f.startswith('timeseries')])
synthetic_test_timeseries_list = [np.load(os.path.join(synthetic_data_test_dir, f"timeseries_{i}.npy")) for i in range(num_synthetic_test_timeseries_files)]
synthetic_test_timeseries = np.concatenate(synthetic_test_timeseries_list, axis=0)

num_synthetic_val_timeseries_files = len([f for f in os.listdir(synthetic_data_val_dir) if f.startswith('timeseries')])
synthetic_val_timeseries_list = [np.load(os.path.join(synthetic_data_val_dir, f"timeseries_{i}.npy")) for i in range(num_synthetic_val_timeseries_files)]
synthetic_val_timeseries = np.concatenate(synthetic_val_timeseries_list, axis=0)

num_synthetic_train_timeseries_files = len([f for f in os.listdir(synthetic_data_train_dir) if f.startswith('timeseries')])
synthetic_train_timeseries_list = [np.load(os.path.join(synthetic_data_train_dir, f"timeseries_{i}.npy")) for i in range(num_synthetic_train_timeseries_files)]
synthetic_train_timeseries = np.concatenate(synthetic_train_timeseries_list, axis=0)

num_test_discrete_conditions_files = len([f for f in os.listdir(synthetic_data_test_dir) if f.startswith('discrete_conditions')])
test_discrete_conditions_list = [np.load(os.path.join(synthetic_data_test_dir, f"discrete_conditions_{i}.npy")) for i in range(num_test_discrete_conditions_files)]
test_discrete_conditions = np.concatenate(test_discrete_conditions_list, axis=0)

num_val_discrete_conditions_files = len([f for f in os.listdir(synthetic_data_val_dir) if f.startswith('discrete_conditions')])
val_discrete_conditions_list = [np.load(os.path.join(synthetic_data_val_dir, f"discrete_conditions_{i}.npy")) for i in range(num_val_discrete_conditions_files)]
val_discrete_conditions = np.concatenate(val_discrete_conditions_list, axis=0)

num_train_discrete_conditions_files = len([f for f in os.listdir(synthetic_data_train_dir) if f.startswith('discrete_conditions')])
train_discrete_conditions_list = [np.load(os.path.join(synthetic_data_train_dir, f"discrete_conditions_{i}.npy")) for i in range(num_train_discrete_conditions_files)]
train_discrete_conditions = np.concatenate(train_discrete_conditions_list, axis=0)

num_test_continuous_conditions_files = len([f for f in os.listdir(synthetic_data_test_dir) if f.startswith('continuous_conditions')])
test_continuous_conditions_list = [np.load(os.path.join(synthetic_data_test_dir, f"continuous_conditions_{i}.npy")) for i in range(num_test_continuous_conditions_files)]
test_continuous_conditions = np.concatenate(test_continuous_conditions_list, axis=0)

num_val_continuous_conditions_files = len([f for f in os.listdir(synthetic_data_val_dir) if f.startswith('continuous_conditions')])
val_continuous_conditions_list = [np.load(os.path.join(synthetic_data_val_dir, f"continuous_conditions_{i}.npy")) for i in range(num_val_continuous_conditions_files)]
val_continuous_conditions = np.concatenate(val_continuous_conditions_list, axis=0)

num_train_continuous_conditions_files = len([f for f in os.listdir(synthetic_data_train_dir) if f.startswith('continuous_conditions')])
train_continuous_conditions_list = [np.load(os.path.join(synthetic_data_train_dir, f"continuous_conditions_{i}.npy")) for i in range(num_train_continuous_conditions_files)]
train_continuous_conditions = np.concatenate(train_continuous_conditions_list, axis=0)

print(f"synthetic_test_timeseries shape: {synthetic_test_timeseries.shape}")
print(f"synthetic_val_timeseries shape: {synthetic_val_timeseries.shape}")
print(f"synthetic_train_timeseries shape: {synthetic_train_timeseries.shape}")

print(f"test_discrete_conditions shape: {test_discrete_conditions.shape}")
print(f"val_discrete_conditions shape: {val_discrete_conditions.shape}")
print(f"train_discrete_conditions shape: {train_discrete_conditions.shape}")

print(f"test_continuous_conditions shape: {test_continuous_conditions.shape}")
print(f"val_continuous_conditions shape: {val_continuous_conditions.shape}")
print(f"train_continuous_conditions shape: {train_continuous_conditions.shape}")

synthetic_train_timeseries_loc = os.path.join(synthetic_data_root_dir, "train_timeseries.npy")
synthetic_val_timeseries_loc = os.path.join(synthetic_data_root_dir, "val_timeseries.npy")
synthetic_test_timeseries_loc = os.path.join(synthetic_data_root_dir, "test_timeseries.npy")

train_discrete_conditions_loc = os.path.join(synthetic_data_root_dir, "train_discrete_conditions.npy")
val_discrete_conditions_loc = os.path.join(synthetic_data_root_dir, "val_discrete_conditions.npy")
test_discrete_conditions_loc = os.path.join(synthetic_data_root_dir, "test_discrete_conditions.npy")

train_continuous_conditions_loc = os.path.join(synthetic_data_root_dir, "train_continuous_conditions.npy")
val_continuous_conditions_loc = os.path.join(synthetic_data_root_dir, "val_continuous_conditions.npy")
test_continuous_conditions_loc = os.path.join(synthetic_data_root_dir, "test_continuous_conditions.npy")

np.save(synthetic_train_timeseries_loc, synthetic_train_timeseries)
np.save(synthetic_val_timeseries_loc, synthetic_val_timeseries)
np.save(synthetic_test_timeseries_loc, synthetic_test_timeseries)

np.save(train_discrete_conditions_loc, train_discrete_conditions)
np.save(val_discrete_conditions_loc, val_discrete_conditions)
np.save(test_discrete_conditions_loc, test_discrete_conditions)

np.save(train_continuous_conditions_loc, train_continuous_conditions)
np.save(val_continuous_conditions_loc, val_continuous_conditions)
np.save(test_continuous_conditions_loc, test_continuous_conditions)