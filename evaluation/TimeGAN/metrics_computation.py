## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import warnings
import pickle as pkl 
warnings.filterwarnings("ignore")

# 3. Metrics
# from metrics.predictive_metrics import predictive_score_metrics, predictive_score_metrics_with_train_test_split
from metrics.discriminative_metrics import discriminative_score_metrics, discriminative_score_metrics_with_train_test_split
import tensorflow as tf
 
def get_real_and_synthetic_paths(dataset_name):
    if dataset_name == "air_quality":
        real_timeseries_dir = ''
        synthetic_timeseries_dir = ''
    if dataset_name == "air_quality_conditional":
        real_timeseries_dir = ''
        synthetic_timeseries_dir = ''
    if dataset_name == "traffic":
        real_timeseries_dir = ''
        synthetic_timeseries_dir = ''
    if dataset_name == "traffic_conditional":
        real_timeseries_dir = ''
        synthetic_timeseries_dir = ''
    if dataset_name == "stocks":
        real_timeseries_dir = ''
        synthetic_timeseries_dir = ''
    if dataset_name == "waveforms":
        real_timeseries_dir = ''
        synthetic_timeseries_dir = ''
    
    return real_timeseries_dir, synthetic_timeseries_dir

datasets = ['air_quality', 'air_quality_conditional', 'traffic', 'traffic_conditional', 'stocks', 'waveforms']

dataset_name = sys.argv[1]

ds_dict = {}
for dataset in datasets:
    if dataset != dataset_name:
        continue
    real_timeseries_dir, synthetic_timeseries_dir = get_real_and_synthetic_paths(dataset)

    real_train_timeseries = np.load(os.path.join(real_timeseries_dir, 'train_timeseries.npy'))
    real_val_timeseries = np.load(os.path.join(real_timeseries_dir, 'val_timeseries.npy'))
    real_test_timeseries = np.load(os.path.join(real_timeseries_dir, 'test_timeseries.npy'))


    split_real_train_timeseries = np.transpose(real_train_timeseries, (0, 2, 1))
    split_real_val_timeseries = np.transpose(real_val_timeseries, (0, 2, 1))
    split_real_test_timeseries = np.transpose(real_test_timeseries, (0, 2, 1))
        
    train_synthetic_experiment_dir = os.path.join(synthetic_timeseries_dir, 'train')
    val_synthetic_experiment_dir = os.path.join(synthetic_timeseries_dir, 'val')
    test_synthetic_experiment_dir = os.path.join(synthetic_timeseries_dir, 'test')
    
    num_train_timeseries_files = len([f for f in os.listdir(train_synthetic_experiment_dir) if 'timeseries' in f])
    num_val_timeseries_files = len([f for f in os.listdir(val_synthetic_experiment_dir) if 'timeseries' in f])
    num_test_timeseries_files = len([f for f in os.listdir(test_synthetic_experiment_dir) if 'timeseries' in f])
        
    train_synthetic_timeseries_list = [os.path.join(train_synthetic_experiment_dir, 'timeseries_%d.npy' % i) for i in range(num_train_timeseries_files)]
    val_synthetic_timeseries_list = [os.path.join(val_synthetic_experiment_dir, 'timeseries_%d.npy' % i) for i in range(num_val_timeseries_files)]
    test_synthetic_timeseries_list = [os.path.join(test_synthetic_experiment_dir, 'timeseries_%d.npy' % i) for i in range(num_test_timeseries_files)]
        
    train_synthetic_timeseries = np.concatenate([np.load(synthetic_timeseries_file) for synthetic_timeseries_file in train_synthetic_timeseries_list], axis=0)
    val_synthetic_timeseries = np.concatenate([np.load(synthetic_timeseries_file) for synthetic_timeseries_file in val_synthetic_timeseries_list], axis=0)
    test_synthetic_timeseries = np.concatenate([np.load(synthetic_timeseries_file) for synthetic_timeseries_file in test_synthetic_timeseries_list], axis=0)

    split_train_synthetic_timeseries = np.transpose(train_synthetic_timeseries, (0, 2, 1))
    split_val_synthetic_timeseries = np.transpose(val_synthetic_timeseries, (0, 2, 1))
    split_test_synthetic_timeseries = np.transpose(test_synthetic_timeseries, (0, 2, 1))
        
    if split_real_train_timeseries.shape[-1] == 1:
        split_real_train_timeseries = np.concatenate([split_real_train_timeseries,] * 6, axis=-1)
        split_real_val_timeseries = np.concatenate([split_real_val_timeseries,] * 6, axis=-1)
        split_real_test_timeseries = np.concatenate([split_real_test_timeseries,] * 6, axis=-1)
    if split_train_synthetic_timeseries.shape[-1] == 1:
        split_train_synthetic_timeseries = np.concatenate([split_train_synthetic_timeseries,] * 6, axis=-1)
        split_val_synthetic_timeseries = np.concatenate([split_val_synthetic_timeseries,] * 6, axis=-1)
        split_test_synthetic_timeseries = np.concatenate([split_test_synthetic_timeseries,] * 6, axis=-1)
        
    print('Real Train Time series shape:', split_real_train_timeseries.shape)
    print('Synthetic Train Time series shape:', split_train_synthetic_timeseries.shape)
    print('Real Val Time series shape:', split_real_val_timeseries.shape)
    print('Synthetic Val Time series shape:', split_val_synthetic_timeseries.shape)
    print('Real Test Time series shape:', split_real_test_timeseries.shape)
    print('Synthetic Test Time series shape:', split_test_synthetic_timeseries.shape)
    
    for seed in [0,1,2,3,4]:
        ds = discriminative_score_metrics_with_train_test_split(split_real_train_timeseries, split_train_synthetic_timeseries, split_real_test_timeseries, split_test_synthetic_timeseries, seed=seed)
        ds_dict[(dataset, seed)] = ds
        
        print('Discriminative Score for seed %d: %f' % (seed, ds))
            
    print(ds_dict)
  