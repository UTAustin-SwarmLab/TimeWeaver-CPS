import numpy as np
import os
import pickle


class AirQualityTransformationUtils:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config.air_quality_dataset
        self.dataset_loc = self.dataset_config.log_dir

        # load the scalers using pickle
        self.gan_scaler = pickle.load(
            open(os.path.join(self.dataset_loc, "gan_scaler.pkl"), "rb")
        )
        self.scaler = pickle.load(
            open(os.path.join(self.dataset_loc, "scaler.pkl"), "rb")
        )

        self.condn_scaler = pickle.load(
            open(os.path.join(self.dataset_loc, "condn_scaler.pkl"), "rb")
        )
        self.gan_condn_scaler = pickle.load(
            open(os.path.join(self.dataset_loc, "gan_condn_scaler.pkl"), "rb")
        )

        self.channel_names = ["pm_25", "pm_10", "so_2", "no_2", "co", "o_3"]

    def convert_from_gan_to_normal(self, dataset_dict):
        timeseries = dataset_dict["timeseries"]
        discrete_conditions = dataset_dict["discrete_conditions"]
        continuous_conditions = dataset_dict["continuous_conditions"]
        batch_size = timeseries.shape[0]
        assert timeseries.shape[0] == discrete_conditions.shape[0]
        assert timeseries.shape[1] == self.dataset_config.num_channels
        assert timeseries.shape[2] == self.dataset_config.time_series_length
        reshaped_timeseries = np.einsum("bct->btc", timeseries)
        flattened_timeseries = reshaped_timeseries.reshape(
            batch_size * self.dataset_config.time_series_length,
            self.dataset_config.num_channels,
        )

        flattened_dataset_timeseries = self.gan_scaler.inverse_transform(
            flattened_timeseries
        )
        flattened_normal_timeseries = self.scaler.transform(
            flattened_dataset_timeseries
        )

        normal_reshaped_timeseries = flattened_normal_timeseries.reshape(
            batch_size,
            self.dataset_config.time_series_length,
            self.dataset_config.num_channels,
        )

        normal_timeseries = np.einsum("btc->bct", normal_reshaped_timeseries)

        assert (
            continuous_conditions.shape[2] == self.dataset_config.num_continuous_labels
        )
        flattened_continuous_conditions = continuous_conditions.reshape(
            batch_size * self.dataset_config.time_series_length,
            self.dataset_config.num_continuous_labels,
        )
        flattened_dataset_continuous_conditions = (
            self.gan_condn_scaler.inverse_transform(flattened_continuous_conditions)
        )
        flattened_normal_continuous_conditions = self.condn_scaler.transform(
            flattened_dataset_continuous_conditions
        )
        normal_continuous_conditions = flattened_normal_continuous_conditions.reshape(
            batch_size,
            self.dataset_config.time_series_length,
            self.dataset_config.num_continuous_labels,
        )

        transformed_dataset_dict = {
            "timeseries": normal_timeseries,
            "discrete_conditions": discrete_conditions,
            "continuous_conditions": normal_continuous_conditions,
        }

        return transformed_dataset_dict

