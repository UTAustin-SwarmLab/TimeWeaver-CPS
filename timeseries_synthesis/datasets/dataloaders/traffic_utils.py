import numpy as np
import os
import pickle


class TrafficTransformationUtils:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config.traffic_dataset
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

    def convert_from_normal_to_gan(self, dataset_dict):
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

        flattened_dataset_timeseries = self.scaler.inverse_transform(
            flattened_timeseries
        )
        flattened_gan_timeseries = self.gan_scaler.transform(
            flattened_dataset_timeseries
        )

        gan_reshaped_timeseries = flattened_gan_timeseries.reshape(
            batch_size,
            self.dataset_config.time_series_length,
            self.dataset_config.num_channels,
        )

        gan_timeseries = np.einsum("btc->bct", gan_reshaped_timeseries)

        assert (
            continuous_conditions.shape[2] == self.dataset_config.num_continuous_labels
        )
        flattened_continuous_conditions = continuous_conditions.reshape(
            batch_size * self.dataset_config.time_series_length,
            self.dataset_config.num_continuous_labels,
        )
        flattened_dataset_continuous_conditions = self.condn_scaler.inverse_transform(
            flattened_continuous_conditions
        )
        flattened_gan_continuous_conditions = self.gan_condn_scaler.transform(
            flattened_dataset_continuous_conditions
        )
        gan_continuous_conditions = flattened_gan_continuous_conditions.reshape(
            batch_size,
            self.dataset_config.time_series_length,
            self.dataset_config.num_continuous_labels,
        )

        transformed_dataset_dict = {
            "timeseries": gan_timeseries,
            "discrete_conditions": discrete_conditions,
            "continuous_conditions": gan_continuous_conditions,
        }

        return transformed_dataset_dict

    def convert_gradient_from_gan_to_normal(self, gradient):
        batch_size = gradient.shape[0]
        reshaped_gradient = np.einsum("bct->btc", gradient)
        flattened_gradient = reshaped_gradient.reshape(
            batch_size * self.dataset_config.time_series_length,
            self.dataset_config.num_channels,
        )


        std = self.scaler.var_ ** 0.5
        assert np.all(std == self.scaler.scale_)
        min = self.gan_scaler.data_min_
        max = self.gan_scaler.data_max_
        diff = max - min
        assert np.all(diff == self.gan_scaler.data_range_)

        grad_multiplier = 2 * std / diff
        normal_flattened_gradient = np.zeros_like(flattened_gradient)
        for i in range(normal_flattened_gradient.shape[1]):
            normal_flattened_gradient[:,i] = grad_multiplier[i] * flattened_gradient[:,i]
        normal_reshaped_gradient = normal_flattened_gradient.reshape(
            batch_size,
            self.dataset_config.time_series_length,
            self.dataset_config.num_channels,
        )

        normal_gradient = np.einsum("btc->bct", normal_reshaped_gradient)

        return normal_gradient
