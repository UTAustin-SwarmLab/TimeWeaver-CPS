import numpy as np
import os
import pickle


class StocksTransformationUtils:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config.stocks_dataset
        self.dataset_loc = self.dataset_config.log_dir

        # load the scalers using pickle
        self.gan_scaler = pickle.load(
            open(os.path.join(self.dataset_loc, "gan_scaler.pkl"), "rb")
        )
        self.scaler = pickle.load(
            open(os.path.join(self.dataset_loc, "scaler.pkl"), "rb")
        )

    def convert_from_gan_to_normal(self, timeseries):
        batch_size = timeseries.shape[0]
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
        return normal_timeseries