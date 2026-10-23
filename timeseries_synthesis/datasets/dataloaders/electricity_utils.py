import numpy as np
import os


class ElectricityTransformationUtils:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config.electricity_dataset
        self.dataset_loc = self.dataset_config.log_dir

        self.per_user_scalers_for_gan = np.load(
            os.path.join(self.dataset_loc, "per_user_scalers_for_gan.npy"),
            allow_pickle=True,
        ).item()
        self.per_user_scalers = np.load(
            os.path.join(self.dataset_loc, "per_user_scalers.npy"), allow_pickle=True
        ).item()

    def convert_from_gan_to_normal(self, dataset_dict):
        timeseries = dataset_dict["timeseries"]
        discrete_conditions = dataset_dict["discrete_conditions"]
        continuous_conditions = dataset_dict["continuous_conditions"]
        batch_size = timeseries.shape[0]
        assert (
            timeseries.shape[0]
            == discrete_conditions.shape[0]
        )
        assert timeseries.shape[1] == self.dataset_config.num_channels
        assert timeseries.shape[2] == self.dataset_config.time_series_length
        reshaped_timeseries = np.einsum("bct->btc", timeseries)

        scaled_ts = []
        for batch_idx in range(batch_size):
            ts = reshaped_timeseries[batch_idx]
            user_one_hot = discrete_conditions[batch_idx][0:370]  # 370 users
            user_id = np.argmax(user_one_hot) + 1  # 1-indexed
            user_id = "MT_" + str(user_id).zfill(3)  # MT_001, MT_002, etc.
            mm_scaler = self.per_user_scalers_for_gan[user_id]
            norm_scaler = self.per_user_scalers[user_id]
            orig_ts = mm_scaler.inverse_transform(ts)
            norm_ts = norm_scaler.transform(orig_ts)
            scaled_ts.append(norm_ts)

        scaled_ts = np.stack(scaled_ts, axis=0)
        scaled_ts = np.einsum("btc->bct", scaled_ts)

        transformed_dataset_dict = {
            "timeseries": scaled_ts,
            "discrete_conditions": discrete_conditions,
            "continuous_conditions": continuous_conditions,
        }
        return transformed_dataset_dict
