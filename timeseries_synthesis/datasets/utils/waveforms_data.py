import os
import hydra
import numpy as np
from itertools import product

from tqdm import tqdm
from time import sleep

from typing import Optional
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from scipy import signal


from timeseries_synthesis.utils.basic_utils import (
    OKBLUE,
    OKYELLOW,
    ENDC,
    seed_everything,
)


class WaveformDataGenerator:
    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.config = config

        self.final_path = os.path.join(self.config.base_path, self.config.save_path)
        os.makedirs(self.final_path, exist_ok=True)

        self.config = config
        self.time_stamps = np.linspace(
            self.config.start_time, self.config.end_time, self.config.horizon
        )

        self.minimum_amplitude = self.config.amplitude_min
        self.maximum_amplitude = self.config.amplitude_max
        self.amplitude_interval = self.config.del_amplitude
        self.amplitudes = list(
            np.arange(
                self.minimum_amplitude,
                self.maximum_amplitude + self.amplitude_interval,
                self.amplitude_interval,
            )
        )
        print(OKBLUE + "Amplitudes: %s" % self.amplitudes + ENDC)

        self.minimum_frequency = self.config.frequency_min
        self.maximum_frequency = self.config.frequency_max
        self.frequency_interval = self.config.del_frequency
        self.frequencies = list(
            np.arange(
                self.minimum_frequency,
                self.maximum_frequency + self.frequency_interval,
                self.frequency_interval,
            )
        )

        self.minimum_phase = self.config.phase_min
        self.maximum_phase = self.config.phase_max
        self.phase_interval = self.config.del_phase
        self.phases = list(
            np.arange(self.minimum_phase, self.maximum_phase, self.phase_interval)
        )

        self.params = list(product(self.amplitudes, self.frequencies, self.phases))

        self.waveform_indicators = {
            "sine": 1,
            "triangle": 2,
            "square": 3,
            "sawtooth": 4,
        }

    def generate_sine_data(self):
        print(OKYELLOW + "Generating sine data" + ENDC)
        self.sine_data = []
        self.sine_labels = []
        for params in tqdm(self.params, total=len(self.params)):
            sleep(0.01)
            data = params[0] * np.sin(
                2 * np.pi * params[1] * self.time_stamps + params[2]
            )
            data = np.expand_dims(data, axis=0)
            data = data.astype(np.float16)
            self.sine_data.append(data)

            params = list(params)
            params.append(self.waveform_indicators["sine"])
            params = np.expand_dims(params, axis=0)
            self.sine_labels.append(params)

        self.sine_data = np.concatenate(self.sine_data, axis=0)
        self.sine_labels = np.concatenate(self.sine_labels, axis=0)

        print(OKBLUE + "Sine data shape: %d" % self.sine_data.shape[0] + ENDC)
        print(OKBLUE + "Sine labels shape: %d" % self.sine_labels.shape[0] + ENDC)

    def generate_triangle_data(self):
        print(OKYELLOW + "Generating triangle data" + ENDC)
        self.triangle_data = []
        self.triangle_labels = []
        for params in tqdm(self.params, total=len(self.params)):
            sleep(0.01)
            data = params[0] * signal.sawtooth(
                2 * np.pi * params[1] * self.time_stamps + params[2], 0.5
            )
            data = np.expand_dims(data, axis=0)
            data = data.astype(np.float16)
            self.triangle_data.append(data)

            params = list(params)
            params.append(self.waveform_indicators["triangle"])
            params = np.expand_dims(params, axis=0).astype(np.float16)
            self.triangle_labels.append(params)

        self.triangle_data = np.concatenate(self.triangle_data, axis=0)
        self.triangle_labels = np.concatenate(self.triangle_labels, axis=0)

        print(OKBLUE + "Triangle data shape: %d" % self.triangle_data.shape[0] + ENDC)
        print(
            OKBLUE + "Triangle labels shape: %d" % self.triangle_labels.shape[0] + ENDC
        )

    def generate_square_data(self):
        print(OKYELLOW + "Generating square data" + ENDC)
        self.square_data = []
        self.square_labels = []
        for params in tqdm(self.params, total=len(self.params)):
            sleep(0.01)
            data = params[0] * signal.square(
                2 * np.pi * params[1] * self.time_stamps + params[2]
            )
            data = np.expand_dims(data, axis=0)
            data = data.astype(np.float16)
            self.square_data.append(data)

            params = list(params)
            params.append(self.waveform_indicators["square"])
            params = np.expand_dims(params, axis=0).astype(np.float16)
            self.square_labels.append(params)

        self.square_data = np.concatenate(self.square_data, axis=0)
        self.square_labels = np.concatenate(self.square_labels, axis=0)

        print(OKBLUE + "Square data shape: %d" % self.square_data.shape[0] + ENDC)
        print(OKBLUE + "Square labels shape: %d" % self.square_labels.shape[0] + ENDC)

    def generate_sawtooth_data(self):
        print(OKYELLOW + "Generating sawtooth data" + ENDC)
        self.sawtooth_data = []
        self.sawtooth_labels = []
        for params in tqdm(self.params, total=len(self.params)):
            sleep(0.01)
            data = params[0] * signal.sawtooth(
                2 * np.pi * params[1] * self.time_stamps + params[2], 0.5
            )
            data = np.expand_dims(data, axis=0)
            data = data.astype(np.float16)
            self.sawtooth_data.append(data)

            params = list(params)
            params.append(self.waveform_indicators["sawtooth"])
            params = np.expand_dims(params, axis=0).astype(np.float16)
            self.sawtooth_labels.append(params)

        self.sawtooth_data = np.concatenate(self.sawtooth_data, axis=0)
        self.sawtooth_labels = np.concatenate(self.sawtooth_labels, axis=0)

        print(OKBLUE + "Sawtooth data shape: %d" % self.sawtooth_data.shape[0] + ENDC)
        print(
            OKBLUE + "Sawtooth labels shape: %d" % self.sawtooth_labels.shape[0] + ENDC
        )

    def generate_data(self):
        print(OKYELLOW + "Generating data" + ENDC)
        self.generate_sine_data()

        self.timeseries = np.expand_dims(self.sine_data, axis=1)
        self.discrete_labels = self.sine_labels[:, -1]
        self.continuous_conditions = self.sine_labels[:, :-1]

        self.discrete_conditions = np.zeros((self.discrete_labels.shape[0], 4))
        for idx in range(self.discrete_labels.shape[0]):
            self.discrete_conditions[idx, int(self.discrete_labels[idx] - 1)] = 1

        print(
            self.timeseries.shape,
            self.discrete_conditions.shape,
            self.continuous_conditions.shape,
        )

        num_samples = self.timeseries.shape[0]
        indices = np.arange(num_samples)
        random_indices = np.random.permutation(indices)

        train_indices = random_indices[: int(0.8 * num_samples)]
        val_indices = random_indices[int(0.8 * num_samples) : int(0.9 * num_samples)]
        test_indices = random_indices[int(0.9 * num_samples) :]

        self.train_timeseries = self.timeseries[train_indices]
        self.val_timeseries = self.timeseries[val_indices]
        self.test_timeseries = self.timeseries[test_indices]

        self.train_discrete_conditions = self.discrete_conditions[train_indices]
        self.val_discrete_conditions = self.discrete_conditions[val_indices]
        self.test_discrete_conditions = self.discrete_conditions[test_indices]

        self.train_continuous_conditions = self.continuous_conditions[train_indices]
        self.val_continuous_conditions = self.continuous_conditions[val_indices]
        self.test_continuous_conditions = self.continuous_conditions[test_indices]

        print(OKBLUE + "Dataset shape: %d" % self.timeseries.shape[0] + ENDC)

    def save_data(self):
        train_timeseries_save_loc = os.path.join(
            self.final_path, "train_timeseries.npy"
        )
        print(
            OKYELLOW
            + "Saving train timeseries to %s" % train_timeseries_save_loc
            + ENDC
        )
        np.save(train_timeseries_save_loc, self.train_timeseries)

        val_timeseries_save_loc = os.path.join(self.final_path, "val_timeseries.npy")
        print(OKYELLOW + "Saving val timeseries to %s" % val_timeseries_save_loc + ENDC)
        np.save(val_timeseries_save_loc, self.val_timeseries)

        test_timeseries_save_loc = os.path.join(self.final_path, "test_timeseries.npy")
        print(
            OKYELLOW + "Saving test timeseries to %s" % test_timeseries_save_loc + ENDC
        )
        np.save(test_timeseries_save_loc, self.test_timeseries)

        train_discrete_conditions_save_loc = os.path.join(
            self.final_path, "train_discrete_conditions.npy"
        )
        print(
            OKYELLOW
            + "Saving train discrete conditions to %s"
            % train_discrete_conditions_save_loc
            + ENDC
        )
        np.save(train_discrete_conditions_save_loc, self.train_discrete_conditions)

        val_discrete_conditions_save_loc = os.path.join(
            self.final_path, "val_discrete_conditions.npy"
        )
        print(
            OKYELLOW
            + "Saving val discrete conditions to %s" % val_discrete_conditions_save_loc
            + ENDC
        )
        np.save(val_discrete_conditions_save_loc, self.val_discrete_conditions)

        test_discrete_conditions_save_loc = os.path.join(
            self.final_path, "test_discrete_conditions.npy"
        )
        print(
            OKYELLOW
            + "Saving test discrete conditions to %s"
            % test_discrete_conditions_save_loc
            + ENDC
        )
        np.save(test_discrete_conditions_save_loc, self.test_discrete_conditions)

        train_continuous_conditions_save_loc = os.path.join(
            self.final_path, "train_continuous_conditions.npy"
        )
        print(
            OKYELLOW
            + "Saving train continuous conditions to %s"
            % train_continuous_conditions_save_loc
            + ENDC
        )
        np.save(train_continuous_conditions_save_loc, self.train_continuous_conditions)

        val_continuous_conditions_save_loc = os.path.join(
            self.final_path, "val_continuous_conditions.npy"
        )
        print(
            OKYELLOW
            + "Saving val continuous conditions to %s"
            % val_continuous_conditions_save_loc
            + ENDC
        )
        np.save(val_continuous_conditions_save_loc, self.val_continuous_conditions)

        test_continuous_conditions_save_loc = os.path.join(
            self.final_path, "test_continuous_conditions.npy"
        )
        print(
            OKYELLOW
            + "Saving test continuous conditions to %s"
            % test_continuous_conditions_save_loc
            + ENDC
        )
        np.save(test_continuous_conditions_save_loc, self.test_continuous_conditions)


@hydra.main(config_path="../../../configs/", version_base="1.1")
def main(config: DictConfig):
    seed_everything(config.seed)
    data_generator = WaveformDataGenerator(config)
    data_generator.generate_data()
    data_generator.save_data()


if __name__ == "__main__":
    main()
