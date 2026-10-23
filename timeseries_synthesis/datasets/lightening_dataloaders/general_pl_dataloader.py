import torch
import os
import numpy as np

from timeseries_synthesis.datasets.lightening_dataloaders.dataloader_utils import (
    BaseDataModule,
    TimeWeaverBaseDataModule,
)

from timeseries_synthesis.datasets.utils.dataset_utils import (
    get_dataset_paths,
)
from timeseries_synthesis.utils.basic_utils import get_dataset_config
from timeseries_synthesis.utils.basic_utils import (
    OKBLUE,
    ENDC,
)


class TimeweaverDataset(torch.utils.data.Dataset):
    def __init__(
        self, config: dict, train: bool = False, val: bool = False, test: bool = False
    ) -> None:
        torch.manual_seed(config.seed)
        dataset_config = get_dataset_config(config)
        dataset_log_dir = dataset_config.log_dir
        self.horizon = dataset_config.time_series_length
        self.num_channels = dataset_config.num_channels
        experiment = config.experiment

        (
            self.timeseries_dataset_loc,
            self.discrete_conditions_loc,
            self.continuous_conditions_loc,
        ) = get_dataset_paths(
            dataset_log_dir,
            experiment,
            train=train,
            val=val,
            test=test,
        )
        print(
            OKBLUE + "The dataset location is : " + self.timeseries_dataset_loc + ENDC
        )
        print(
            OKBLUE
            + "The discrete labels location is : "
            + self.discrete_conditions_loc
            + ENDC
        )
        print(
            OKBLUE
            + "The continuous labels location is : "
            + self.continuous_conditions_loc
            + ENDC
        )

        # load the dataset if the path exists

        if os.path.exists(self.timeseries_dataset_loc):
            self.timeseries_dataset = np.load(
                self.timeseries_dataset_loc, allow_pickle=True
            )
        else:
            self.timeseries_dataset = np.array([])

        if os.path.exists(self.discrete_conditions_loc):
            self.discrete_conditions = np.load(
                self.discrete_conditions_loc, allow_pickle=True
            )
        else:
            self.discrete_conditions = np.array([])

        if os.path.exists(self.continuous_conditions_loc):
            self.continuous_conditions = np.load(
                self.continuous_conditions_loc, allow_pickle=True
            )
        else:
            self.continuous_conditions = np.array([])

        if self.continuous_conditions.shape[0] == 0:
            self.continuous_conditions_exist = False
        else:
            self.continuous_conditions_exist = True

        if self.discrete_conditions.shape[0] == 0:
            self.discrete_conditions_exist = False
        else:
            self.discrete_conditions_exist = True
        print(self.timeseries_dataset.shape)
        assert (
            self.timeseries_dataset.shape[-1] == self.horizon
        ), "The horizon is not correct"
        assert (
            self.timeseries_dataset.shape[-2] == self.num_channels
        ), "The number of channels is not correct"

    def __len__(self):
        return self.timeseries_dataset.shape[0]

    def __getitem__(self, index):
        timeseries_full = self.timeseries_dataset[index]

        if self.discrete_conditions_exist:
            discrete_label_embedding = self.discrete_conditions[index]
        else:
            discrete_label_embedding = np.array([])

        if self.continuous_conditions_exist:
            continuous_label_embedding = self.continuous_conditions[index]
        else:
            continuous_label_embedding = np.array([])

        data_dict = {
            "timeseries_full": timeseries_full,
            "discrete_label_embedding": discrete_label_embedding,
            "continuous_label_embedding": continuous_label_embedding,
        }
        return data_dict


class TimeweaverDataLoader(TimeWeaverBaseDataModule):
    def __init__(self, config):
        print("Loading train dataset")
        train_dataset = TimeweaverDataset(config, train=True)
        print("Loading val dataset")
        val_dataset = TimeweaverDataset(config, val=True)
        print("Loading test dataset")
        test_dataset = TimeweaverDataset(config, test=True)
        super().__init__(config, train_dataset, val_dataset, test_dataset)
