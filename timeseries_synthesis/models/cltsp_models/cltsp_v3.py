from typing import List, Dict, Tuple
import torch
import numpy as np
import pickle
import os
from timeseries_synthesis.utils.basic_utils import (
    OKBLUE,
    ENDC,
)
from timeseries_synthesis.models.cltsp_models.utils import (
    TimeSeriesEncoder,
    ConditionEncoder,
)


class CLTSP_v3(torch.nn.Module):
    def __init__(self, cltsp_config, dataset_config, device):
        super(CLTSP_v3, self).__init__()
        self.cltsp_config = cltsp_config
        self.dataset_config = dataset_config
        self.device = device
        self.timeseries_encoder = TimeSeriesEncoder(
            cltsp_config=self.cltsp_config,
            dataset_config=self.dataset_config,
            device=self.device,
        )
        self.condition_encoder = ConditionEncoder(
            cltsp_config=self.cltsp_config,
            dataset_config=self.dataset_config,
            device=self.device,
        )

        if (
            self.dataset_config.num_discrete_labels == 0
            and self.dataset_config.num_continuous_labels == 0
        ):
            self.metadata_available = False
        else:
            self.metadata_available = True

    def forward(self, input) -> Tuple[torch.tensor, torch.tensor]:
        timeseries_input = input["timeseries_input"]
        discrete_condition_input = input["discrete_condition_input"]
        continuous_condition_input = input["continuous_condition_input"]
 
        # print("within forward function of CLTSP_v1")
        # print(
        #     timeseries_input.shape,
        #     discrete_condition_input.shape,
        #     continuous_condition_input.shape,
        # )

        timeseries_embedding = self.timeseries_encoder(timeseries_input)

        if self.metadata_available:
            condition_embedding = self.condition_encoder(
                discrete_condition_input, continuous_condition_input
            )
            return timeseries_embedding, condition_embedding
        else:
            return timeseries_embedding

    def prepare_training_input(self, train_batch):
        # timeseries
        timeseries_full = train_batch["timeseries_full"].float().to(self.device)
        timeseries_full = timeseries_full.permute(0, 2, 1)
        assert (
            timeseries_full.shape[2] == self.dataset_config.num_channels
        ), "The number of input features is not correct"

        actual_horizon = self.dataset_config.time_series_length
        required_horizon = self.dataset_config.required_time_series_length
        assert required_horizon <= actual_horizon, "required_horizon > actual_horizon"

        if self.metadata_available:
            # discrete label embedding
            discrete_label_embedding = (
                train_batch["discrete_label_embedding"].float().to(self.device)
            )
            if len(discrete_label_embedding.shape) == 2:
                # we will enter here only if we have constant discrete labels
                # so we repeat the discrete label embedding along the time dimension, that is the first dimension
                discrete_label_embedding = discrete_label_embedding.unsqueeze(1)
                discrete_label_embedding = discrete_label_embedding.repeat(
                    1, actual_horizon, 1
                )
                assert torch.all(
                    discrete_label_embedding[:, 0, :]
                    == discrete_label_embedding[:, 1, :]
                ), "discrete label embedding is not constant"

            assert discrete_label_embedding.shape[1] == actual_horizon, "Wrong shape"
            assert (
                discrete_label_embedding.shape[2]
                == self.dataset_config.num_discrete_labels
            ), "The number of discrete labels is not correct"

            # continuous label embedding
            continuous_label_embedding = (
                train_batch["continuous_label_embedding"].float().to(self.device)
            )
            if len(continuous_label_embedding.shape) == 2:
                # we will enter here only if we have constant continuous labels
                # so we repeat the continuous label embedding along the time dimension, that is the first dimension
                continuous_label_embedding = continuous_label_embedding.unsqueeze(1)
                continuous_label_embedding = continuous_label_embedding.repeat(
                    1, actual_horizon, 1
                )
                assert torch.all(
                    continuous_label_embedding[:, 0, :]
                    == continuous_label_embedding[:, 1, :] 
                ), "continuous label embedding is not constant"
        else:
            discrete_label_embedding = torch.zeros(
                timeseries_full.shape[0], actual_horizon, 1
            ).to(self.device) 
            continuous_label_embedding = torch.zeros(
                timeseries_full.shape[0], actual_horizon, 1
            ).to(self.device)

            # these are dummy, they will not be used anyways

        # obtaining random patches
        timeseries_list = []
        discrete_label_embedding_list = []
        continuous_label_embedding_list = []
        num_positive_samples = self.cltsp_config.num_positive_samples
        random_indices = np.random.randint(
            0, actual_horizon - required_horizon, size=num_positive_samples
        )
        for index in random_indices:
            timeseries_list.append(timeseries_full[:, index : index + required_horizon])
            discrete_label_embedding_list.append(
                discrete_label_embedding[:, index : index + required_horizon]
            )
            continuous_label_embedding_list.append(
                continuous_label_embedding[:, index : index + required_horizon]
            )
        timeseries_tensor = torch.cat(timeseries_list, dim=0)
        discrete_label_embeddings_tensor = torch.cat(
            discrete_label_embedding_list, dim=0
        )
        continuous_label_embeddings_tensor = torch.cat(
            continuous_label_embedding_list, dim=0
        )
        bs = timeseries_full.shape[0]
        assert torch.all(
            timeseries_tensor[:bs]
            == timeseries_full[
                :, random_indices[0] : random_indices[0] + required_horizon
            ]
        )

        # print(OKBLUE + "timeseries_tensor.shape: ", timeseries_tensor.shape, ENDC)
        # print(
        #     OKBLUE + "discrete_label_embeddings_tensor.shape: ",
        #     discrete_label_embeddings_tensor.shape,
        #     ENDC,
        # )
        # print(
        #     OKBLUE + "continuous_label_embeddings_tensor.shape: ",
        #     continuous_label_embeddings_tensor.shape,
        #     ENDC,
        # )

        input = {
            "timeseries_input": timeseries_tensor,
            "discrete_condition_input": discrete_label_embeddings_tensor,
            "continuous_condition_input": continuous_label_embeddings_tensor,
        }
        return input

    def get_timeseries_embedding(self, timeseries):
        return self.timeseries_encoder(timeseries)

    def get_condition_embedding(self, discrete_condition, continuous_condition):
        return self.condition_encoder(discrete_condition, continuous_condition)
