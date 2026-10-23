import os
import torch
import numpy as np
import hydra
from tqdm import tqdm
from time import sleep
import logging
from scipy.linalg import sqrtm
import lightning as L

from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger

from timeseries_synthesis.utils.basic_utils import (
    seed_everything,
    import_from,
    get_dataloader_purpose,
    get_dataset_config,
    generate_synthesis_experiment_details,
    OKBLUE,
    OKYELLOW,
    ENDC,
)


from timeseries_synthesis.metrics.torchmetrics import FID_Metric


def get_similarity_checker_embeddings(
    similarity_checker,
    synthetic_timeseries,
    original_timeseries,
    discrete_conditions,
    continuous_conditions,
    dataset_config,
):
    synthetic_timeseries = (
        torch.from_numpy(synthetic_timeseries).float().to(similarity_checker.device)
    )
    original_timeseries = (
        torch.from_numpy(original_timeseries).float().to(similarity_checker.device)
    )
    discrete_conditions = (
        torch.from_numpy(discrete_conditions).float().to(similarity_checker.device)
    )
    continuous_conditions = (
        torch.from_numpy(continuous_conditions).float().to(similarity_checker.device)
    )
    
    required_timeseries_length = dataset_config.required_time_series_length
    actual_timeseries_length = dataset_config.time_series_length

    batched_synthetic_timeseries = split_into_batches(synthetic_timeseries, 256)
    batched_original_timeseries = split_into_batches(original_timeseries, 256)
    batched_discrete_conditions = split_into_batches(discrete_conditions, 256)
    batch_continuous_conditions = split_into_batches(continuous_conditions, 256)

    num_batches = len(batched_synthetic_timeseries)

    condition_embedding_list = []
    synthetic_timeseries_embedding_list = []
    original_timeseries_embedding_list = []
    # corresponding_labels = []

    for batch_idx in tqdm(range(num_batches), total=num_batches):
        sleep(0.001)
        synthetic_single_batch = batched_synthetic_timeseries[batch_idx]
        original_single_batch = batched_original_timeseries[batch_idx]
        if similarity_checker.metadata_available:   
            discrete_single_batch = batched_discrete_conditions[batch_idx]
            continuous_single_batch = batch_continuous_conditions[batch_idx]

        for tidx in range(
            0,
            actual_timeseries_length - required_timeseries_length + 1):
            synthetic_single_batch_part_timeseries = synthetic_single_batch[
                :, :, tidx : tidx + required_timeseries_length
            ]
            original_single_batch_part_timeseries = original_single_batch[
                :, :, tidx : tidx + required_timeseries_length
            ]
            synthetic_single_batch_part_timeseries = torch.einsum(
                "ijk->ikj", synthetic_single_batch_part_timeseries
            )
            original_single_batch_part_timeseries = torch.einsum(
                "ijk->ikj", original_single_batch_part_timeseries
            )

            synthetic_single_batch_part_timeseries_embedding = (
                similarity_checker.cltsp_model.get_timeseries_embedding(
                    synthetic_single_batch_part_timeseries
                )
            )
            original_single_batch_part_timeseries_embedding = (
                similarity_checker.cltsp_model.get_timeseries_embedding(
                    original_single_batch_part_timeseries
                )
            )

            synthetic_timeseries_embedding_list.append(
                synthetic_single_batch_part_timeseries_embedding
            )
            original_timeseries_embedding_list.append(
                original_single_batch_part_timeseries_embedding
            )
            
            if similarity_checker.metadata_available:
                # only if metadata is available
                if dataset_config.num_discrete_labels > 0:
                    if len(discrete_single_batch.shape) == 2:
                        discrete_single_batch = discrete_single_batch.unsqueeze(1)
                        discrete_single_batch = discrete_single_batch.repeat(
                            1, dataset_config.time_series_length, 1
                        )
                discrete_single_batch_part = discrete_single_batch[
                    :, tidx : tidx + required_timeseries_length
                ]

                if dataset_config.num_continuous_labels > 0:
                    if len(continuous_single_batch.shape) == 2:
                        continuous_single_batch = continuous_single_batch.unsqueeze(1)
                        continuous_single_batch = continuous_single_batch.repeat(
                            1, dataset_config.time_series_length, 1
                        )
                continuous_single_batch_part = continuous_single_batch[
                    :, tidx : tidx + required_timeseries_length
                ]
                
                if continuous_single_batch_part.shape[0] != discrete_single_batch_part.shape[0]:
                    min_batch_size = min(continuous_single_batch_part.shape[0], discrete_single_batch_part.shape[0])
                    discrete_single_batch_part = discrete_single_batch_part[:min_batch_size]
                    continuous_single_batch_part = continuous_single_batch_part[:min_batch_size]

                single_batch_part_condition_embedding = (
                    similarity_checker.cltsp_model.get_condition_embedding(
                        discrete_single_batch_part, continuous_single_batch_part
                    )
                )
                condition_embedding_list.append(single_batch_part_condition_embedding)

            # corresponding_labels.append(discrete_single_batch)

    if similarity_checker.metadata_available:
        condition_embedding_tensor = torch.cat(condition_embedding_list, dim=0)
    else:
        condition_embedding_tensor = None
    synthetic_timeseries_embedding_tensor = torch.cat(
        synthetic_timeseries_embedding_list, dim=0
    )
    original_timeseries_embedding_tensor = torch.cat(
        original_timeseries_embedding_list, dim=0
    )
    # corresponding_labels = torch.cat(corresponding_labels, dim=0)

    return (
        synthetic_timeseries_embedding_tensor,
        original_timeseries_embedding_tensor,
        condition_embedding_tensor,
    )


def load_pretrained_model(model_type, model_checkpoint_path, config):
    print(OKBLUE + "loading model from checkpoint" + ENDC)
    model = model_type.load_from_checkpoint(
        model_checkpoint_path, config=config, scaler=None, strict=True
    )
    print(OKBLUE + "model loaded from checkpoint" + ENDC)
    return model


def split_into_batches(tensor, batch_size): 
    num_samples = tensor.shape[0]
    batches = []
    start_idx = 0
    end_idx = batch_size
    while end_idx <= num_samples:
        batch = tensor[start_idx:end_idx]
        batches.append(batch)
        start_idx = end_idx
        end_idx += batch_size
    batches.append(tensor[start_idx:])
    return batches

@hydra.main(config_path="../../configs/constrained_generation_configs/", version_base="1.1")
def main(config: DictConfig):
    seed_everything(config.seed)
    pl_dataloader = import_from(
        f"timeseries_synthesis.datasets.lightening_dataloaders.{config.dataloader_file}",
        f"{config.dataloader_model}",
    )(config)
    get_dataloader_purpose(pl_dataloader)
    dataset_config = get_dataset_config(config)
    num_constraints = len(dataset_config.equality_constraints_to_extract)
    
    synthetic_data_dir = generate_synthesis_experiment_details(config.synthetic_dataset_dir, config, num_constraints, dataset_config.log_dir, eval_mode=True)['save_dir']
    print(
        OKYELLOW
        + "All the results will be stored in this directory: "
        + str(synthetic_data_dir)
        + ENDC
    )


    similarity_checker_type = import_from(
        f"timeseries_synthesis.models.lightening_modules.{config.similarity_checker_wrapper_model_file}_trainer",
        config.similarity_checker_wrapper_model_name,
    )

    similarity_checker = load_pretrained_model(
        similarity_checker_type,
        config.similarity_checker_wrapper_checkpoint_path,
        config,
    )
    L.seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    similarity_checker.eval()
    for parameter in similarity_checker.parameters():
        parameter.requires_grad = False

    test_dataset_obj = pl_dataloader.test_dataloader().dataset
    test_discrete_conditions = test_dataset_obj.discrete_conditions
    test_continuous_conditions = test_dataset_obj.continuous_conditions

    test_timeseries = test_dataset_obj.timeseries_dataset
    synthetic_test_timeseries_dir = os.path.join(synthetic_data_dir, "test")
    num_synthetic_test_timeseries_files = len([f for f in os.listdir(synthetic_test_timeseries_dir) if f.startswith('timeseries')])
    synthetic_test_timeseries_list = [np.load(os.path.join(synthetic_test_timeseries_dir, f"timeseries_{i}.npy")) for i in range(num_synthetic_test_timeseries_files)]
    synthetic_test_timeseries = np.concatenate(synthetic_test_timeseries_list, axis=0)


    (
        synthetic_timeseries_embedding_tensor,
        real_timeseries_embedding_tensor,
        condition_embedding_tensor,
    ) = get_similarity_checker_embeddings(
        similarity_checker=similarity_checker,
        synthetic_timeseries=synthetic_test_timeseries,
        original_timeseries=test_timeseries,
        discrete_conditions=test_discrete_conditions,
        continuous_conditions=test_continuous_conditions,
        dataset_config=dataset_config,
    )

    use_conditions = config.use_conditions
    
    if use_conditions:
        real_embeddings = real_timeseries_embedding_tensor.detach().cpu().numpy()
        synthetic_embeddings = synthetic_timeseries_embedding_tensor.detach().cpu().numpy()
        label_embeddings = condition_embedding_tensor.detach().cpu().numpy()

        test_jftsd_metric = FID_Metric(num_features=dataset_config.latent_dim * 2)

        real = np.concatenate([real_embeddings, label_embeddings], axis=1)
        fake = np.concatenate([synthetic_embeddings, label_embeddings], axis=1)


        real = torch.from_numpy(real).float()
        fake = torch.from_numpy(fake).float()

        test_jftsd_metric.update(real, real=True)
        test_jftsd_metric.update(fake, real=False)

        print(OKYELLOW + "The JFTSD score is : %f" % test_jftsd_metric.compute() + ENDC)
    
    else:
        real_embeddings = real_timeseries_embedding_tensor.detach().cpu().numpy()
        synthetic_embeddings = synthetic_timeseries_embedding_tensor.detach().cpu().numpy()

        test_jftsd_metric = FID_Metric(num_features=dataset_config.latent_dim)

        real = torch.from_numpy(real_embeddings).float()
        fake = torch.from_numpy(synthetic_embeddings).float()

        test_jftsd_metric.update(real, real=True)
        test_jftsd_metric.update(fake, real=False)

        print(OKYELLOW + "The FTSD score is : %f" % test_jftsd_metric.compute() + ENDC)

    return 0


if __name__ == "__main__":
    main()