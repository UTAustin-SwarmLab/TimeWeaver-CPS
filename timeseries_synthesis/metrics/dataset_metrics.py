import numpy as np
from omegaconf import DictConfig
from enum import Enum
from sklearn.metrics.pairwise import (
    euclidean_distances,
    manhattan_distances,
    cosine_distances,
)

from timeseries_synthesis.metrics.distance_utils import (
    fid_scores,
)

from timeseries_synthesis.metrics.search_metrics import SearchMetrics, MetricType


class DatasetMetricType(Enum):
    """Dataset metric type"""

    one_to_one = "one_to_one"
    one_to_many = "one_to_many"
    fid = "fid"


class DatasetMetrics(SearchMetrics):
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def one_to_one_score(self, dataset1, dataset2, metrictype: MetricType) -> float:
        """Compute one to one score between two datasets. The datasets must have the same length.
        Args:
            dataset1 (np.ndarray): First dataset
            dataset2 (np.ndarray): Second dataset
            metrictype (MetricType): Metric type
        Returns:
            float: One to one score between two datasets
        """

        assert (
            dataset1.shape == dataset2.shape
        ), f"Shape of dataset1: {dataset1.shape}, Shape of dataset2: {dataset2.shape}"

        score = 0
        for i in range(len(dataset1)):
            score += self.distance_metric(dataset1[i], dataset2[i], metrictype)
        return score / len(dataset1)

    def one_to_many_score(self, dataset1, dataset2, metrictype: MetricType):
        """Compute one to many score between two datasets. The dataset1 should be 2dimensional and dataset2 should be 3 dimensional with the same length.
        Args:
            dataset1 (np.ndarray): First dataset (N_features)
            dataset2 (np.ndarray): Second dataset (N_data, N_sample_per_data, N_features)
            metrictype (MetricType): Metric type
        Returns:
            float: One to many score between two datasets
        """

        assert (
            len(dataset1.shape) == 2 and len(dataset2.shape) == 3
        ), f"Shape of dataset1: {dataset1.shape}, Shape of dataset2: {dataset2.shape}"

        assert (
            dataset1.shape[0] == dataset2.shape[0]
        ), f" Number of data in datasets are different. Shape of dataset1: {dataset1.shape[0]}, Shape of dataset2: {dataset2.shape[0]}"

        assert (
            dataset1.shape[1] == dataset2.shape[2]
        ), f"Shape of dataset1: {dataset1.shape}, Shape of dataset2: {dataset2.shape}"

        score = 0

        for i in range(len(dataset1)):
            score += self.calculate_avg_distance_between_point_and_embedding(
                dataset1[i].reshape(1, -1), dataset2[i], metrictype
            )
        return score / len(dataset1)

    def calculate_avg_distance_between_point_and_embedding(
        self,
        point: np.ndarray,
        embedding: np.ndarray,
        metrictype: MetricType = MetricType.cosine,
    ) -> float:
        """Calculate average distance between point and embedding.
        Args:
            point (np.ndarray): Point
            embedding (np.ndarray): Embedding
            metrictype (MetricType): Metric type
        Returns:
            np.ndarray: Average distance between point and embedding
        """
        assert len(point.shape) == 2, f"Point shape: {point.shape}"
        assert len(embedding.shape) == 2, f"Embedding shape: {embedding.shape}"
        assert point.shape[1] == embedding.shape[1], f"Point shape: {point.shape}, Embedding shape: {embedding.shape}"

        # Calculate average distance between point and embedding

        if metrictype == MetricType.cosine:
            distance = cosine_distances(point, embedding)
        elif metrictype == MetricType.l1_norm:
            distance = manhattan_distances(point, embedding)
        elif metrictype == MetricType.l2_norm:
            distance = euclidean_distances(point, embedding)
        else:
            raise NotImplementedError(f"Metric type {metrictype} not implemented")

        return np.mean(distance)

    def calculate_fid_score_between_datasets(self, query_dataset: np.ndarray, embedding_dataset: np.ndarray) -> float:
        """Calculate FID score between query and dataset.
        Args:
            query_dataset (np.ndarray): Query dataset
            embedding_dataset (np.ndarray): Embedding dataset
        Returns:
            float: FID score between query and dataset
        """

        assert len(query_dataset.shape) <= 2, f"Query shape: {query_dataset.shape}"
        assert len(embedding_dataset.shape) <= 2, f"Embedding dataset shape: {embedding_dataset.shape}"

        if len(query_dataset.shape) == 1:
            query_dataset = np.expand_dims(query_dataset, axis=0)
        if len(embedding_dataset.shape) == 1:
            embedding_dataset = np.expand_dims(embedding_dataset, axis=0)

        assert (
            query_dataset.shape[1] == embedding_dataset.shape[1]
        ), f"Query shape: {query_dataset.shape}, Embedding dataset shape: {embedding_dataset.shape}"

        fid_score = fid_scores(query_dataset, embedding_dataset)

        return fid_score

    def calculate_dataset_metric(
        self,
        dataset1,
        dataset2,
        datasetmetrictype: DatasetMetricType,
        distancemetrictype: MetricType = MetricType.cosine,
    ) -> float:
        """Calculate dataset metric between two datasets
        Args:
            dataset1 (np.ndarray): First dataset
            dataset2 (np.ndarray): Second dataset
            datasetmetrictype (DatasetMetricType): Dataset metric type
            distancemetrictype (MetricType): Distance metric type
        Returns:
            float: Dataset metric between two datasets
        """

        if datasetmetrictype == DatasetMetricType.one_to_one:
            return self.one_to_one_score(dataset1, dataset2, distancemetrictype)
        elif datasetmetrictype == DatasetMetricType.one_to_many:
            return self.one_to_many_score(dataset1, dataset2, distancemetrictype)
        elif datasetmetrictype == DatasetMetricType.fid:
            return self.calculate_fid_score_between_datasets(dataset1, dataset2)
        else:
            raise NotImplementedError(f"Dataset metric type {datasetmetrictype} not implemented")
