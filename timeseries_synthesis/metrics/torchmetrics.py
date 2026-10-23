from torchmetrics import Metric
from torch import Tensor
from copy import deepcopy
import torch
from torchmetrics.functional.pairwise import (
    pairwise_manhattan_distance,
    pairwise_euclidean_distance,
    pairwise_cosine_similarity,
)
from enum import Enum

class MetricType(Enum):
    """Metric type"""

    cosine = "cosine"
    l1_norm = "l1_norm"
    l2_norm = "l2_norm"
    dtw = "dtw"
    fid = "fid"

def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class FID_Metric(Metric):
    """
    FID metric implementation for PyTorch Lightning.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    real_features_loaded: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(
        self,
        num_features: int,
        reset_real_features: bool = True,
    ):
        super().__init__()

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        mx_num_feets = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def update(self, features: Tensor, real: bool) -> None:
        """
        Updates the metric state.
        Args:
            features: Tensor containing features of generated images
            real: Boolean indicating if features are from real or generated images
        """

        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on stored features."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(
            0
        )
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(
            0
        )

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()


class One_to_Many_Metric(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0

    num_real_samples: Tensor
    num_fake_samples: Tensor
    total_distance: Tensor

    def __init__(self, metric_type: MetricType = MetricType.l1_norm):
        super().__init__()

        self.add_state("num_fake_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
        self.add_state(
            "total_distance", torch.tensor(0.0).double(), dist_reduce_fx="sum"
        )
        self.add_state("num_real_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
        self.metric_type = metric_type

    def update(self, real: Tensor, fake: Tensor) -> None:
        """
        Updates the metric state.
        Args:
            real: Tensor containing features of real images (N_samples, N_features)
            fake: Tensor containing features of generated images ( N_samples, N_sample_per_data, N_features)
        """

        assert (
            len(real.shape) == 2 and len(fake.shape) == 3
        ), "Real samples are not 2 dimensional or Generated Samples are "
        assert (
            real.shape[0] == fake.shape[0]
        ), "Number of real and fake samples are different"
        assert (
            real.shape[1] == fake.shape[2]
        ), "Number of features in real and fake samples are different"

        self.num_real_samples += real.shape[0]
        self.num_fake_samples += fake.shape[0] * fake.shape[1]

        for i in range(len(real)):
            self._update_per_sample(real[i].unsqueeze(0), fake[i])

    def _update_per_sample(self, real: Tensor, fake: Tensor) -> Tensor:
        """
        Calculate distance between real and fake samples.
            Args:
            real: Tensor containing features of real images (1, N_features)
            fake: Tensor containing features of generated images ( 1, N_sample_per_data, N_features)
            Return:
            Distance between real and fake samples averaged over fake samples
        """

        if self.metric_type == MetricType.l1_norm:
            self.total_distance += (
                pairwise_manhattan_distance(real, fake, reduction="sum").sum()
                / fake.shape[0]
            )
        elif self.metric_type == MetricType.cosine:
            self.total_distance += (
                1
                - pairwise_cosine_similarity(real, fake, reduction="sum").sum()
                / fake.shape[0]
            )
        elif self.metric_type == MetricType.l2_norm:
            self.total_distance += (
                pairwise_euclidean_distance(real, fake, reduction="sum").sum()
                / fake.shape[0]
            )
        else:
            raise ValueError("Metric type not supported")

    def compute(self) -> Tensor:
        """
        Calculate one to many score based on stored features.
        """

        assert self.num_fake_samples > 0, "Number of fake samples is zero"
        assert self.num_real_samples > 0, "Number of real samples is zero"

        return self.total_distance / self.num_real_samples
