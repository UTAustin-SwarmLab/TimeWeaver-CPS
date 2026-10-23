import numpy as np
from pyts.metrics import dtw
from scipy.spatial import distance
import warnings
from scipy import linalg


def cosine_distance_metric(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine distance between two vectors. The distance should be between 0 and 1.
    The distance is 0 is the vectors are same and 1 if the vectors are orthogonal.
    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
    Returns:
        float: Cosine distance between two vectors
    """
    assert (
        vec1.shape == vec2.shape
    ), f"Shape of vec1: {vec1.shape}, Shape of vec2: {vec2.shape}"
    assert len(vec1.shape) == 1, f"Shape of vec1: {vec1.shape}"

    cosine_distance = distance.cosine(vec1, vec2)
    return cosine_distance


def l1_norm_metric(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L1 norm between two vectors.
    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
    Returns:
        float: L1 norm between two vectors
    """
    assert (
        vec1.shape == vec2.shape
    ), f"Shape of vec1: {vec1.shape}, Shape of vec2: {vec2.shape}"

    l1_norm = distance.cityblock(vec1, vec2)
    return l1_norm


def l2_norm_metric(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L2 norm between two vectors.
    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
    Returns:
        float: L2 norm between two vectors
    """
    assert (
        vec1.shape == vec2.shape
    ), f"Shape of vec1: {vec1.shape}, Shape of vec2: {vec2.shape}"

    l2_norm = distance.euclidean(vec1, vec2)
    return l2_norm


def dtw_metric(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute DTW between two vectors.
    Args:
        vec1 (np.ndarray): First vector
        vec2 (np.ndarray): Second vector
    Returns:
        float: DTW between two vectors
    """
    assert (
        vec1.shape == vec2.shape
    ), f"Shape of vec1: {vec1.shape}, Shape of vec2: {vec2.shape}"

    dtw_score = dtw(vec1, vec2)
    return dtw_score


def fid_scores(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """
    Frechet Distance between two vectors
    :param x: vector 1
    :param y: vector 2
    :return: Frechet Inception Distance between x and y
    """

    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)

    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)

    mu_x = np.atleast_1d(mu_x)
    mu_y = np.atleast_1d(mu_y)

    sigma_x = np.atleast_2d(sigma_x)
    sigma_y = np.atleast_2d(sigma_y)

    assert (
        mu_x.shape == mu_y.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma_x.shape == sigma_y.shape
    ), "Training and test covariances have different dimensions"

    diff = mu_x - mu_y

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_x.dot(sigma_y), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma_x.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_x + offset).dot(sigma_y + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma_x) + np.trace(sigma_y) - 2 * tr_covmean
