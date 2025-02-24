from typing import Tuple, Optional, Dict, Any

import numpy as np

import scipy.stats as stats
from scipy.optimize import curve_fit, minimize


def approximate_overlap_curves(
    m1: float,
    s1: float,
    m2: float,
    s2: float,
    bounds: Tuple[int, int],
) -> np.ndarray:
    """Calculates the overlap between two Gaussian curves.

    Computes the overlap between two Gaussian probability density functions by finding
    the minimum value at each point and integrating.

    Args:
        m1 (float): Mean of first Gaussian
        s1 (float): Standard deviation of first Gaussian
        m2 (float): Mean of second Gaussian
        s2 (float): Standard deviation of second Gaussian
        bounds (Tuple[int, int]): Integration bounds (min, max)

    Returns:
        np.ndarray: Overlap ratio between the two curves
    """
    x = np.linspace(*bounds, 10000)
    p1 = stats.norm(m1, s1).pdf(x)
    p2 = stats.norm(m2, s2).pdf(x)

    p_min = np.min(np.vstack([p1, p2]), axis=0)

    return p_min.sum() / p1.sum()


def gaussian_curve(
    x: np.ndarray,
    x0: float,
    s0: float,
) -> np.ndarray:
    """Computes unit-scaled Gaussian curve.

    Args:
        x (np.ndarray): Input x values
        x0 (float): Mean of Gaussian
        s0 (float): Standard deviation of Gaussian

    Returns:
        (np.ndarray): Gaussian curve values
    """
    return np.exp(-((x - x0) ** 2) / (2 * s0**2))


def fit_gaussian_elution(
    t: np.ndarray,
    m: np.ndarray,
    mu_bounds: Optional[Tuple[float, float]] = None,
    sigma_bounds: Optional[Tuple[float, float]] = [0.1, 10.0],
    clip_width: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> np.ndarray:
    """Fits Gaussian shape to NMF peak elutions.

    Args:
        t (np.ndarray): RT time access
        m (np.ndarray): Input elution profile
        mu_bounds (Tuple[float, float]): Bounds for mean parameter
        sigma_bounds (Tuple[float, float]): Bounds for standard deviation
        clip_width (int): Width to clip profile edges

    Returns:
        popt (List[float]): [mean, std] parameters
        pcov (np.ndarray): Covariance matrix
        error (float): Fit error
    """
    m_scale = m.copy()

    if clip_width:
        m_scale = m_scale[clip_width : -1 * clip_width]
        t = t[clip_width : -1 * clip_width]

    if m_scale.shape[0] == 0 or m_scale.max() == 0:
        return [None, None], None, None
    m_scale = (m_scale - m_scale.min()) / m_scale.max()

    m0 = sum(t * m_scale) / sum(m_scale)
    # m0 = sum(m_scale[:, 0] * m_scale[:, 1]) / sum(m_scale[:, 1])
    s0 = np.sqrt(sum(m_scale * (t - m0) ** 2) / sum(m_scale))
    # s0 = np.sqrt(sum(m_scale[:, 1] * (m_scale[:, 0] - m0) ** 2) / sum(m_scale[:, 1]))

    bounds = [[np.NINF, np.NINF], [np.Inf, np.Inf]]
    if mu_bounds:
        bounds[0][0] = mu_bounds[0]
        bounds[1][0] = mu_bounds[1]
    if sigma_bounds:
        bounds[0][1] = sigma_bounds[0]
        bounds[1][1] = sigma_bounds[1]

    m0 = (
        m0
        if not mu_bounds or (m0 < mu_bounds[1] and m0 > mu_bounds[0])
        else mu_bounds[1]
    )
    s0 = (
        s0
        if not sigma_bounds or (s0 < sigma_bounds[1] and s0 > sigma_bounds[0])
        else sigma_bounds[1]
    )

    popt, pcov = curve_fit(
        gaussian_curve,
        t,
        m_scale,
        p0=[m0, s0],
        bounds=bounds,
    )

    e = np.absolute(stats.norm(*popt).pdf(t) - m_scale).mean()

    return popt, pcov, e


def least_squares_with_l1_bounds(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Performs L1-regularized least squares optimization with bounds.

    Minimizes the objective function:
    sum((X @ coef - y)^2) + alpha * sum(|coef|)
    subject to the given bounds constraints on coefficients.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        alpha (float, optional): L1 regularization strength. Defaults to 0.1
        bounds (Tuple[float, float], optional): (min, max) bounds for coefficients.
            Defaults to (0.0, 1.0)

    Returns:
        np.ndarray: Optimized coefficient vector
    """

    def objective(coef: np.ndarray) -> float:
        return np.sum((np.dot(X, coef) - y) ** 2) + alpha * np.sum(np.abs(coef))

    coef_init = np.zeros(X.shape[1])

    coef_bounds = [bounds for _ in coef_init]

    result = minimize(
        objective,
        coef_init,
        method="SLSQP",
        bounds=coef_bounds,
        options={"maxiter": 100},
    )

    return result.x
