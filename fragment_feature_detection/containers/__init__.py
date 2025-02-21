from typing import (
    Union,
    Optional,
    Tuple,
    Type,
    Literal,
    List,
    Dict,
    Any,
)
from pathlib import Path
import logging
import copy
import warnings
import time

import numpy as np
import h5py
from tqdm import tqdm
from sklearn.utils.parallel import Parallel, delayed
import scipy.stats as stats

from .ms1feature import MS1Feature
from .scanwindow import ScanWindow
from .gpfrun import GPFRun
from .msrun import MSRun
from ..config import Config
from ..utils import fraction_explained_variance
from ..fitpeaks import (
    fit_gaussian_elution,
    least_squares_with_l1_bounds,
)
from fragment_feature_detection.decomposition import fit_nmf_matrix_custom_init

logger = logging.getLogger(__name__)


def fit_scanwindow(
    w: ScanWindow,
    **fitpeaks_kwargs: Dict[str, Any],
) -> None:
    """Fit Gaussian peaks to components in a scan window.

    Args:
        w (ScanWindow): Scan window object to fit peaks to.
        **fitpeaks_kwargs: Keyword arguments passed to fit_gaussian_elution.
    """
    mus = []
    sigmas = []
    maes = []
    keep = []
    for i, s in enumerate(w.component_indices):
        try:
            (mu, s), pcov, mae = fit_gaussian_elution(
                w.retention_time,
                w.w[:, i],
                **fitpeaks_kwargs,
            )
            mus.append(mu)
            sigmas.append(s)
            maes.append(mae)
            keep.append(True)
        except:
            mus.append(0)
            sigmas.append(0)
            maes.append(0)
            keep.append(False)
    w.set_peak_fits(np.array(mus), np.array(sigmas), np.array(maes), np.array(keep))


def fit_nmf_matrix_scanwindow(
    w: ScanWindow,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    **nmf_kwargs: Dict[str, Any],
) -> None:
    """Fit NMF decomposition to a scan window's intensity matrix.

    Args:
        w (ScanWindow): Scan window object to fit NMF to.
        W_init (Optional[np.ndarray]): Initial guess for W matrix.
        H_init (Optional[np.ndarray]): Initial guess for H matrix.
        **nmf_kwargs: Keyword arguments passed to fit_nmf_matrix_custom_init.
    """
    W, H, model = fit_nmf_matrix_custom_init(
        w.m,
        W_init=W_init,
        H_init=H_init,
        return_model=True,
        **nmf_kwargs,
    )
    w.set_nmf_fit(W, H, model)


def fit_nmf_matrix_gpfrun(
    gpfrun: GPFRun,
    n_jobs: int = 8,
    config: Config = Config(),
) -> None:
    """Fit NMF decomposition to all scan windows in a GPF run in parallel.

    Args:
        gpfrun (GPFRun): GPF run object to fit.
        n_jobs (int): Number of parallel jobs to run.
        nmf_kwargs (Dict[str, Any]): Keyword arguments for NMF fitting.
        fitpeaks_kwargs (Dict[str, Any]): Keyword arguments for peak fitting.
    """

    def modify_fit_scanwindow(sw: ScanWindow) -> ScanWindow:
        """ """
        if not sw.is_filtered:
            sw.filter_scans()
        fit_nmf_matrix_scanwindow(sw, **config.nmf)
        fit_scanwindow(sw, **config.fitpeaks)
        return sw

    with Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs") as parallel:
        fit_scan_windows = parallel(
            delayed(modify_fit_scanwindow)(sw)
            for sw in tqdm(
                gpfrun.scan_windows,
                disable=(not config.tqdm_enabled),
                total=len(gpfrun.scan_windows),
            )
        )

    gpfrun.scan_windows = fit_scan_windows
    gpfrun.collapse_redundant_components()


def fit_nmf_matrix_msrun(
    msrun: MSRun,
    n_jobs: int = 8,
    config: Config = Config(),
) -> None:
    """ """
    start_time = time.time()
    logging.info("Begin fitting GPFRuns...")

    for gpf_index in tqdm(msrun._gpf_runs):
        gpf = msrun.get_gpf(gpf_index)
        logging.info(f"Starting to work on GPF index: {gpf_index} -> {gpf._gpf}.")
        fit_nmf_matrix_gpfrun(
            gpf,
            n_jobs=n_jobs,
            config=config,
        )
    logging.info(f"Finished fitting GPFRuns in {time.time() - start_time}.")


def fit_ms1_ms2_feature_matching_scanwindow(
    w: ScanWindow,
    **feature_matching_kwargs: Dict[str, Any],
) -> None:
    """Match MS1 features to MS2 components within a scan window.

    Args:
        w (ScanWindow): Scan window object containing MS1 and MS2 data
        **feature_matching_kwargs: Keyword arguments for feature matching including:
            alpha (float): L1 regularization parameter
            extend_w_fraction (float): Fraction to extend component width
            c_cutoff (float): Minimum coefficient threshold
    """
    alpha = feature_matching_kwargs.get("alpha")
    extend_w_fraction = feature_matching_kwargs.get("extend_w_fraction")
    c_cutoff = feature_matching_kwargs.get("c_cutoff")

    sw_fit_mean, sw_fit_sigma = w.component_fit_parameters
    ms1_elution_matrix, ms1_feature_name = w.ms1_features_information
    rt = w.retention_time

    ms1_ms2_feature_match = np.zeros(
        (
            w.component_indices.shape[0],
            ms1_feature_name.shape[0],
        )
    )
    global_ms2_explained_variance = np.zeros((w.component_indices.shape[0]))
    individual_ms2_explained_variance = np.zeros(
        (
            w.component_indices.shape[0],
            ms1_feature_name.shape[0],
        )
    )

    for i, s in enumerate(w.component_indices):
        # Pull 99% range of normal distribution centered around fit parameters.
        w_ci = stats.norm.interval(0.99, loc=sw_fit_mean[i], scale=sw_fit_sigma[i])
        # Extend this range by extend_w_fraction.
        w_low = w_ci[0] - (extend_w_fraction * (w_ci[1] - w_ci[0]))
        w_high = w_ci[1] + (extend_w_fraction * (w_ci[1] - w_ci[0]))
        idx_low = max(np.searchsorted(rt, w_low, side="left") - 1, 0)
        idx_high = min(np.searchsorted(rt, w_high, side="right"), rt.shape[0] - 1)

        sub_ms2 = w.w[idx_low:idx_high, i].flatten()
        sub_ms2 = sub_ms2 / np.linalg.norm(sub_ms2)
        sub_ms1_keep = ms1_elution_matrix[idx_low:idx_high]
        ms1_keep_mask = sub_ms1_keep.sum(axis=0) > 0
        sub_ms1_keep = sub_ms1_keep[:, ms1_keep_mask] / np.linalg.norm(
            sub_ms1_keep[:, ms1_keep_mask], axis=0
        )

        if sub_ms1_keep.size == 0:
            ms1_ms2_feature_match[i, :] = 0.0
            continue

        coef = least_squares_with_l1_bounds(sub_ms1_keep, sub_ms2, alpha=alpha)
        keep_coef = ~np.isclose(coef, 0.0) & (coef > c_cutoff)
        coef[~keep_coef] = 0.0

        ms1_ms2_feature_match[i, ms1_keep_mask] = coef

        global_ms2_explained_variance[i] = fraction_explained_variance(
            sub_ms1_keep, sub_ms2, coef
        )
        individual_ms2_explained_variance[i, keep_coef] = np.array(
            [
                fraction_explained_variance(sub_ms1_keep[c], sub_ms2, coef[c])
                for c in range(coef.shape[0])
            ]
        )

    w.set_ms1_ms2_feature_matches(
        ms1_ms2_feature_match,
        global_ms2_explained_variance,
        individual_ms2_explained_variance,
    )


def fit_ms1_ms2_feature_matching_gpfrun(
    gpfrun: GPFRun,
    n_jobs: int = 8,
    config: Config = Config(),
) -> None:
    """Match MS1 features to MS2 components across all scan windows in a GPF run.

    Args:
        gpfrun (GPFRun): GPF run object containing scan windows
        n_jobs (int): Number of parallel jobs to run
        config (Config): Configuration object containing feature matching parameters
    """

    def modify_match_scanwindow(sw: ScanWindow) -> ScanWindow:
        """ """
        fit_ms1_ms2_feature_matching_scanwindow(sw, **config.feature_matchinga)
        return sw

    with Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs") as parallel:
        fit_scan_windows = parallel(
            delayed(modify_match_scanwindow)(sw)
            for sw in tqdm(
                gpfrun.scan_windows,
                disable=(not config.tqdm_enabled),
                total=len(gpfrun.scan_windows),
            )
        )

    gpfrun.scan_windows = fit_scan_windows
