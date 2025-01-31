from typing import Optional, Tuple, Dict, List

import numpy as np
from tqdm import tqdm 
import pandas as pd

import scipy.sparse as sps

from config import Config, Constants


def ms2_df_to_long(
    df: pd.DataFrame, 
    config: Config=Config,
) -> np.ndarray:
    """ """

    mz_arrays = []

    for mass in tqdm(df["MS2TargetMass"].unique(), disable=(not config.tqdm_enabled)):
        sub_df = df.loc[df["MS2TargetMass"] == mass]
        sub_df.set_index(["RetentionTime", "ScanNum"], inplace=True)
        mzs = []

        for (retention_time, scan_number), r in sub_df.loc[:, ["m/zArray", "IntensityArray", "lowerMass", "higherMass"]].iterrows():
            mz_array = r['m/zArray']
            intensity_array = r['IntensityArray']
            if config.ms2_preprocessing.filter_spectra:
                percentile_intensity = np.percentile(r['IntensityArray'], config.ms2_preprocessing.filter_spectra_percentile)
                mz_array = mz_array[intensity_array > percentile_intensity]
                intensity_array = intensity_array[intensity_array > percentile_intensity]
            for mz, intensity in zip(mz_array, intensity_array):
                if config.ms2_preprocessing.include_gpf_bounds:
                    mzs.append([mass, scan_number, retention_time, mz, intensity, r['lowerMass'], r['higherMass']])
                else:
                    mzs.append([mass, scan_number, retention_time, mz, intensity])

        mz_arrays.append(np.vstack(mzs))

    return np.vstack(mz_arrays)


def pivot_unique_binned_mz_dense(
    m: np.ndarray,
    mz_bin_index: int = 5,
    scan_index: int = 1,
    intensity_index: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Efficiently pivots binned mass spectrometry data into a scan-mz-intensity matrix.
    
    Args:
        m (np.ndarray): Output array from overlapping_discretize_bins
        mz_index (int): Index into m to aggregate on, defaults to bin_start position
        
    Returns:
        agg_m (np.ndarray): Array of shape (n_scans, n_mz_bins, 2) containing [mz_bin, aggregated_intensity]
    """
    unique_scans, scans_pos = np.unique(m[:, scan_index], return_inverse=True)
    unique_mz, mz_pos = np.unique(m[:, mz_bin_index], return_inverse=True)

    agg_m = np.zeros((unique_scans.shape[0], unique_mz.shape[0], 2))
    agg_m[:, :, 0] = unique_mz

    agg_m[:, :, 1] = sps.coo_matrix(
        (m[:, intensity_index], (scans_pos, mz_pos)),
        shape=(unique_scans.shape[0], unique_mz.shape[0]),
    ).A

    return agg_m, unique_scans, unique_mz

def pivot_unique_binned_mz_sparse(
    m: np.ndarray,
    mz_bin_index: int = 5, 
    scan_index: int = 1, 
    intensity_index: int = 4,
) -> Tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """ """
    unique_scans, scans_pos = np.unique(m[:, scan_index], return_inverse=True)
    unique_mz, mz_pos = np.unique(m[:, mz_bin_index], return_inverse=True)

    # Array of scan_id, mz_bin_id
    agg_m = sps.coo_matrix(
        (m[:,intensity_index], (scans_pos, mz_pos)),
        shape=(unique_scans.shape[0], unique_mz.shape[0])
    ).tocsr()

    return agg_m, unique_scans, unique_mz

def calculate_hoyer_sparsity(m: np.ndarray) -> float:
    """ """
    if np.allclose(m, np.zeros(m.shape)):
        return 0.0

    sparseness_num = np.sqrt(m.size) - (m.sum() / np.sqrt(np.multiply(m, m).sum()))
    sparseness_den = np.sqrt(m.size) - 1

    return sparseness_num / sparseness_den

def calculate_simple_sparsity(m: np.ndarray) -> float:
    """ """
    if np.allclose(m, np.zeros(m.shape)):
        return 0.0

    return (m > 0).sum() / m.size

def calculate_nmf_summary(W: np.ndarray, H: np.ndarray) -> Dict[str, float]:
    """ """
    non_zero_components = (~np.isclose(W.sum(axis=0), 0.0)) & (~np.isclose(H.sum(axis=1), 0.0))
    if non_zero_components.sum() >= 2:
        H_nonzero = H[non_zero_components,:]
        W_nonzero = W[:,non_zero_components]
        orthogonality_H = H_nonzero @ H_nonzero.T 
        orthogonality_W = W_nonzero.T @ W_nonzero
        weight_identity_matrix_approximation = orthogonality_H / np.linalg.norm(orthogonality_H, axis=1)[..., np.newaxis]
        sample_identity_matrix_approximation = orthogonality_W / np.linalg.norm(orthogonality_W, axis=0)[..., np.newaxis]
        weight_deviation_identity = np.linalg.norm(
            weight_identity_matrix_approximation - np.eye(H_nonzero.shape[0])
        )
        sample_deviation_identity = np.linalg.norm(
            sample_identity_matrix_approximation - np.eye(W_nonzero.shape[1])
        )
        # We usually want to maximize sparsity, so making these negative here because the sign gets 
        # automatically inverted in _fit_and_score. 
        weight_sparsity = -1.*np.apply_along_axis(calculate_hoyer_sparsity, axis=0, arr=W_nonzero).mean()
        sample_sparsity = -1.*np.apply_along_axis(calculate_hoyer_sparsity, axis=1, arr=H_nonzero).mean()
    else:
        weight_deviation_identity = 0.0
        sample_deviation_identity = 0.0
        weight_sparsity = 0.0
        sample_sparsity = 0.0

    return {
        'weight_sparsity': weight_sparsity,
        'sample_sparsity': sample_sparsity,
        'weight_deviation_identity': weight_deviation_identity,
        'sample_deviation_identity': sample_deviation_identity,
        'nonzero_component_fraction': (W.sum(axis=0) > 0).sum() / W.shape[1],
        'fraction_window_component': (W.sum(axis=1) > 0).sum() / W.shape[0],
    }


def calculate_theoretical_precursor_monoisotopic_masses(
    fragments: np.ndarray,
    charge_states: List[int] = np.arange(1, 4),
    isotope_mu: float = Constants.isotope_mu,
) -> np.ndarray:
    """ """
    monoisotopic_masses = []
    for c in charge_states:
        monoisotopic_masses.append(
            (fragments * c) - (c - 1) * isotope_mu
        )
    monoisotopic_masses = np.concatenate(monoisotopic_masses)

    theoretical_precursor_masses = (monoisotopic_masses[...,np.newaxis]+monoisotopic_masses).flatten()

    return theoretical_precursor_masses

def calculate_mz_from_masses(
    monoisotopic_masses: np.ndarray,
    charge_states: List[int] = np.arange(1, 9),
    isotope_mu: float = Constants.isotope_mu,
) -> np.ndarray:
    """ """
    mz = []
    for c in charge_states:
        mz.append(
            (monoisotopic_masses + ((c-1)*isotope_mu)) / c
        )
    return np.concatenate(mz)




