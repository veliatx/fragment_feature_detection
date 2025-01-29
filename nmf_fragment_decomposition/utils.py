from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm 
import pandas as pd

import scipy.sparse as sps

from config import Config


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
                median_intensity = np.median(r['IntensityArray'])
                mz_array = mz_array[intensity_array > median_intensity]
                intensity_array = intensity_array[intensity_array > median_intensity]
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





