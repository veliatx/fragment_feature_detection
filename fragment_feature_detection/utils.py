from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm

from fragment_feature_detection.config import Config, Constants


def ms2_df_to_long(
    df: pd.DataFrame,
    config: Config = Config(),
) -> np.ndarray:
    """Converts MS2 DataFrame to a long-format numpy array.

    Args:
        df (pd.DataFrame): DataFrame containing MS2 data with columns for MS2TargetMass,
            RetentionTime, ScanNum, m/zArray, and IntensityArray
        config (Config): Configuration object. Defaults to Config()

    Returns:
        np.ndarray: Array with columns [mass, scan_number, retention_time, mz, intensity]
            or [mass, scan_number, retention_time, mz, intensity, lower_mass, higher_mass]
            if config.ms2_preprocessing.include_gpf_bounds is True
    """

    mz_arrays = []

    for mass in tqdm(df["MS2TargetMass"].unique(), disable=(not config.tqdm_enabled)):
        sub_df = df.loc[df["MS2TargetMass"] == mass]
        sub_df.set_index(["RetentionTime", "ScanNum"], inplace=True)
        mzs = []

        for (retention_time, scan_number), r in sub_df.loc[
            :, ["m/zArray", "IntensityArray", "lowerMass", "higherMass"]
        ].iterrows():
            mz_array = r["m/zArray"]
            intensity_array = r["IntensityArray"]
            if config.ms2_preprocessing.filter_spectra:
                percentile_intensity = np.percentile(
                    r["IntensityArray"],
                    config.ms2_preprocessing.filter_spectra_percentile,
                )
                mz_array = mz_array[intensity_array > percentile_intensity]
                intensity_array = intensity_array[
                    intensity_array > percentile_intensity
                ]
            for mz, intensity in zip(mz_array, intensity_array):
                if config.ms2_preprocessing.include_gpf_bounds:
                    mzs.append(
                        [
                            mass,
                            scan_number,
                            retention_time,
                            mz,
                            intensity,
                            r["lowerMass"],
                            r["higherMass"],
                        ]
                    )
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
    """Efficiently pivots binned mass spectrometry data into a sparse scan-mz-intensity matrix.

    Args:
        m (np.ndarray): Output array from overlapping_discretize_bins
        mz_bin_index (int): Column index for mz bin values. Defaults to 5
        scan_index (int): Column index for scan numbers. Defaults to 1
        intensity_index (int): Column index for intensity values. Defaults to 4

    Returns:
        Tuple containing:
            sps.csr_matrix: Sparse matrix of shape (n_scans, n_mz_bins) containing intensities
            np.ndarray: Array of unique scan numbers
            np.ndarray: Array of unique mz bin values
    """
    unique_scans, scans_pos = np.unique(m[:, scan_index], return_inverse=True)
    unique_mz, mz_pos = np.unique(m[:, mz_bin_index], return_inverse=True)

    # Array of scan_id, mz_bin_id
    agg_m = sps.coo_matrix(
        (m[:, intensity_index], (scans_pos, mz_pos)),
        shape=(unique_scans.shape[0], unique_mz.shape[0]),
    ).tocsr()

    return agg_m, unique_scans, unique_mz


def calculate_hoyer_sparsity(m: np.ndarray) -> float:
    """Calculates the Hoyer sparsity measure of an array.

    The Hoyer sparsity measure is based on the relationship between L1 and L2 norms.
    Returns a value between 0 (dense) and 1 (sparse).

    Args:
        m (np.ndarray): Input array

    Returns:
        float: Hoyer sparsity measure
    """
    if np.allclose(m, np.zeros(m.shape)):
        return 0.0

    sparseness_num = np.sqrt(m.size) - (m.sum() / np.sqrt(np.multiply(m, m).sum()))
    sparseness_den = np.sqrt(m.size) - 1

    return sparseness_num / sparseness_den


def calculate_simple_sparsity(m: np.ndarray) -> float:
    """Calculates the simple sparsity measure of an array.

    Simple sparsity is defined as the fraction of non-zero elements.
    Returns a value between 0 (dense) and 1 (sparse).

    Args:
        m (np.ndarray): Input array

    Returns:
        float: Simple sparsity measure
    """
    if np.allclose(m, np.zeros(m.shape)):
        return 0.0

    return (m > 0).sum() / m.size


def calculate_nmf_summary(W: np.ndarray, H: np.ndarray) -> Dict[str, float]:
    """Calculates summary statistics for Non-negative Matrix Factorization results.

    Args:
        W (np.ndarray): Weight matrix from NMF
        H (np.ndarray): Feature matrix from NMF

    Returns:
        Dict[str, float]: Dictionary containing:
            - weight_sparsity: Average Hoyer sparsity of weight matrix columns
            - sample_sparsity: Average Hoyer sparsity of feature matrix rows
            - weight_deviation_identity: Deviation from identity matrix for weights
            - sample_deviation_identity: Deviation from identity matrix for features
            - nonzero_component_fraction: Fraction of components with non-zero weights
            - fraction_window_component: Fraction of windows with non-zero components
    """
    non_zero_components = (~np.isclose(W.sum(axis=0), 0.0)) & (
        ~np.isclose(H.sum(axis=1), 0.0)
    )
    if non_zero_components.sum() >= 2:
        H_nonzero = H[non_zero_components, :]
        W_nonzero = W[:, non_zero_components]
        orthogonality_H = H_nonzero @ H_nonzero.T
        orthogonality_W = W_nonzero.T @ W_nonzero
        weight_identity_matrix_approximation = (
            orthogonality_H / np.linalg.norm(orthogonality_H, axis=1)[..., np.newaxis]
        )
        sample_identity_matrix_approximation = (
            orthogonality_W / np.linalg.norm(orthogonality_W, axis=0)[..., np.newaxis]
        )
        weight_deviation_identity = np.linalg.norm(
            weight_identity_matrix_approximation - np.eye(H_nonzero.shape[0])
        )
        sample_deviation_identity = np.linalg.norm(
            sample_identity_matrix_approximation - np.eye(W_nonzero.shape[1])
        )
        # We usually want to maximize sparsity, so making these negative here because the sign gets
        # automatically inverted in _fit_and_score.
        weight_sparsity = (
            -1.0
            * np.apply_along_axis(
                calculate_hoyer_sparsity, axis=0, arr=W_nonzero
            ).mean()
        )
        sample_sparsity = (
            -1.0
            * np.apply_along_axis(
                calculate_hoyer_sparsity, axis=1, arr=H_nonzero
            ).mean()
        )
    else:
        weight_deviation_identity = 0.0
        sample_deviation_identity = 0.0
        weight_sparsity = 0.0
        sample_sparsity = 0.0

    return {
        "weight_sparsity": weight_sparsity,
        "sample_sparsity": sample_sparsity,
        "weight_deviation_identity": weight_deviation_identity,
        "sample_deviation_identity": sample_deviation_identity,
        "nonzero_component_fraction": (W.sum(axis=0) > 0).sum() / W.shape[1],
        "fraction_window_component": (W.sum(axis=1) > 0).sum() / W.shape[0],
    }


def fraction_explained_variance(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
) -> float:
    """Calculates the fraction of variance explained by a linear model.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        coef (np.ndarray): Model coefficients

    Returns:
        float: R-squared value representing fraction of explained variance
    """
    return 1 - (np.sum((y - np.dot(X, coef)) ** 2) / np.sum((y - y.mean()) ** 2))


def calculate_theoretical_precursor_monoisotopic_masses(
    fragments: np.ndarray,
    charge_states: List[int] = np.arange(1, 4),
    proton_mu: float = Constants.proton_mu,
) -> np.ndarray:
    """Calculates theoretical precursor monoisotopic masses from fragment masses.

    Args:
        fragments (np.ndarray): Array of fragment masses
        charge_states (List[int]): List of charge states to consider.
            Defaults to [1, 2, 3]
        proton_mu (float): Mass of proton.
            Defaults to Constants.proton_mu

    Returns:
        np.ndarray: Array of theoretical precursor monoisotopic masses
    """
    monoisotopic_masses = []
    for c in charge_states:
        monoisotopic_masses.append((fragments * c) - (c - 1) * proton_mu)
    monoisotopic_masses = np.concatenate(monoisotopic_masses)

    theoretical_precursor_masses = (
        monoisotopic_masses[..., np.newaxis] + monoisotopic_masses
    ).flatten()

    return theoretical_precursor_masses


def calculate_mz_from_masses(
    monoisotopic_masses: np.ndarray,
    charge_states: List[int] = list(np.arange(1, 9)),
    proton_mu: float = Constants.proton_mu,
) -> np.ndarray:
    """Calculates m/z values from monoisotopic masses for different charge states.

    Args:
        monoisotopic_masses (np.ndarray): Array of monoisotopic masses
        charge_states (List[int]): List of charge states to consider.
            Defaults to [1, 2, ..., 8]
        proton_mu (float): Mass of proton.
            Defaults to Constants.proton_mu

    Returns:
        np.ndarray: Array of m/z values for all combinations of masses and charge states
    """
    mz = []
    for c in charge_states:
        mz.append((monoisotopic_masses + ((c - 1) * proton_mu)) / c)
    return np.concatenate(mz)


def feature_df_to_ms2(
    feature_df: pd.DataFrame,
    ms2_path: Union[str, Path],
    extractor_options: Dict[str, Any],
    extractor_version: str = version("fragment_feature_detection"),
    comments: Optional[str] = None,
    proton_mu: float = Constants.proton_mu,
) -> None:
    """Converts a feature DataFrame to MS2 file format.

    Args:
        feature_df (pd.DataFrame): DataFrame containing MS2 feature data
        ms2_path (Union[str, Path]): Path to output MS2 file
        extractor_options (Dict[str, Any]): Options used for feature extraction. Must include:
            output_type (str): Type of output data to use:
                - "raw": Uses raw MS2 data (ms2_mz_array and ms2_intensity_array)
                - "intensity_transform_weight": Uses reweighted component data
                  (ms2_component_reweight_mz and ms2_component_reweight_intensity)
                - "raw_weight": Uses component weights data
                  (ms2_component_weight_mz and ms2_component_weight_intensity)
            wide_window (bool): If True, uses GPF-based precursor information.
                              If False, uses MS1-based precursor information.
        extractor_version (str): Version of the extractor. Defaults to current package version
        comments (Optional[str]): Comments to include in MS2 header. Defaults to None
        proton_mu (float): Proton mass. Defaults to Constants.proton_mu
    """
    header = (
        """H\tCreationDate\t{creation_date}\nH\tExtractor\tffd\nH\tExtractorVersion\t{extractor_version}"""
        """\nH\tComments\t{comments}\nH\tExtractorOptions\t{extractor_options}\n\n"""
    ).format(
        creation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        extractor_version=extractor_version,
        comments=comments if comments else "Nothing to note.",
        extractor_options=",".join([f"{k}->{v}" for k, v in extractor_options.items()]),
    )

    # Configure which columns will be used for writing ms2.
    if extractor_options["output_type"] == "raw":
        mz_column_name = "ms2_mz_array"
        intensity_column_name = "ms2_intensity_array"
    elif extractor_options["output_type"] == "intensity_transform_weight":
        mz_column_name = "ms2_component_reweight_mz"
        intensity_column_name = "ms2_component_reweight_intensity"
    elif extractor_options["output_type"] == "raw_weight":
        mz_column_name = "ms2_component_weight_mz"
        intensity_column_name = "ms2_component_weight_intensity"

    if not extractor_options["wide_window"]:
        precursor_charge_column_name = "ms1_charge"
        precursor_mz_column_name = "ms1_mz"
        precursor_mass_column_name = "ms1_mass"
        df = feature_df.loc[~feature_df["ms1_matching_coef"].isna()].copy()
        # Biosaur2 reports precursor mass, convert to MH+ for ms2 file.
        df[precursor_mass_column_name] += proton_mu
    else:
        precursor_charge_column_name = "ms2_gpf_charge"
        precursor_mz_column_name = "ms2_gpf_center"
        precursor_mass_column_name = "ms2_gpf_mass"
        df = feature_df.drop_duplicates("ms2_component").copy()
        # Default to charge state 2
        df[precursor_charge_column_name] = 2
        df[precursor_mass_column_name] = (
            df[precursor_mz_column_name] * df[precursor_charge_column_name]
            - df[precursor_charge_column_name] * proton_mu
            + proton_mu
        )
    df["ms2_file_scan_number"] = np.arange(1, df.shape[0] + 1)

    ms2_scan = (
        """S\t{scan_number}\t{scan_number}\t{precursor_mz:.5f}\n"""
        """Z\t{precursor_charge}\t{precursor_mhplus:.5f}\n{scan_data}\n"""
    )

    with open(ms2_path, "w") as f_out:
        f_out.write(header)
        for i, r in df.iterrows():
            scan_data = "\n".join(
                f"{mz:.5f} {intensity:.3f}"
                for mz, intensity in zip(r[mz_column_name], r[intensity_column_name])
            )
            f_out.write(
                ms2_scan.format(
                    scan_number=int(r["ms2_file_scan_number"]),
                    precursor_mz=r[precursor_mz_column_name],
                    precursor_charge=int(r[precursor_charge_column_name]),
                    precursor_mhplus=r[precursor_mass_column_name],
                    scan_data=scan_data,
                )
            )
