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

import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd

from sklearn.utils.parallel import Parallel, delayed
from sklearn.decomposition import NMF

import scipy.ndimage as ndi
import scipy.sparse as sps
import scipy.stats as stats
from scipy.spatial import distance

from fragment_feature_detection.discretization import MzDiscretize
from fragment_feature_detection.utils import (
    pivot_unique_binned_mz_sparse,
    fraction_explained_variance,
)
from fragment_feature_detection.fitpeaks import (
    approximate_overlap_curves,
    fit_gaussian_elution,
    least_squares_with_l1_bounds,
)
from fragment_feature_detection.decomposition import fit_nmf_matrix_custom_init
from fragment_feature_detection.config import Config

logger = logging.getLogger(__name__)


class ScanWindow:
    """A class representing a window of mass spectrometry scans for a specific GPF (Gas Phase Fraction) value.

    This class handles the processing and filtering of mass spectrometry data within a specific scan window,
    including operations like denoising, scaling, and filtering of m/z values.

    Attributes:
        _scan_number (np.ndarray): Array of scan numbers.
        _retention_time (np.ndarray): Array of retention times for each scan.
        _gpf (float): Gas Phase Fractionation value.
        _mz_masked (np.ndarray): Boolean array indicating filtered m/z windows.
        _mz_unfilter (np.ndarray): Unfiltered m/z values.
        _mz (np.ndarray): Filtered m/z values.
        _mz_bin_indices_unfilter (np.ndarray): Unfiltered bin indices.
        _mz_bin_indices (np.ndarray): Filtered bin indices.
        _m_unfilter (scipy.sparse.csr_matrix): Original sparse intensity matrix.
        _m (np.ndarray): Dense filtered intensity matrix.
        _m_max (np.ndarray): Maximum intensity in each m/z window.
        _denoise_scans (bool): Flag to enable scan denoising.
        _scale_scans (bool): Flag to enable scan scaling.
        _ms1_features (List[MS1Feature]): List of MS1 features associated with this window.
        _component_ms1_coef_matrix (np.ndarray): Matrix of coefficients relating MS1 to MS2 features.
        _component_ms1_global_exp_var (np.ndarray): Global explained variance for MS1-MS2 matches.
        _component_ms1_individual_exp_var (np.ndarray): Individual explained variance for each MS1-MS2 match.
        _is_ms1_features_fit (bool): Whether MS1 features have been fit to MS2 components.
    """

    _filter_edge_nscans = 5
    _denoise_scans = True
    _scale_scans = True
    _percentile_filter_scans = True
    _percentile_filter = 10
    _log_scans = False
    _filter_edge_scans = True
    _is_filtered = False
    _downcast_h5 = True
    _downcast_scans = True
    _downcast_bins = True

    _w = None
    _h = None
    _nmf_reconstruction_error = None
    _non_zero_components = None
    _component_keep = None
    _is_fit = False

    _component_means = None
    _component_sigmas = None
    _component_maes = None
    _component_names = None
    _is_component_fit = False

    _ms1_features = None
    _component_ms1_coef_matrix = None
    _component_ms1_global_exp_var = None
    _component_ms1_individual_exp_var = None
    _is_ms1_features_fit = False

    def __init__(
        self,
        gpf: float,
        m: sps.csr_matrix,
        scan_number: np.ndarray,
        scan_index: np.ndarray,
        bin_indices: np.ndarray,
        retention_time: np.ndarray,
        mz: np.ndarray,
        gpf_low: Optional[float] = None,
        gpf_high: Optional[float] = None,
    ):
        """Initialize a ScanWindow instance.

        Args:
            gpf (float): Gas Phase Fractionation value.
            m (scipy.sparse.csr_matrix): Sparse intensity matrix.
            scan_number (np.ndarray): Array of scan numbers.
            bin_indices (np.ndarray): Array of bin indices.
            retention_time (np.ndarray): Array of retention times.
            mz (np.ndarray): Array of m/z values.
        """
        self._scan_number = scan_number
        self._scan_index = scan_index
        self._retention_time = retention_time
        self._gpf = gpf
        self._gpf_low = gpf_low
        self._gpf_high = gpf_high
        self._mz_masked = np.zeros_like(bin_indices, dtype=bool)
        self._mz_unfilter = mz
        self._mz = mz.copy()
        self._mz_bin_indices_unfilter = bin_indices
        self._mz_bin_indices = bin_indices.copy()
        self._m_unfilter = m
        self._m = m.copy()
        self._m_max = None
        self._m_max_bin_indices = None

    def __copy__(self):
        """ """
        cls = self.__class__
        new_instance = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(new_instance, k, copy.deepcopy(v))
        return new_instance

    @property
    def m(self) -> np.ndarray:
        """ """
        if not isinstance(self._m, np.ndarray):
            return self._m.toarray()
        return self._m

    @property
    def retention_time(self) -> np.ndarray:
        """ """
        return self._retention_time

    @property
    def scan_number(self) -> np.ndarray:
        """ """
        return self._scan_number

    @property
    def scan_index(self) -> np.ndarray:
        """ """
        return self._scan_index

    @property
    def mz_bin_indices(self) -> np.ndarray:
        """ """
        return self._mz_bin_indices

    @property
    def w(self) -> np.ndarray:
        """ """
        if not getattr(self, "_is_fit", False):
            raise AttributeError("ScanWindow not fit.")
        return self._w[:, self.component_indices]

    @property
    def h(self) -> np.ndarray:
        """ """
        if not getattr(self, "_is_fit", False):
            raise AttributeError("ScanWindow not fit.")
        return self._h[self.component_indices, :]

    @property
    def component_indices(self) -> np.ndarray:
        """ """
        if not getattr(self, "_is_fit", False):
            raise AttributeError("ScanWindow not fit.")
        return np.where(self._non_zero_components & self._component_keep)[0]

    @property
    def component_maes(self) -> np.ndarray:
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError("Need to fit gaussians.")
        return self._component_maes[self.component_indices]

    @property
    def component_fit_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError("Need to fit gaussians.")
        return (
            self._component_means[self.component_indices],
            self._component_sigmas[self.component_indices],
        )

    @property
    def component_names(self) -> np.ndarray:
        """ """
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError("Need to fit gaussians.")
        return self._component_names[self.component_indices]

    @property
    def is_filtered(self) -> bool:
        """ """
        return self._is_filtered

    @property
    def is_fit(self) -> bool:
        """ """
        return self._is_fit

    @property
    def is_component_fit(self) -> bool:
        """ """
        return self._is_component_fit

    @property
    def ms1_features(self) -> List["MS1Feature"]:
        """ """
        return self._ms1_features

    @property
    def ms1_features_information(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if not getattr(self, "_ms1_features", False):
            raise AttributeError("No ms1 features added to this scanwindow.")

        ms1_elution = np.vstack(
            [f._interpolated_intensity for f in self._ms1_features]
        ).T

        ms1_feature_name = np.array([f._id for f in self._ms1_features])

        return (ms1_elution, ms1_feature_name)

    @property
    def is_ms1_features_fit(self) -> bool:
        """ """
        return self._is_ms1_features_fit

    def mask_component(self, component_index) -> None:
        """ """
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError
        self._component_keep[component_index] = False

    def filter_scans(self) -> None:
        """ """
        self.filter_zero_mz()
        self.filter_clip_constant()
        if self._percentile_filter_scans:
            self.filter_percentile_scans()
        if self._log_scans:
            self.transform_log_scans()
        if self._scale_scans:
            self.transform_maxscale_scans()
        if self._denoise_scans:
            self.filter_denoise_scans()
        if self._filter_edge_scans:
            self.filter_edge_scans()
        if self._downcast_scans:
            # Push through downcast to make results reproducible if ran through h5
            # intermediate or not.
            if self._downcast_h5:
                self._m = self._m.astype(np.float16)
            self._m = self._m.astype(np.float32)
        if self._downcast_bins:
            # These are pretty safe options for most scenarios. Need to revisit if for whatever reason
            # int32 insufficient for bin_indices or float32 insufficient for mz.
            self._mz = self._mz.astype(np.float32)
            self._mz_unfilter = self._mz_unfilter.astype(np.float32)
            self._mz_bin_indices = self._mz_bin_indices.astype(np.int32)
            self._mz_bin_indices_unfilter = self._mz_bin_indices_unfilter.astype(
                np.int32
            )
        self._is_filtered = True

    def filter_mz_axis(self, mask: np.ndarray) -> None:
        """ """
        self._mz_masked[
            np.searchsorted(
                self._mz_bin_indices_unfilter,
                self._mz_bin_indices[~mask],
            )
        ] = True
        self._mz = self._mz[mask]
        self._mz_bin_indices = self._mz_bin_indices[mask]
        self._m = self._m[:, mask]

    def filter_edge_scans(self) -> None:
        """ """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._m[: self._filter_edge_nscans, :] = 0.0
            self._m[-1 * self._filter_edge_nscans :, :] = 0.0
        self.filter_zero_mz()

    def filter_zero_mz(self) -> None:
        """ """
        mask = (self._m.sum(axis=0) > 0.0).flatten()
        if isinstance(mask, np.matrix):
            mask = np.array(mask).flatten()
        self.filter_mz_axis(mask)

    def transform_maxscale_scans(
        self,
        axis: int = 0,
        robust_scaling: bool = False,
        robust_scaling_percentile: float = 99.5,
    ) -> None:
        """ """
        if not robust_scaling:
            m_m = self._m.max(axis=axis)
        else:
            m_m = np.percentile(
                self._m,
                robust_scaling_percentile,
                axis=axis,
                method="lower",
            )
        if not isinstance(m_m, np.ndarray):
            m_m = m_m.toarray().flatten()
        if axis == 1:
            m_m = m_m[..., np.newaxis]
        self._m_max = m_m
        self._m_max_bin_indices = self._mz_bin_indices.copy()
        if not isinstance(self._m, (np.ndarray, np.matrix)):
            self._m = self._m.multiply(1.0 / m_m)
        else:
            self._m = self._m / m_m

    def transform_log_scans(self) -> None:
        """ """
        self._m = np.log1p(self._m)

    def filter_percentile_scans(self) -> None:
        """ """
        if isinstance(self._m, np.ndarray):
            threshold_signal = np.percentile(
                self._m.flatten()[self._m.flatten() > 0],
                self._percentile_filter,
            )
        else:
            threshold_signal = np.percentile(self._m.data, self._percentile_filter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._m[self._m < threshold_signal] = 0.0
        self.filter_zero_mz()

    def filter_clip_constant(self, p: float = 0.3) -> None:
        """ """
        if isinstance(self._m, np.ndarray):
            mask = ~(((self._m / self._m.max(axis=0)).mean(axis=0)) > p)
        else:
            mask = np.array(
                ~(((self._m / self._m.max(axis=0).toarray()).mean(axis=0)) > p)
            ).flatten()
        self.filter_mz_axis(mask)

    def filter_denoise_scans(
        self,
        grey_erosion_size: int = 2,
        grey_closing_size: int = 2,
    ) -> np.ndarray:
        """ """
        is_sparse = False
        if not isinstance(self._m, (np.ndarray, np.matrix)):
            is_sparse = True
            self._m = self._m.toarray()
        self._m = ndi.grey_erosion(
            ndi.grey_closing(
                self._m,
                size=(grey_closing_size, 1),
            ),
            size=(grey_erosion_size, 1),
        )
        self.filter_zero_mz()
        if is_sparse:
            self._m = sps.csr_matrix(self._m)

    def reverse_transform_maxscale_scans(self) -> np.ndarray:
        """ """
        if not isinstance(self._m_max, np.ndarray):
            raise AttributeError

        scale_indices = np.searchsorted(
            self._m_max_bin_indices,
            self._mz_bin_indices,
        )

        return self._m_max[scale_indices]

    def unfiltered_with_filter(self, scale_if_exists: bool = True) -> np.ndarray:
        """ """
        m = self._mz_bin_indices_unfilter[:, ~self._mz_masked].toarray()
        if scale_if_exists and self._m_max:
            scale_indices = np.searchsorted(
                self._m_max_bin_indices,
                self._mz_bin_indices,
            )
            return m / self._m_max[scale_indices]
        return m

    def expand_array(self, array_name: str) -> np.ndarray:
        """ """
        pass

    def set_nmf_fit(self, w: np.ndarray, h: np.ndarray, model: NMF) -> None:
        """ """
        if self.is_fit:
            self._w = np.hstack([self._w, w.copy()])
            self._h = np.vstack([self._h, h.copy()])
            self._nmf_reconstruction_error = np.concatenate(
                [self._nmf_reconstruction_error, model._reconstruction_err_]
            )
            self._non_zero_components = np.concatenate(
                [self._non_zero_components, w.sum(axis=0) != 0.0]
            )
            component_keep = np.zeros(w.shape[1], dtype=bool)
            component_keep = component_keep[w.sum(axis=0) != 0.0]
            self._component_keep = np.concatenate(
                [self._component_keep, component_keep]
            )

        self._w = w.copy()
        self._h = h.copy()
        self._nmf_reconstruction_error = model.reconstruction_err_
        self._non_zero_components = self._w.sum(axis=0) != 0.0
        self._component_keep = np.zeros(self._w.shape[1], dtype=bool)
        self._component_keep[self._non_zero_components] = True
        self._is_fit = True

    def set_peak_fits(
        self, m: np.ndarray, s: np.ndarray, maes: np.ndarray, keep: np.ndarray
    ) -> None:
        """ """
        self._component_keep[self._non_zero_components] = keep.copy()
        self._component_means = np.zeros(self._w.shape[1])
        self._component_means[self._non_zero_components] = m.copy()
        self._component_sigmas = np.zeros(self._w.shape[1])
        self._component_sigmas[self._non_zero_components] = s.copy()
        self._component_maes = np.zeros(self._w.shape[1])
        self._component_maes[self._non_zero_components] = maes.copy()

        # Filter fits that are within 2s of clipped edge.
        if self._filter_edge_scans:
            mask = (
                self.retention_time[self._filter_edge_nscans]
                > (self._component_means - 2 * self._component_sigmas)
            ) | (
                self.retention_time[-1 * self._filter_edge_nscans]
                < (self._component_means + 2 * self._component_sigmas)
            )
            self._component_keep[mask] = False
        self._component_names = np.array(
            [
                (
                    f"{self._gpf}_{i}-"
                    f"{self._scan_index[np.argmin(np.absolute(self._retention_time-self._component_means[i]))]}_"
                    f"{self._component_means[i]}_"
                    f"{self._component_sigmas[i]}"
                )
                for i in range(self._w.shape[1])
            ]
        )
        self._is_component_fit = True

    def query_component_info(
        self, value: str, by: Literal["name"] = "name"
    ) -> List[Dict[str, Any]]:
        """ """
        matching_components = []

        if by == "name":
            idxes = np.where(self.component_names == value)[0]
            for idx in idxes:
                matching_components.append(
                    {
                        # TODO implement stuff here
                    }
                )

        return matching_components

    def set_ms1_ms2_feature_matches(
        self,
        coef_matrix: np.ndarray,
        global_variance_explained: np.ndarray,
        individual_variance_explained: np.ndarray,
    ) -> None:
        """ """
        self._component_ms1_coef_matrix = coef_matrix
        self._component_ms1_global_exp_var = global_variance_explained
        self._component_ms1_individual_exp_var = individual_variance_explained
        self._is_ms1_features_fit = True

    def dump_h5(
        self,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "scan_window",
        save_m_unfilter: bool = True,
        mode: Literal["write", "update"] = "write",
    ) -> None:
        """Save the ScanWindow object to an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group to store the object
            save_m_unfilter (bool): Whether to save unfiltered matrix data
            mode (Literal["write", "update"]): Whether to overwrite ("write") or update existing data
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, "r+" if mode == "update" else "w")
            if mode == "update" and dataset_name in f:
                grp = f[dataset_name]
            else:
                grp = f.create_group(dataset_name)
            close_file = True
        else:
            if mode == "update" and dataset_name in h5file:
                grp = h5file[dataset_name]
            else:
                grp = h5file.create_group(dataset_name)
            close_file = False

        def check_arrays_equal(a1: np.ndarray, a2: np.ndarray) -> bool:
            """ """
            if a1 is None or a2 is None:
                return a1 is a2
            return np.array_equal(a1, a2)

        def save_dataset(
            name: str, data: Union[np.ndarray, sps.csr_matrix, None]
        ) -> None:
            """ """
            if data is None:
                if name in grp:
                    del grp[name]
                return

            if isinstance(data, sps.csr_matrix):
                # For sparse matrices
                if mode == "update":
                    data_changed = (
                        f"{name}_data" not in grp
                        or not np.array_equal(
                            data.data, grp[f"{name}_data"][:].astype(np.float32)
                        )
                        or not np.array_equal(data.indices, grp[f"{name}_indices"][:])
                        or not np.array_equal(data.indptr, grp[f"{name}_indptr"][:])
                        or not np.array_equal(
                            data.shape, grp.attrs.get(f"{name}_shape", None)
                        )
                    )
                    if not data_changed:
                        return

                # Delete existing datasets if they exist
                for suffix in ["_data", "_indices", "_indptr"]:
                    if f"{name}{suffix}" in grp:
                        del grp[f"{name}{suffix}"]

                # Create new datasets
                grp.create_dataset(
                    f"{name}_data",
                    data=(
                        data.data
                        if not self._downcast_h5
                        else data.data.astype(np.float16)
                    ),
                )
                grp.create_dataset(f"{name}_indices", data=data.indices)
                grp.create_dataset(f"{name}_indptr", data=data.indptr)
                grp.attrs[f"{name}_shape"] = data.shape
            else:
                # For regular arrays
                if mode == "update":
                    if name in grp and check_arrays_equal(data, grp[name][:]):
                        return
                if name in grp:
                    del grp[name]
                grp.create_dataset(name, data=data)

        # Save/update scalar attributes
        scalar_attrs = (
            "gpf",
            "gpf_low",
            "gpf_high",
            "is_filtered",
            "is_fit",
            "is_component_fit",
        )

        for sa in scalar_attrs:
            v = getattr(self, f"_{sa}", -1)
            if mode == "write" or sa not in grp.attrs or grp.attrs[sa] != v:
                grp.attrs[sa] = v

        # Save arrays
        array_datasets = (
            "scan_number",
            "scan_index",
            "retention_time",
            "mz_masked",
            "mz_unfilter",
            "mz",
            "mz_bin_indices_unfilter",
            "mz_bin_indices",
        )

        for ds in array_datasets:
            if hasattr(self, f"_{ds}"):
                save_dataset(ds, getattr(self, f"_{ds}", None))

        # Save sparse intensity matrix, or if dense.
        if isinstance(self._m, (np.ndarray, np.matrix)):
            save_dataset("m", self._m)
            grp.attrs["m_is_sparse"] = False
        else:
            save_dataset("m", self._m)
            grp.attrs["m_is_sparse"] = True

        if save_m_unfilter:
            # Save unfiltered sparse matrix
            save_dataset("m_unfilter", self._m_unfilter)

        # Save NMF results if they exist
        if self._is_fit and hasattr(self, "_w"):
            nmf_arrays = (
                "w",
                "h",
                "component_keep",
                "non_zero_components",
            )
            for na in nmf_arrays:
                if hasattr(self, f"_{na}"):
                    save_dataset(na, getattr(self, f"_{na}"))
            if hasattr(self, "_nmf_reconstruction_error"):
                grp.attrs["nmf_reconstruction_error"] = self._nmf_reconstruction_error

        # Save component fits if they exist
        if self._is_component_fit:
            component_arrays = (
                "component_means",
                "component_sigmas",
                "component_maes",
            )
            for ca in component_arrays:
                if hasattr(self, f"_{ca}"):
                    save_dataset(ca, getattr(self, f"_{ca}"))
            if hasattr(self, "_component_names") and self._component_names is not None:
                save_dataset("component_names", self._component_names.astype("S"))

        # Need to check and update MS1 features here.
        # Save MS1 features if they exist
        if self._ms1_features is not None:
            ms1_grp = grp.create_group("ms1_features")
            for i, feature in enumerate(self._ms1_features):
                feature.dump_h5(ms1_grp, f"feature_{i}")

        # Save class-level configuration attributes
        grp.attrs["filter_edge_nscans"] = self._filter_edge_nscans
        grp.attrs["denoise_scans"] = self._denoise_scans
        grp.attrs["scale_scans"] = self._scale_scans
        grp.attrs["percentile_filter_scans"] = self._percentile_filter_scans
        grp.attrs["percentile_filter"] = self._percentile_filter
        grp.attrs["log_scans"] = self._log_scans
        grp.attrs["filter_edge_scans"] = self._filter_edge_scans

        # Save m_max data if it exists
        if self._m_max is not None:
            save_dataset("m_max", self._m_max)
            save_dataset("m_max_bin_indices", self._m_max_bin_indices)

        # Save component names if they exist
        if self._component_names is not None:
            save_dataset("component_names", self._component_names.astype("S"))

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "scan_window",
    ) -> "ScanWindow":
        """Load a ScanWindow object from an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group containing the object

        Returns:
            ScanWindow: Loaded object
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, "r")
            grp = f[dataset_name]
            close_file = True
        else:
            grp = h5file[dataset_name]
            close_file = False

        # Load sparse matrix
        if grp.attrs["m_is_sparse"]:
            _m = sps.csr_matrix(
                (
                    grp["m_data"][:].astype(np.float32),
                    grp["m_indices"][:],
                    grp["m_indptr"][:],
                ),
                shape=grp.attrs["m_shape"],
            )
        else:
            _m = grp["m"][:].astype(np.float32)

        if "m_unfilter_data" in grp:
            # Load sparse matrix
            m = sps.csr_matrix(
                (
                    grp["m_unfilter_data"][:],
                    grp["m_unfilter_indices"][:],
                    grp["m_unfilter_indptr"][:],
                ),
                shape=grp.attrs["m_unfilter_shape"],
            )
        else:
            m = _m.copy()

        # Create instance
        obj = cls(
            gpf=grp.attrs["gpf"],
            m=m,
            scan_number=grp["scan_number"][:],
            scan_index=grp["scan_index"][:],
            bin_indices=grp["mz_bin_indices"][:],
            retention_time=grp["retention_time"][:],
            mz=grp["mz"][:],
            gpf_low=None if grp.attrs["gpf_low"] == -1 else grp.attrs["gpf_low"],
            gpf_high=None if grp.attrs["gpf_high"] == -1 else grp.attrs["gpf_high"],
        )

        # Restore additional attributes
        obj._mz_masked = grp["mz_masked"][:]
        obj._mz_unfilter = grp["mz_unfilter"][:]
        obj._mz_bin_indices_unfilter = grp["mz_bin_indices_unfilter"][:]

        obj._m = _m
        obj._is_filtered = grp.attrs["is_filtered"]

        # Restore NMF results if they exist
        if grp.attrs["is_fit"]:
            obj._w = grp["w"][:]
            obj._h = grp["h"][:]
            obj._component_keep = grp["component_keep"][:]
            obj._non_zero_components = grp["non_zero_components"][:]
            obj._nmf_reconstruction_error = grp.attrs["nmf_reconstruction_error"]
            obj._is_fit = True

        # Restore component fits if they exist
        if grp.attrs["is_component_fit"]:
            obj._component_means = grp["component_means"][:]
            obj._component_sigmas = grp["component_sigmas"][:]
            obj._component_maes = grp["component_maes"][:]
            obj._is_component_fit = True

        # Load MS1 features if they exist
        if "ms1_features" in grp:
            obj._ms1_features = []
            ms1_grp = grp["ms1_features"]
            for i in range(len(ms1_grp)):
                feature = MS1Feature.from_h5(ms1_grp, f"feature_{i}")
                obj._ms1_features.append(feature)

        # Load class-level configuration attributes
        obj._filter_edge_nscans = grp.attrs["filter_edge_nscans"]
        obj._denoise_scans = grp.attrs["denoise_scans"]
        obj._scale_scans = grp.attrs["scale_scans"]
        obj._percentile_filter_scans = grp.attrs["percentile_filter_scans"]
        obj._percentile_filter = grp.attrs["percentile_filter"]
        obj._log_scans = grp.attrs["log_scans"]
        obj._filter_edge_scans = grp.attrs["filter_edge_scans"]

        # Load m_max data if it exists
        if "m_max" in grp:
            obj._m_max = grp["m_max"][:]
            obj._m_max_bin_indices = grp["m_max_bin_indices"][:]

        # Load component names if they exist
        if "component_names" in grp:
            obj._component_names = grp["component_names"][:].astype("U")

        if close_file:
            f.close()

        return obj

    def add_ms1_features(self, ms1_df: pd.DataFrame, isotope_mu: float = 1.008) -> None:
        """ """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sub_ms1_df = ms1_df.loc[
                (ms1_df["rtStart"] < self.retention_time[-1])
                & (ms1_df["rtEnd"] > self.retention_time[0])
                & (
                    ms1_df["rtApex"].map(
                        lambda x: self.retention_time[0] < x < self.retention_time[-1]
                    )
                )
            ]
            sub_ms1_df.loc[:, "mz_isotopes"] = sub_ms1_df.apply(
                lambda x: [
                    x.mz + (i * isotope_mu / x.charge)
                    for i in range(0, x.nIsotopes)
                    if x.mz + (i * isotope_mu / x.charge)
                    >= getattr(self, "_gpf_low", self._gpf - 1.0)
                    and x.mz + (i * isotope_mu / x.charge)
                    <= getattr(self, "_gpf_high", self._gpf + 1.0)
                ],
                axis=1,
            )
            sub_ms1_df = sub_ms1_df.loc[
                sub_ms1_df["mz_isotopes"].map(lambda x: len(x) > 0)
            ]

        self._ms1_features = []

        for i, r in sub_ms1_df.iterrows():
            ms1_feature = MS1Feature.from_biosaur2_series_with_ms2(
                r.copy(),
                self.retention_time.copy(),
            )
            self._ms1_features.append(ms1_feature)


class MS1Feature:

    _interpolated_retention_time = None
    _interpolated_intensity_list = None

    def __init__(
        self,
        df_index: int,
        calibrated_mass: float,
        retention_time_apex: float,
        intensity_apex: float,
        intensity_sum: float,
        charge: int,
        n_isotopes: int,
        n_scans: int,
        mz: float,
        rt_start: float,
        rt_end: float,
        scan_index: List[int],
        intensity: List[float],
        scan_apex: int,
    ) -> None:
        self._id = df_index
        self._calibrated_mass = calibrated_mass
        self._retention_time_apex = retention_time_apex
        self._intensity_apex = intensity_apex
        self._intensity_sum = intensity_sum
        self._charge = charge
        self._n_isotopes = n_isotopes
        self._n_scans = n_scans
        self._mz = mz
        self._rt_start = rt_start
        self._rt_end = rt_end
        self._scan_index = np.array(scan_index)
        self._intensity = np.array(intensity)
        self._scan_apex = scan_apex
        self._retention_time = np.linspace(
            rt_start,
            rt_end,
            len(scan_index),
            endpoint=True,
        )

    @classmethod
    def from_biosaur2_series(cls, series: pd.Series) -> "MS1Feature":
        """Create an MS1Feature from a Biosaur2 pandas Series.

        Args:
            series (pd.Series): Series containing Biosaur2 feature data

        Returns:
            MS1Feature: New instance initialized with Biosaur2 data
        """
        return cls(
            df_index=series.name,
            calibrated_mass=series["massCalib"],
            retention_time_apex=series["rtApex"],
            intensity_apex=series["intensityApex"],
            intensity_sum=series["intensitySum"],
            charge=series["charge"],
            n_isotopes=series["nIsotopes"],
            n_scans=series["nScans"],
            mz=series["mz"],
            rt_start=series["rtStart"],
            rt_end=series["rtEnd"],
            scan_index=series["mono_hills_scan_lists"],
            intensity=series["mono_hills_intensity_list"],
            scan_apex=series["scanApex"],
        )

    @classmethod
    def from_biosaur2_series_with_ms2(
        cls, series: pd.Series, ms2_rts: np.ndarray
    ) -> "MS1Feature":
        """ """
        obj = cls.from_biosaur2_series(series)
        obj.interpolate_onto_ms2(ms2_rts)

        return obj

    def interpolate_onto_ms2(self, ms2_rts: np.ndarray) -> None:
        """ """
        intensities = np.interp(
            ms2_rts,
            self._retention_time,
            [0.0] + self._intensity[1:-1].tolist() + [0.0],
        )

        self._interpolated_retention_time = ms2_rts
        self._interpolated_intensity = intensities

    def dump_h5(
        self,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "ms1_feature",
    ) -> None:
        """Save the MS1Feature object to an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group to store the object
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, "w")
            grp = f.create_group(dataset_name)
            close_file = True
        else:
            grp = h5file.create_group(dataset_name)
            close_file = False

        # Save scalar attributes
        grp.attrs["id"] = self._id
        grp.attrs["calibrated_mass"] = self._calibrated_mass
        grp.attrs["retention_time_apex"] = self._retention_time_apex
        grp.attrs["intensity_apex"] = self._intensity_apex
        grp.attrs["intensity_sum"] = self._intensity_sum
        grp.attrs["charge"] = self._charge
        grp.attrs["n_isotopes"] = self._n_isotopes
        grp.attrs["n_scans"] = self._n_scans
        grp.attrs["mz"] = self._mz
        grp.attrs["rt_start"] = self._rt_start
        grp.attrs["rt_end"] = self._rt_end
        grp.attrs["scan_apex"] = self._scan_apex

        # Save arrays
        grp.create_dataset("scan_index", data=self._scan_index)
        grp.create_dataset("intensity", data=self._intensity)
        grp.create_dataset("retention_time", data=self._retention_time)

        # Save interpolated data if it exists
        if self._interpolated_retention_time is not None:
            grp.create_dataset(
                "interpolated_retention_time", data=self._interpolated_retention_time
            )
            grp.create_dataset(
                "interpolated_intensity", data=self._interpolated_intensity
            )

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "ms1_feature",
    ) -> "MS1Feature":
        """Load an MS1Feature object from an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group containing the object

        Returns:
            MS1Feature: Loaded object
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, "r")
            grp = f[dataset_name]
            close_file = True
        else:
            grp = h5file[dataset_name]
            close_file = False

        # Create instance
        obj = cls(
            df_index=grp.attrs["id"],
            calibrated_mass=grp.attrs["calibrated_mass"],
            retention_time_apex=grp.attrs["retention_time_apex"],
            intensity_apex=grp.attrs["intensity_apex"],
            intensity_sum=grp.attrs["intensity_sum"],
            charge=grp.attrs["charge"],
            n_isotopes=grp.attrs["n_isotopes"],
            n_scans=grp.attrs["n_scans"],
            mz=grp.attrs["mz"],
            rt_start=grp.attrs["rt_start"],
            rt_end=grp.attrs["rt_end"],
            scan_index=grp["scan_index"][:],
            intensity=grp["intensity"][:],
            scan_apex=grp.attrs["scan_apex"],
        )

        # Load interpolated data if it exists
        if "interpolated_retention_time" in grp:
            obj._interpolated_retention_time = grp["interpolated_retention_time"][:]
            obj._interpolated_intensity = grp["interpolated_intensity"][:]

        if close_file:
            f.close()

        return obj


class GPFRun:
    """A class representing a Gas Phase Fractionation (GPF) run in mass spectrometry data.

    This class handles the processing and analysis of mass spectrometry data for a specific
    GPF value, including managing multiple scan windows and component detection.

    Attributes:
        _gpf_runs (Dict): Dictionary mapping GPF indices to GPFRun instances.
        _gpf (float): Gas Phase Fractionation value.
        _m (scipy.sparse.csr_matrix): Sparse intensity matrix.
        _scan_number (np.ndarray): Array of scan numbers.
        _bin_indices (np.ndarray): Array of bin indices.
        _retention_time (np.ndarray): Array of retention times.
        _discretize (MzDiscretize): Object for m/z discretization.
        _window_scan_overlap (int): Number of overlapping scans between windows.
        _window_scan_width (int): Width of scan windows in number of scans.
        _scan_index (np.ndarray): Array of scan indices.
        _gpf_low (Optional[float]): Lower bound of GPF window.
        _gpf_high (Optional[float]): Upper bound of GPF window.
        _scan_windows (List[ScanWindow]): List of scan window objects.
    """

    _msrun = None
    _scan_windows = None

    def __init__(
        self,
        gpf: float,
        m: sps.csr_matrix,
        scan_number: np.ndarray,
        bin_indices: np.ndarray,
        retention_time: np.ndarray,
        discretize: MzDiscretize,
        window_scan_overlap: int = 30,
        window_scan_width: int = 150,
        gpf_low: Optional[float] = None,
        gpf_high: Optional[float] = None,
        index: Optional[int] = None,
    ):
        self._gpf = gpf
        self._m = m
        self._scan_number = scan_number
        self._bin_indices = bin_indices
        self._retention_time = retention_time
        self._discretize = discretize
        self._window_scan_overlap = window_scan_overlap
        self._window_scan_width = window_scan_width
        self._scan_index = np.arange(0, scan_number.shape[0])
        self._gpf_low = gpf_low
        self._gpf_high = gpf_high
        self._index = index

        self.build_scan_windows()

    def slice_scan_window(self, start: int, end: int) -> ScanWindow:
        """ """
        return ScanWindow(
            self._gpf,
            self._m[start:end].copy(),
            self._scan_number[start:end].copy(),
            self._scan_index[start:end].copy(),
            self._bin_indices.copy(),
            self._retention_time[start:end].copy(),
            self._discretize.indices_to_mz(self._bin_indices),
            gpf_low=self._gpf_low,
            gpf_high=self._gpf_high,
        )

    @property
    def scan_windows(self) -> List[ScanWindow]:
        """ """
        return self._scan_windows

    @scan_windows.setter
    def scan_windows(self, modified_scan_windows: List[ScanWindow]) -> None:
        """ """
        if not all(isinstance(sw, ScanWindow) for sw in modified_scan_windows):
            raise ValueError("Modified scan windows not correct object.")
        if not len(modified_scan_windows) == len(self.scan_windows):
            raise ValueError("List of modified scan windows incorrect len.")
        self._scan_windows = modified_scan_windows
        if self._msrun is not None and self._msrun._lazy:
            for i, scan_window in enumerate(self._scan_windows):
                self.dump_scanwindow_to_h5(
                    i,
                    scan_window,
                    self._msrun._h5_file["gpf_runs"][f"gpf_{self._index}"],
                    mode="update",
                )

    def update_scan_window_at_index(
        self, scan_window_index: int, scan_window: ScanWindow
    ) -> None:
        """ """
        self._scan_windows[scan_window_index] = scan_window
        if self._msrun is not None and self._msrun._lazy:
            self.dump_scanwindow_to_h5(
                scan_window_index,
                scan_window,
                self._msrun._h5_file["gpf_runs"][f"gpf_{self._index}"],
                mode="update",
            )

    def build_scan_windows(
        self, assign: bool = True, indices: Optional[np.ndarray] = None
    ) -> Optional[List[ScanWindow]]:
        """ """
        if not isinstance(indices, np.ndarray):
            indices = np.arange(
                0,
                self._scan_number.shape[0] - self._window_scan_width,
                self._window_scan_overlap,
            )
        if assign:
            self._scan_windows = [
                self.slice_scan_window(i, i + self._window_scan_width) for i in indices
            ]
        else:
            return [
                self.slice_scan_window(i, i + self._window_scan_width) for i in indices
            ]

    def sample_scan_windows(
        self,
        window_sampling_fraction: float = 0.05,
        exclude_scan_window_edges: int = 3,
        rng: np.random.Generator = np.random.default_rng(seed=Config.random_seed),
    ) -> List[ScanWindow]:
        """ """
        indices = np.arange(
            exclude_scan_window_edges * self._window_scan_overlap,
            self._scan_number.shape[0]
            - exclude_scan_window_edges * self._window_scan_overlap,
            self._window_scan_overlap,
        )
        sample = np.where((rng.random(indices.shape[0]) < window_sampling_fraction))[0]

        return self.build_scan_windows(assign=False, indices=indices[sample])

    def get_overlapping_scanwindow_indices(
        self,
        start: float,
        end: float,
        units: Literal["retention_time", "scan_index", "scan_number"] = "scan_index",
        ignore_indexed: bool = True,
    ) -> List[int]:
        """ """
        if not hasattr(self._scan_windows[0], units):
            raise AttributeError(f"Units not an attribute on ScanWindow: {units}")
        if ignore_indexed:
            return [
                i
                for i, sw in enumerate(self._scan_windows)
                if getattr(sw, units)[0] < end
                and getattr(sw, units)[-1] > start
                and getattr(sw, units)[0] != start
            ]
        return [
            i
            for i, sw in enumerate(self._scan_windows)
            if getattr(sw, units)[0] < end and getattr(sw, units)[-1] > start
        ]

    def get_overlapping_components(
        self,
        start: float,
        end: float,
        units: Literal["retention_time", "scan_index", "scan_number"] = "scan_index",
    ) -> Tuple[np.ndarray]:
        """ """

        indices = self.get_overlapping_scanwindow_indices(
            start, end, units=units, ignore_indexed=False
        )

        scan_windows = [self._scan_windows[i] for i in indices]
        component_names = np.concatenate([sw.component_names for sw in scan_windows])
        all_retention_times = np.unique(
            [r for sw in scan_windows for r in sw.retention_time]
        )
        w_matrices = [w for sw in scan_windows for w in sw.w.T]
        scan_indices = [
            sw.scan_index for sw in scan_windows for _ in sw.component_names
        ]
        all_scan_indices = np.unique(
            [idx for sw in scan_windows for idx in sw.scan_index]
        )

        m = np.zeros((all_scan_indices.shape[0], component_names.shape[0]))

        for i in range(component_names.shape[0]):
            m[scan_indices[i] - all_scan_indices.min(), i] = w_matrices[i]

        return m, component_names, all_scan_indices, all_retention_times

    def collapse_redundant_components(
        self,
        overlap_threshold: float = 0.5,
        similarity_threshold: float = 0.75,
    ) -> None:
        """ """
        for sw in self._scan_windows:

            ol_scan_indices = self.get_overlapping_scanwindow_indices(
                sw.scan_index[0],
                sw.scan_index[-1],
                units="scan_index",
                ignore_indexed=True,
            )
            for oli in ol_scan_indices:
                ol_sw = self._scan_windows[oli]

                union_mz_bin_indices = np.unique(
                    np.concatenate([sw.mz_bin_indices, ol_sw.mz_bin_indices])
                )
                sw_mz_bin_indices = np.searchsorted(
                    union_mz_bin_indices, sw.mz_bin_indices
                )
                sw_component_weights = np.zeros_like(union_mz_bin_indices, dtype=float)
                ol_mz_bin_indices = np.searchsorted(
                    union_mz_bin_indices, ol_sw.mz_bin_indices
                )
                ol_component_weights = np.zeros_like(union_mz_bin_indices, dtype=float)

                sw_fit_mean, sw_fit_sigma = sw.component_fit_parameters

                for i, ni in enumerate(sw.component_indices):

                    sw_component_weights[:] = 0.0
                    sw_component_weights[sw_mz_bin_indices] = sw._h[ni]

                    ol_fit_mean, ol_fit_sigma = ol_sw.component_fit_parameters

                    for j, nj in enumerate(ol_sw.component_indices):

                        overlap_density = approximate_overlap_curves(
                            sw_fit_mean[i],
                            sw_fit_sigma[i],
                            ol_fit_mean[j],
                            ol_fit_sigma[j],
                            bounds=(
                                min(sw_fit_mean[i], ol_fit_mean[j])
                                - self._window_scan_overlap,
                                max(sw_fit_mean[i], ol_fit_mean[j])
                                + self._window_scan_overlap,
                            ),
                        )
                        if overlap_density <= overlap_threshold:
                            continue

                        ol_component_weights[:] = 0.0
                        ol_component_weights[ol_mz_bin_indices] = ol_sw._h[nj]

                        component_distance = distance.cosine(
                            sw_component_weights, ol_component_weights
                        )
                        if (overlap_density > overlap_threshold) and (
                            component_distance < 1 - similarity_threshold
                        ):
                            if sw._component_maes[ni] < ol_sw._component_maes[nj]:
                                ol_sw.mask_component(nj)
                            else:
                                sw.mask_component(ni)
                                break

    def query_component_by_name(
        self,
        component: str,
    ) -> List[Dict[str, Any]]:
        """ """
        for sw in self.scan_windows:
            if sw.is_component_fit and (sw.component_names == component).any():
                return sw.query_component_info(
                    component,
                    by="name",
                )
        return []

    def dump_h5(
        self,
        h5_fh: Union[str, h5py.File, h5py.Group],
        config: Config = Config(),
    ) -> None:
        """Save the GPFRun object to an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "w")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Save main attributes
        f.attrs["gpf"] = self._gpf
        f.attrs["gpf_low"] = self._gpf_low if self._gpf_low is not None else -1
        f.attrs["gpf_high"] = self._gpf_high if self._gpf_high is not None else -1
        f.attrs["window_scan_overlap"] = self._window_scan_overlap
        f.attrs["window_scan_width"] = self._window_scan_width
        f.attrs["index"] = self._index if self._index is not None else -1

        # Save arrays
        f.create_dataset("scan_number", data=self._scan_number)
        f.create_dataset("scan_index", data=self._scan_index)
        f.create_dataset("bin_indices", data=self._bin_indices)
        f.create_dataset("retention_time", data=self._retention_time)

        # Save sparse matrix
        f.create_dataset("m_data", data=self._m.data)
        f.create_dataset("m_indices", data=self._m.indices)
        f.create_dataset("m_indptr", data=self._m.indptr)
        f.attrs["m_shape"] = self._m.shape

        # Save discretize object
        discretize_grp = f.create_group("discretize")
        self._discretize.dump_h5(discretize_grp)

        # Save scan windows
        if self._scan_windows is not None:
            windows_grp = f.create_group("scan_windows")
            for i, window in enumerate(self._scan_windows):
                window.dump_h5(
                    windows_grp,
                    f"window_{i}",
                    save_m_unfilter=config.h5.scan_window_save_m_unfilter,
                )

        if close_file:
            f.close()

    def dump_scanwindow_to_h5(
        self,
        scanwindow_index: int,
        scan_window: ScanWindow,
        h5_fh: Union[str, h5py.File, h5py.Group],
        mode: Literal["write", "update"] = "update",
    ) -> None:
        """
        Update/append a new scanwindow run to an existing HDF5 file.

        Args:
            scanwindow_index (int): Index for the scan window
            scan_window (ScanWindow): Scan window to save
            h5_fh(Optional[Union[str, h5py.File]]):
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "a")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        scan_window.dump_h5(
            f["scan_windows"],
            f"window_{scanwindow_index}",
            mode=mode,
            save_m_unfilter=False,
        )

        if close_file:
            f.close()

    @classmethod
    def from_h5(cls, h5_fh: Union[str, h5py.File, h5py.Group]) -> "GPFRun":
        """Load a GPFRun object from an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object

        Returns:
            GPFRun: Loaded object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "r")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Load discretize object
        discretize = MzDiscretize.from_h5(f["discretize"])

        # Load sparse matrix
        m = sps.csr_matrix(
            (f["m_data"][:], f["m_indices"][:], f["m_indptr"][:]),
            shape=f.attrs["m_shape"],
        )

        # Create instance
        obj = cls(
            gpf=f.attrs["gpf"],
            m=m,
            scan_number=f["scan_number"][:],
            bin_indices=f["bin_indices"][:],
            retention_time=f["retention_time"][:],
            discretize=discretize,
            window_scan_overlap=f.attrs["window_scan_overlap"],
            window_scan_width=f.attrs["window_scan_width"],
            gpf_low=None if f.attrs["gpf_low"] == -1 else f.attrs["gpf_low"],
            gpf_high=None if f.attrs["gpf_high"] == -1 else f.attrs["gpf_high"],
            index=None if f.attrs["index"] == -1 else f.attrs["index"],
        )

        # Load scan windows if they exist
        if "scan_windows" in f:
            obj._scan_windows = []
            windows_grp = f["scan_windows"]
            for i in range(len(windows_grp)):
                window = ScanWindow.from_h5(windows_grp, f"window_{i}")
                obj._scan_windows.append(window)

        if close_file:
            f.close()

        return obj

    @classmethod
    def from_undiscretized_long(
        cls: Type["GPFRun"],
        gpf: float,
        m: np.ndarray,
        discretize: MzDiscretize,
        index: Optional[int] = None,
        config: Config = Config(),
    ) -> "GPFRun":
        """ """
        sub_m = m[np.isclose(m[:, 0], gpf)]
        discretized_m = discretize.discretize_mz_array(sub_m)

        return cls.from_discretized_long(
            gpf,
            discretized_m,
            discretize,
            config=config,
            index=index,
        )

    @classmethod
    def from_discretized_long(
        cls: Type["GPFRun"],
        gpf: float,
        m: np.ndarray,
        discretize: MzDiscretize,
        index: Optional[int] = None,
        config: Config = Config(),
    ) -> "GPFRun":
        """ """
        sub_m = m[np.isclose(m[:, 0], gpf)]
        mz_bin_index = 5
        gpf_low = None
        gpf_high = None
        if config.ms2_preprocessing.include_gpf_bounds:
            mz_bin_index += 2
            gpf_low = sub_m[:, 5][0]
            gpf_high = sub_m[:, 6][0]
        sub_m_p, scans, bin_indices = pivot_unique_binned_mz_sparse(
            sub_m, mz_bin_index=mz_bin_index
        )

        if config.downcast_intensities:
            sub_m_p = sub_m_p.astype(np.float32)

        retention_times = np.unique(sub_m[:, [1, 2]], axis=0)[:, 1]

        return cls(
            gpf,
            sub_m_p,
            scans,
            bin_indices,
            retention_times,
            discretize,
            gpf_low=gpf_low,
            gpf_high=gpf_high,
            window_scan_overlap=config.scan_filter.scan_overlap,
            window_scan_width=config.scan_filter.scan_width,
            index=index,
        )


class MSRun:
    """A class representing a complete mass spectrometry run with multiple GPF windows.

    This class manages multiple GPF runs and provides methods for analyzing and processing
    the entire MS dataset.

    Attributes:
        _gpf_runs (Dict): Dictionary mapping GPF indices to GPFRun instances.
        _isolation_low (Optional[float]): Lower bound of isolation window.
        _isolation_high (Optional[float]): Upper bound of isolation window.
    """

    _gpf_runs = None
    _h5_file = None

    def __init__(
        self,
        isolation_low: Optional[float] = None,
        isolation_high: Optional[float] = None,
        h5_fh: Optional[Union[str, Path]] = None,
        lazy: bool = False,
    ):
        self._isolation_low = isolation_low
        self._isolation_high = isolation_high
        self._lazy = lazy
        self._gpf_runs = {}

        if lazy and h5_fh is None:
            raise ValueError("If lazy, must provide h5_fh.")
        if h5_fh:
            self._h5_file = h5py.File(h5_fh, "a")
            # Instantiate all of the appropriate groups for writing to h5.
            self.setup_h5(self._h5_file)

    def __del__(self) -> None:
        """Close h5 file if open handle exists on instance."""
        if self._h5_file is not None:
            self._h5_file.close()

    def close(self) -> None:
        """ """
        if self._h5_file:
            self._h5_file.close()

    def get_gpf(self, gpf_index: int) -> GPFRun:
        """Get or retrieve GPFrun from disk.

        Args:
            gpf_index (int): Index of GPF run to retrieve

        Returns:
            GPFRun: The requested GPF run

        Raises:
            KeyError: If GPF index doesn't exist
        """
        if gpf_index not in self._gpf_runs:
            raise KeyError(f"GPF index {gpf_index} not present in MSRun.")

        if isinstance(self._gpf_runs[gpf_index], str):
            gpf = GPFRun.from_h5(self._h5_file[self._gpf_runs[gpf_index]])
            gpf._msrun = self
            return gpf

        return self._gpf_runs[gpf_index]

    def add_gpf(
        self,
        gpf_index: int,
        gpf: Union[GPFRun, None],
        remove_if_exists: bool = False,
        config: Config = Config(),
    ) -> None:
        """ """
        # Maintain a relationship between msrun-gpf objects.
        if gpf is not None:
            gpf._msrun = self

        if config.h5.scan_window_filter_during_load and gpf is not None:
            for sw in gpf.scan_windows:
                sw.filter_scans()
        if not self._lazy:
            self._gpf_runs[gpf_index] = gpf
        else:
            group_name = f"gpf_runs/gpf_{gpf_index}"
            if group_name in self._h5_file and remove_if_exists:
                del self._h5_file[group_name]
            if group_name not in self._h5_file:
                self.dump_gpfrun_to_h5(gpf_index, gpf, self._h5_file)
            self._gpf_runs[gpf_index] = group_name

    def get_tuning_windows(self, config: Config = Config()) -> None:
        """ """
        rng = np.random.default_rng(seed=config.random_seed)

        windows = []

        for gpf_index in tqdm(self._gpf_runs, disable=(not config.tqdm_enabled)):
            gpf = self.get_gpf(gpf_index)
            scan_windows = gpf.sample_scan_windows(
                window_sampling_fraction=config.tuning.window_sampling_fraction,
                exclude_scan_window_edges=config.tuning.exclude_scan_window_edges,
                rng=rng,
            )

            if not all(sw.is_filtered for sw in scan_windows):
                for sw in scan_windows:
                    sw.filter_scans()

            windows.extend(
                [
                    sw.m.copy()
                    for sw in scan_windows
                    if sw.m.shape[1] > 0
                    and sw.m.shape[0] == config.scan_filter.scan_width
                ]
            )

        return windows

    def query_components_by_name(
        self,
        components: List[str],
    ) -> None:
        """ """
        matching_components = []

        if not self._lazy:
            for gpf_index in self._gpf_runs:
                gpf = self.get_gpf(gpf_index)
                for c in components:
                    matching_components.extend(gpf.query_component_by_name(c))
        else:
            for gpf_index in self._gpf_runs:
                gpf_path = self._gpf_runs[gpf_index]
                gpf_group = self._h5_file[gpf_path]

                if "scan_windows" not in gpf_group:
                    continue

                for sw_index in gpf_group["scan_windows"]:
                    sw_group = gpf_group["scan_windows"][sw_index]

                    if "component_names" not in sw_group:
                        continue

                    component_names = sw_group["component_names"][:].astype("U")

                    for c in components:
                        idx = np.where(c == component_names)[0]

                        if len(idx) > 0:
                            sw = ScanWindow.from_h5(sw_group)
                            matching_components.extend(
                                sw.query_component_info(c, by="name")
                            )

        return matching_components

    @classmethod
    def from_h5_long(
        cls: Type["MSRun"],
        h5_fh: Union[Path, str],
        gpf_masses: Optional[List[str]] = None,
        discretize: Optional[MzDiscretize] = None,
        msrun_h5_fh: Optional[Union[Path, str]] = None,
        config: Config = Config(),
    ) -> "MSRun":
        """ """

        if discretize is None:
            discretize = MzDiscretize.from_config(config)

        msrun = cls(
            lazy=config.lazy,
            h5_fh=msrun_h5_fh if msrun_h5_fh else Path(h5_fh).with_suffix(".msrun.h5"),
        )

        with h5py.File(h5_fh, "r") as f:
            unique_gpfs = [k for k in f.keys() if "ms2_long" in f[k].keys()]
            if gpf_masses is not None:
                unique_gpfs = [k for k in unique_gpfs if k in gpf_masses]

            unique_gpfs = sorted(unique_gpfs, key=lambda x: float(x))

            for i, gpf_mass in tqdm(
                enumerate(unique_gpfs),
                disable=(not config.tqdm_enabled),
                total=len(unique_gpfs),
            ):
                sub_m = f[gpf_mass]["ms2_long"][:]

                gpf = GPFRun.from_undiscretized_long(
                    float(gpf_mass),
                    sub_m,
                    discretize,
                    config=config,
                    index=i,
                )
                msrun.add_gpf(i, gpf)

        return msrun

    @classmethod
    def from_long(
        cls: Type["MSRun"],
        m: np.ndarray,
        discretize: Optional[MzDiscretize] = None,
        msrun_h5_fh: Optional[Union[Path, str]] = None,
        config: Config = Config(),
    ) -> "MSRun":
        """ """

        if discretize is None:
            discretize = MzDiscretize.from_config(config)

        msrun = cls(lazy=config.lazy, h5_fh=msrun_h5_fh)

        unique_gpfs = np.unique(m[:, 0])

        for i, gpf_mass in tqdm(
            enumerate(unique_gpfs),
            disable=(not config.tqdm_enabled),
            total=unique_gpfs.size,
        ):

            sub_m = m[np.isclose(m[:, 0], gpf_mass)]

            gpf = GPFRun.from_undiscretized_long(
                gpf_mass,
                sub_m,
                discretize,
                config=config,
                index=i,
            )
            msrun.add_gpf(i, gpf)

        return msrun

    def dump_gpfrun_to_h5(
        self,
        gpf_index: int,
        gpf_run: GPFRun,
        h5_fh: Union[str, h5py.File],
    ) -> None:
        """
        Append a new GPF run to an existing HDF5 file.

        Args:
            gpf_index (int): Index for the GPF run
            gpf_run (GPFRun): GPF run to save
            h5_fh(Optional[Union[str, h5py.File]]):

        Raises:
            ValueError: If GPF index already exists
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "a")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        if f"gpf_runs/gpf_{gpf_index}" in f:
            raise ValueError(f"GPF index {gpf_index} already exists in HDF5 file")

        gpf_run.dump_h5(f["gpf_runs"].create_group(f"gpf_{gpf_index}"))

        if close_file:
            f.close()

    def setup_h5(self, h5_file: h5py.File) -> None:
        """Setup h5 if not instantiated previously.

        Args:
            h5_file (h5py.File): h5py object
        """
        # Save attributes if they don't exist yet.
        if "isolation_low" not in h5_file:
            h5_file.attrs["isolation_low"] = (
                self._isolation_low if self._isolation_low is not None else -1
            )
        if "isolation_high" not in h5_file:
            h5_file.attrs["isolation_high"] = (
                self._isolation_low if self._isolation_low is not None else -1
            )
        if "gpf_runs" not in h5_file:
            h5_file.create_group("gpf_runs")

    def dump_h5(self, h5_fh: Union[str, h5py.File]) -> None:
        """Save the MSRun object to an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "w")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Setup the h5 file.
        self.setup_h5(f)
        gpf_runs_grp = f["gpf_runs"]

        for gpf_index, gpf_run in self._gpf_runs.items():
            gpf_run.dump_h5(gpf_runs_grp.create_group(f"gpf_{gpf_index}"))

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls, h5_fh: Union[str, h5py.File, h5py.Group], config: Config = Config()
    ) -> "MSRun":
        """Load an MSRun object from an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object

        Returns:
            MSRun: Loaded object
        """
        if isinstance(h5_fh, str):
            # Handle case when you want to lazy-load from_h5, use handle from object itself.
            if config.lazy:
                f = None
                close_file = False
            # If lazy-loading not desired, can create handle.
            else:
                f = h5py.File(h5_fh, "r")
                close_file = True
        else:
            if config.lazy:
                raise ValueError(
                    "Can't lazy-load h5 and provide h5 file or group as h5_fh."
                )
            f = h5_fh
            close_file = False

        # Create instance
        obj = cls(
            lazy=config.lazy,
            h5_fh=h5_fh if not isinstance(h5_fh, (h5py.File, h5py.Group)) else None,
        )

        # For loading, either pass the handle from the new instance or the handle above.
        if obj._lazy:
            f = obj._h5_file

        obj._isolation_low = (
            None if f.attrs["isolation_low"] == -1 else f.attrs["isolation_low"]
        )
        obj._isolation_high = (
            None if f.attrs["isolation_high"] == -1 else f.attrs["isolation_high"]
        )

        gpf_runs_grp = f["gpf_runs"]
        for gpf_name in tqdm(gpf_runs_grp, disable=(not config.tqdm_enabled)):
            gpf_index = int(gpf_name.split("_")[1])
            if not obj._lazy:
                gpf_run = GPFRun.from_h5(gpf_runs_grp[gpf_name])
            else:
                gpf_run = None
            obj.add_gpf(gpf_index, gpf_run)

        if close_file:
            f.close()

        return obj


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
