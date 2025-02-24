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
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.decomposition import NMF

import scipy.sparse as sps
import scipy.ndimage as ndi

from .ms1feature import MS1Feature

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
