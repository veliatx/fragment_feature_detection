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
import h5py
from tqdm import tqdm

from scipy.spatial import distance  # Add this import
import scipy.sparse as sps

from .scanwindow import ScanWindow
from ..discretization import MzDiscretize
from ..config import Config
from ..utils import pivot_unique_binned_mz_sparse
from ..fitpeaks import approximate_overlap_curves

logger = logging.getLogger(__name__)


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
