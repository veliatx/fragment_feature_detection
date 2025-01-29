from typing import (
    Union, Optional, Tuple, Type, Literal, List, Dict, Any,
)
from pathlib import Path
import logging
import copy
import warnings

import numpy as np
from sklearn.decomposition import NMF
from tqdm import tqdm 
import h5py

import scipy.ndimage as ndi 
import scipy.sparse as sps 
from scipy.spatial import distance

from discretization import MzDiscretize
from utils import pivot_unique_binned_mz_sparse
from fitpeaks import approximate_overlap_curves, fit_gaussian_elution
from decomposition import fit_nmf_matrix_custom_init
from config import Config

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
    """
    _filter_edge_nscans = 5
    _denoise_scans = True
    _scale_scans = True
    _percentile_filter_scans = True
    _percentile_filter = 10
    _log_scans = False
    _filter_edge_scans = True
    _is_filtered = False

    _w = None 
    _h = None
    _nmf_reconstruction_error = None
    _non_zero_components = None
    _component_keep = None
    _is_fit = False

    _component_means = None 
    _component_sigmas = None 
    _component_maes = None
    _is_component_fit = False

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
        # self._m = m.toarray()
        self._m = m.copy()
        self._m_max= None
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
        if not getattr(self, '_is_fit', False):
            raise AttributeError('ScanWindow not fit.')
        return self._w[:,self.component_indices]

    @property 
    def h(self) -> np.ndarray:
        """ """
        if not getattr(self, '_is_fit', False):
            raise AttributeError('ScanWindow not fit.')
        return self._h[self.component_indices,:]
    
    @property 
    def component_indices(self) -> np.ndarray:
        """ """
        if not getattr(self, '_is_fit', False):
            raise AttributeError('ScanWindow not fit.')
        return np.where(self._non_zero_components & self._component_keep)[0]

    @property 
    def component_maes(self) -> np.ndarray:
        if not getattr(self, '_is_component_fit', False):
            raise AttributeError('Need to fit gaussians.')
        return self._component_maes[self.component_indices]

    @property 
    def component_fit_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if not getattr(self, '_is_component_fit', False):
            raise AttributeError('Need to fit gaussians.')
        return (
            self._component_means[self.component_indices], 
            self._component_sigmas[self.component_indices],
        )

    @property 
    def is_filtered(self) -> bool:
        """ """
        return self._is_filtered 
    
    @property
    def is_fit(self) -> bool:
        """ """
        return self._is_fit

    def mask_component(self, component_index) -> None:
        """ """
        if not getattr(self, '_is_component_fit', False):
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
        self._m = self._m[:,mask]

    def filter_edge_scans(self) -> None:
        """ """
        self._m[:self._filter_edge_nscans,:] = 0.0 
        self._m[-1*self._filter_edge_nscans:,:] = 0.0
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
                method='lower',
            )
        if not isinstance(m_m, np.ndarray):
            m_m = m_m.toarray().flatten()
        if axis == 1:
            m_m = m_m[..., np.newaxis]
        self._m_max = m_m
        self._m_max_bin_indices = self._mz_bin_indices.copy()
        if isinstance(self._m, sps.csr_matrix):
            self._m = self._m.multiply(1 / m_m)
        else:
            self._m = self._m / m_m

    def transform_log_scans(self) -> None:
        """ """
        self._m = np.log1p(self._m).todense()

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
            warnings.simplefilter('ignore')
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
        if isinstance(self._m, sps.csr_matrix):
            self._m = self._m.toarray()
        self._m = ndi.grey_erosion(
            ndi.grey_closing(
                self._m,
                size=(grey_closing_size, 1),
            ),
            size=(grey_erosion_size, 1),
        )
        self.filter_zero_mz()
    
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
        m = self._mz_bin_indices_unfilter[:,~self._mz_masked].toarray()
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
        self._w = w.copy()
        self._h = h.copy()
        self._nmf_reconstruction_error = model.reconstruction_err_
        self._non_zero_components = self._w.sum(axis=0) != 0.0
        self._component_keep = np.zeros(self._w.shape[1], dtype=bool)
        self._component_keep[self._non_zero_components] = True
        self._is_fit = True

    def set_peak_fits(self, m: np.ndarray, s: np.ndarray, maes: np.ndarray) -> None:
        """ """
        self._component_means = np.zeros(self._w.shape[1])
        self._component_means[self._non_zero_components] = m.copy()
        self._component_sigmas = np.zeros(self._w.shape[1])
        self._component_sigmas[self._non_zero_components] = s.copy()
        self._component_maes = np.zeros(self._w.shape[1])
        self._component_maes[self._non_zero_components] = maes.copy()

        # Filter fits that are within 2s of clipped edge. 
        if self._filter_edge_scans:
            mask = (
                self.retention_time[self._filter_edge_nscans] > (self._component_means - 2*self._component_sigmas)
            ) | (
                self.retention_time[-1*self._filter_edge_nscans] < (self._component_means + 2*self._component_sigmas)
            )
            self._component_keep[mask] = False
        self._is_component_fit = True

    def dump_h5(
        self, 
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = 'scan_window',
    ) -> None:
        """Save the ScanWindow object to an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group to store the object
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, 'w')
            grp = f.create_group(dataset_name)
            close_file = True
        else:
            grp = h5file.create_group(dataset_name)
            close_file = False

        # Save arrays
        grp.create_dataset('scan_number', data=self._scan_number)
        grp.create_dataset('scan_index', data=self._scan_index)
        grp.create_dataset('retention_time', data=self._retention_time)
        grp.create_dataset('mz_masked', data=self._mz_masked)
        grp.create_dataset('mz_unfilter', data=self._mz_unfilter)
        grp.create_dataset('mz', data=self._mz)
        grp.create_dataset('mz_bin_indices_unfilter', data=self._mz_bin_indices_unfilter)
        grp.create_dataset('mz_bin_indices', data=self._mz_bin_indices)

        # Save sparse matrix
        if isinstance(self._m, (np.ndarray, np.matrix)):
            grp.create_dataset('m', data=self._m)
            grp.attrs['m_is_sparse'] = False
        else:
            grp.create_dataset('m_data', data=self._m.data)
            grp.create_dataset('m_indices', data=self._m.indices)
            grp.create_dataset('m_indptr', data=self._m.indptr)
            grp.attrs['m_shape'] = self._m.shape
            grp.attrs['m_is_sparse'] = True

        # Save unfiltered sparse matrix
        grp.create_dataset('m_unfilter_data', data=self._m_unfilter.data)
        grp.create_dataset('m_unfilter_indices', data=self._m_unfilter.indices)
        grp.create_dataset('m_unfilter_indptr', data=self._m_unfilter.indptr)
        grp.attrs['m_unfilter_shape'] = self._m_unfilter.shape

        # Save scalar attributes
        grp.attrs['gpf'] = self._gpf
        grp.attrs['gpf_low'] = self._gpf_low if self._gpf_low is not None else -1
        grp.attrs['gpf_high'] = self._gpf_high if self._gpf_high is not None else -1
        grp.attrs['is_filtered'] = self._is_filtered
        grp.attrs['is_fit'] = self._is_fit
        grp.attrs['is_component_fit'] = self._is_component_fit

        # Save NMF results if they exist
        if self._is_fit:
            grp.create_dataset('w', data=self._w)
            grp.create_dataset('h', data=self._h)
            grp.create_dataset('component_keep', data=self._component_keep)
            grp.create_dataset('non_zero_components', data=self._non_zero_components)
            grp.attrs['nmf_reconstruction_error'] = self._nmf_reconstruction_error

        # Save component fits if they exist
        if self._is_component_fit:
            grp.create_dataset('component_means', data=self._component_means)
            grp.create_dataset('component_sigmas', data=self._component_sigmas)
            grp.create_dataset('component_maes', data=self._component_maes)

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls, 
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = 'scan_window',
    ) -> 'ScanWindow':
        """Load a ScanWindow object from an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group containing the object

        Returns:
            ScanWindow: Loaded object
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, 'r')
            grp = f[dataset_name]
            close_file = True
        else:
            grp = h5file[dataset_name]
            close_file = False

        # Load sparse matrix
        if grp.attrs['m_is_sparse']:
            m = sps.csr_matrix(
                (grp['m_data'][:], grp['m_indices'][:], grp['m_indptr'][:]),
                shape=grp.attrs['m_shape']
            )
        else:
            m = grp['m'][:]

        # Create instance
        obj = cls(
            gpf=grp.attrs['gpf'],
            m=m,
            scan_number=grp['scan_number'][:],
            scan_index=grp['scan_index'][:],
            bin_indices=grp['mz_bin_indices'][:],
            retention_time=grp['retention_time'][:],
            mz=grp['mz'][:],
            gpf_low=None if grp.attrs['gpf_low'] == -1 else grp.attrs['gpf_low'],
            gpf_high=None if grp.attrs['gpf_high'] == -1 else grp.attrs['gpf_high'],
        )

        # Restore additional attributes
        obj._mz_masked = grp['mz_masked'][:]
        obj._mz_unfilter = grp['mz_unfilter'][:]
        obj._mz_bin_indices_unfilter = grp['mz_bin_indices_unfilter'][:]
        obj._m_unfilter = sps.csr_matrix(
            (grp['m_unfilter_data'][:], grp['m_unfilter_indices'][:], grp['m_unfilter_indptr'][:]),
            shape=grp.attrs['m_unfilter_shape']
        )
        obj._is_filtered = grp.attrs['is_filtered']

        # Restore NMF results if they exist
        if grp.attrs['is_fit']:
            obj._w = grp['w'][:]
            obj._h = grp['h'][:]
            obj._component_keep = grp['component_keep'][:]
            obj._non_zero_components = grp['non_zero_components'][:]
            obj._nmf_reconstruction_error = grp.attrs['nmf_reconstruction_error']
            obj._is_fit = True

        # Restore component fits if they exist
        if grp.attrs['is_component_fit']:
            obj._component_means = grp['component_means'][:]
            obj._component_sigmas = grp['component_sigmas'][:]
            obj._component_maes = grp['component_maes'][:]
            obj._is_component_fit = True

        if close_file:
            f.close()

        return obj

class GPFRun:

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

        self.build_scan_windows()

    def slice_scan_window(self, start: int, end: int) -> ScanWindow:
        """ """        
        return ScanWindow(
            self._gpf,
            self._m[start: end].copy(),
            self._scan_number[start: end].copy(),
            self._scan_index[start: end].copy(),
            self._bin_indices.copy(),
            self._retention_time[start: end].copy(),
            self._discretize.indices_to_mz(self._bin_indices),
            gpf_low=self._gpf_low,
            gpf_high=self._gpf_high,
        )

    def build_scan_windows(self, assign: bool = True, indices: Optional[np.ndarray] = None) -> Optional[List[ScanWindow]]:
        """ """
        if not isinstance(indices, np.ndarray):
            indices = np.arange(0, self._scan_number.shape[0] - self._window_scan_overlap, self._window_scan_overlap)
        if assign:
            self._scan_windows = [
                self.slice_scan_window(i, i+self._window_scan_width) for i in indices
            ]
        else:
            return [self.slice_scan_window(i, i+self._window_scan_width) for i in indices]
    
    def sample_scan_windows(
        self, 
        window_sampling_fraction: float = 0.05,
        exclude_scan_window_edges: int = 3, 
        rng: np.random.Generator = np.random.default_rng(seed=Config.random_seed),
    ) -> List[ScanWindow]:
        """ """
        indices = np.arange(
            exclude_scan_window_edges * self._window_scan_overlap, 
            self._scan_number.shape[0] - exclude_scan_window_edges * self._window_scan_overlap, 
            self._window_scan_overlap,
        )
        sample = np.where((rng.random(indices.shape[0]) < window_sampling_fraction))[0]

        return self.build_scan_windows(assign=False, indices=indices[sample])

    def get_overlapping_scanwindow_indices(
        self,
        start: float, 
        end: float, 
        units: Literal['retention_time', 'scan_index', 'scan_number'] = 'scan_index',
        ignore_indexed: bool = True, 
    ) -> List[int]:
        """ """
        if not hasattr(self._scan_windows[0], units):
            raise AttributeError(f'Units not an attribute on ScanWindow: {units}')
        if ignore_indexed:
            return [
                i for i, sw in enumerate(self._scan_windows) if 
                getattr(sw, units)[0] < end and getattr(sw, units)[-1] > start and 
                getattr(sw, units)[0] != start
            ]
        return [
                i for i, sw in enumerate(self._scan_windows) if 
                getattr(sw, units)[0] < end and getattr(sw, units)[-1] > start
            ]

    def collapse_redundant_components(
        self,
        overlap_threshold: float = 0.5,
        similarity_threshold: float = 0.75,
    ) -> None:
        """ """
        for sw in self._scan_windows:
            sw_fit_mean, sw_fit_sigma = sw.component_fit_parameters

            ol_scan_indices = self.get_overlapping_scanwindow_indices(
                sw.scan_index[0], sw.scan_index[-1], units='scan_index', ignore_indexed=True,
            )
            for oli in ol_scan_indices:
                ol_sw = self._scan_windows[oli]
                ol_fit_mean, ol_fit_sigma = ol_sw.component_fit_parameters
                
                union_mz_bin_indices = np.unique(np.concatenate([sw.mz_bin_indices, ol_sw.mz_bin_indices]))
                sw_mz_bin_indices = np.searchsorted(union_mz_bin_indices, sw.mz_bin_indices)
                sw_component_weights = np.zeros_like(union_mz_bin_indices, dtype=float)
                ol_mz_bin_indices = np.searchsorted(union_mz_bin_indices, ol_sw.mz_bin_indices)
                ol_component_weights = np.zeros_like(union_mz_bin_indices, dtype=float)

                for i, ni in enumerate(sw.component_indices):
                    sw_component_weights[:] = 0.0
                    sw_component_weights[sw_mz_bin_indices] = sw._h[ni]
                    
                    for j, nj in enumerate(ol_sw.component_indices):
                        ol_component_weights[:] = 0.0 
                        ol_component_weights[ol_mz_bin_indices] = ol_sw._h[nj]

                        component_distance = distance.cosine(sw_component_weights, ol_component_weights)
                        overlap_density = approximate_overlap_curves(
                            sw_fit_mean[i], 
                            sw_fit_sigma[i], 
                            ol_fit_mean[j], 
                            ol_fit_sigma[j],
                            bounds=(
                                min(sw_fit_mean[i], ol_fit_mean[j]) - self._window_scan_overlap, 
                                max(sw_fit_mean[i], ol_fit_mean[j]) + self._window_scan_overlap,
                            ),
                        )
                        if (
                            (overlap_density > overlap_threshold) and (component_distance < 1 - similarity_threshold)
                        ):
                            if sw._component_maes[ni] < ol_sw._component_maes[nj]:
                                ol_sw.mask_component(nj)
                            else:
                                sw.mask_component(ni)
                            break 

    def dump_h5(self, h5_fh: Union[str, h5py.File, h5py.Group]) -> None:
        """Save the GPFRun object to an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, 'w')
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Save main attributes
        f.attrs['gpf'] = self._gpf
        f.attrs['gpf_low'] = self._gpf_low if self._gpf_low is not None else -1
        f.attrs['gpf_high'] = self._gpf_high if self._gpf_high is not None else -1
        f.attrs['window_scan_overlap'] = self._window_scan_overlap
        f.attrs['window_scan_width'] = self._window_scan_width

        # Save arrays
        f.create_dataset('scan_number', data=self._scan_number)
        f.create_dataset('scan_index', data=self._scan_index)
        f.create_dataset('bin_indices', data=self._bin_indices)
        f.create_dataset('retention_time', data=self._retention_time)

        # Save sparse matrix
        f.create_dataset('m_data', data=self._m.data)
        f.create_dataset('m_indices', data=self._m.indices)
        f.create_dataset('m_indptr', data=self._m.indptr)
        f.attrs['m_shape'] = self._m.shape

        # Save discretize object
        discretize_grp = f.create_group('discretize')
        self._discretize.dump_h5(discretize_grp)

        # Save scan windows
        if self._scan_windows is not None:
            windows_grp = f.create_group('scan_windows')
            for i, window in enumerate(self._scan_windows):
                window.dump_h5(windows_grp, f'window_{i}')

        if close_file:
            f.close()

    @classmethod
    def from_h5(cls, h5_fh: Union[str, h5py.File, h5py.Group]) -> 'GPFRun':
        """Load a GPFRun object from an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object

        Returns:
            GPFRun: Loaded object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, 'r')
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Load discretize object
        discretize = MzDiscretize.from_h5(f['discretize'])

        # Load sparse matrix
        m = sps.csr_matrix(
            (f['m_data'][:], f['m_indices'][:], f['m_indptr'][:]),
            shape=f.attrs['m_shape']
        )

        # Create instance
        obj = cls(
            gpf=f.attrs['gpf'],
            m=m,
            scan_number=f['scan_number'][:],
            bin_indices=f['bin_indices'][:],
            retention_time=f['retention_time'][:],
            discretize=discretize,
            window_scan_overlap=f.attrs['window_scan_overlap'],
            window_scan_width=f.attrs['window_scan_width'],
            gpf_low=None if f.attrs['gpf_low'] == -1 else f.attrs['gpf_low'],
            gpf_high=None if f.attrs['gpf_high'] == -1 else f.attrs['gpf_high'],
        )

        # Load scan windows if they exist
        if 'scan_windows' in f:
            obj._scan_windows = []
            windows_grp = f['scan_windows']
            for i in range(len(windows_grp)):
                window = ScanWindow.from_h5(windows_grp, f'window_{i}')
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
        config: Config=Config,
    ) -> "GPFRun":
        """ """
        sub_m = m[np.isclose(m[:,0], gpf)]
        discretized_m = discretize.discretize_mz_array(sub_m)

        return cls.from_discretized_long(
            gpf, discretized_m, discretize, config=config,
        )

    @classmethod
    def from_discretized_long(
        cls: Type["GPFRun"],
        gpf: float,  
        m: np.ndarray, 
        discretize: MzDiscretize,
        config: Config=Config,
    ) -> "GPFRun":
        """ """
        sub_m = m[np.isclose(m[:,0], gpf)]
        mz_bin_index = 5 
        gpf_low = None 
        gpf_high = None
        if config.ms2_preprocessing.include_gpf_bounds:
            mz_bin_index += 2
            gpf_low = sub_m[:,5][0]
            gpf_high = sub_m[:,6][0]
        sub_m_p, scans, bin_indices = pivot_unique_binned_mz_sparse(sub_m, mz_bin_index=mz_bin_index)
        retention_times = np.unique(sub_m[:,[1,2]], axis=0)[:,1]

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
        )

class MSRun:

    _gpf_runs = {}

    def __init__(
        self, 
        isolation_low: Optional[float] = None,
        isolation_high: Optional[float] = None,
    ):
        self._isolation_low = isolation_low
        self._isolation_high = isolation_high

    def add_gpf(self, gpf_index: int, gpf: GPFRun) -> None:
        """ """
        self._gpf_runs[gpf_index] = gpf

    def get_tuning_windows(self, config: Config = Config) -> None:
        """ """
        rng = np.random.default_rng(seed=config.random_seed)
        
        windows = []

        for gpf in self._gpf_runs.values():
            scan_windows = gpf.sample_scan_windows(
                window_sampling_fraction=config.tuning.window_sampling_fraction, 
                exclude_scan_window_edges=config.tuning.exclude_scan_window_edges,
                rng=rng,
            )

            if not all(sw.is_filtered for sw in scan_windows):
                for sw in scan_windows:
                    sw.filter_scans()
                    
            windows.extend(
                [sw.m.copy() for sw in scan_windows if sw.m.shape[1] > 0 and sw.m.shape[0] == config.scan_filter.scan_width]
            )

        return windows

    @classmethod
    def from_long(
        cls: Type["MSRun"],
        m: np.ndarray, 
        discretize: MzDiscretize,
        config: Config = Config, 
    ) -> "MSRun":
        """ """

        gpfs = []

        unique_gpfs = np.unique(m[:,0]) 

        for i, gpf_mass in tqdm(
            enumerate(unique_gpfs), 
            disable=(not config.tqdm_enabled),
            total=unique_gpfs.size,
        ):

            sub_m = m[np.isclose(m[:,0], gpf_mass)]
            
            gpf = GPFRun.from_undiscretized_long(
                gpf_mass,
                sub_m,
                discretize,
                config=config,
            )
            gpfs.append((i, gpf_mass, gpf))  

        msrun = cls()
        for i, gpf_mass, gpf in gpfs:
            msrun.add_gpf(i, gpf)          

        return msrun

    def dump_h5(self, h5_fh: Union[str, h5py.File]) -> None:
        """Save the MSRun object to an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, 'w')
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Save attributes
        f.attrs['isolation_low'] = self._isolation_low if self._isolation_low is not None else -1
        f.attrs['isolation_high'] = self._isolation_high if self._isolation_high is not None else -1

        # Save GPF runs
        gpf_runs_grp = f.create_group('gpf_runs')
        for gpf_index, gpf_run in self._gpf_runs.items():
            gpf_run.dump_h5(gpf_runs_grp.create_group(f'gpf_{gpf_index}'))

        if close_file:
            f.close()

    @classmethod
    def from_h5(cls, h5_fh: Union[str, h5py.File, h5py.Group], config: Config = Config) -> 'MSRun':
        """Load an MSRun object from an HDF5 file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object

        Returns:
            MSRun: Loaded object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, 'r')
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Create instance
        obj = cls(
            isolation_low=None if f.attrs['isolation_low'] == -1 else f.attrs['isolation_low'],
            isolation_high=None if f.attrs['isolation_high'] == -1 else f.attrs['isolation_high'],
        )

        # Load GPF runs
        gpf_runs_grp = f['gpf_runs']
        for gpf_name in tqdm(gpf_runs_grp, disable=(not config.tqdm_enabled)):
            gpf_index = int(gpf_name.split('_')[1])
            gpf_run = GPFRun.from_h5(gpf_runs_grp[gpf_name])
            obj.add_gpf(gpf_index, gpf_run)

        if close_file:
            f.close()

        return obj

def fit_scanwindow(
    w: ScanWindow,  
    fitpeaks_kwargs: Dict[str, Any] = {},
) -> None:
    """ """
    mus = []
    sigmas = []
    maes = []
    for i, s in enumerate(w.component_indices):
        (mu, s), pcov, mae = fit_gaussian_elution(
            w.retention_time, 
            w.w[:,i],
            **fitpeaks_kwargs,
        )
        mus.append(mu)
        sigmas.append(s)
        maes.append(mae)
    w.set_peak_fits(
        np.array(mus),
        np.array(sigmas),
        np.array(maes),
    )

def fit_nmf_matrix_scanwindow(
    w: ScanWindow,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    **nmf_kwargs: Dict[str, Any], 
) -> None:
    """ """
    W, H, model = fit_nmf_matrix_custom_init(
        w.m, W_init=W_init, H_init=H_init, return_model=True, **nmf_kwargs,
    )
    w.set_nmf_fit(W, H, model)
