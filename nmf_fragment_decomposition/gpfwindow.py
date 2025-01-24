from typing import (
    Union, Optional, Tuple, Type, Literal, List, Dict, Any,
)
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF

import scipy.ndimage as ndi 
import scipy.sparse as sps 
from scipy.spatial import distance

from discretization import MzDiscretize
from utils import pivot_unique_binned_mz_sparse
from fitpeaks import approximate_overlap_curves, fit_gaussian_elution
from decomposition import fit_nmf_matrix_custom_init

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
    _is_fit = False
    _denoise_scans = True
    _scale_scans = True
    _filter_edge_scans = True

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
        mz_low: Optional[float] = None,
        mz_high: Optional[float] = None,
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
        self._mz_low = mz_low 
        self._mz_high = mz_high
        self._mz_masked = np.zeros_like(bin_indices, dtype=bool)
        self._mz_unfilter = mz
        self._mz = mz.copy()
        self._mz_bin_indices_unfilter = bin_indices
        self._mz_bin_indices = bin_indices.copy()
        self._m_unfilter = m 
        self._m = m.toarray()
        self._m_max= None
        self._m_max_bin_indices = None

    @property 
    def m(self) -> np.ndarray:
        """ """
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

    def mask_component(self, component_index) -> None:
        """ """
        if not getattr(self, '_is_component_fit', False):
            raise AttributeError
        self._component_keep[component_index] = False 

    def filter_scans(self) -> None:
        self.filter_zero_mz()
        self.filter_clip_constant()
        if self._scale_scans:
            self.transform_maxscale_scans()
        if self._denoise_scans:
            self.filter_denoise_scans()
        if self._filter_edge_scans:
            self.filter_edge_scans()

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
        if axis == 1:
            m_m = m_m[..., np.newaxis]
        self._m_max = m_m
        self._m_max_bin_indices = self._mz_bin_indices.copy()
        self._m = self._m / m_m

    def filter_clip_constant(self, p: float = 0.3) -> None:
        """ """
        mask = ~(((self._m / self._m.max(axis=0)).mean(axis=0)) > p)
        self.filter_mz_axis(mask)

    def filter_denoise_scans(
        self,
        grey_erosion_size: int = 2, 
        grey_closing_size: int = 2, 
    ) -> np.ndarray:
        """ """
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
        )

    def build_scan_windows(self) -> None:
        """ """
        self._scan_windows = [
            self.slice_scan_window(i, i+self._window_scan_width)
            for i in range(
                0, self._scan_number.shape[0] - self._window_scan_overlap, self._window_scan_overlap,
            )
        ]
    
    def get_overlapping_scanwindow_indices(
        self,
        start: float, 
        end: float, 
        units: Literal['retention_time', 'scan_index', 'scan_number'] = 'scan_index',
        ignore_indexed: bool = True, 
    ) -> List[int]:
        """ """
        if not hasattr(self._scan_windows[0], units):
            raise AttributeError
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

    def dump_h5(self, h5_fh: Path) -> None:
        """ """
        pass

    @classmethod
    def read_h5(cls, h5_fh: Path) -> None: 
        """ """
        pass 

    @classmethod 
    def from_undiscretized_long(
        cls: Type["GPFRun"], 
        gpf: float,
        m: np.ndarray, 
        discretize: MzDiscretize,
    ) -> "GPFRun":
        """ """
        sub_m = m[np.isclose(m[:,0], gpf)]
        discretized_m = discretize.discretize_mz_array(sub_m)

        return cls.from_discretized_long(
            gpf, discretized_m, discretize,
        )

    @classmethod
    def from_discretized_long(
        cls: Type["GPFRun"],
        gpf: float,  
        m: np.ndarray, 
        discretize: MzDiscretize,
    ) -> "GPFRun":
        """ """
        sub_m = m[np.isclose(m[:,0], gpf)]
        sub_m_p, scans, bin_indices = pivot_unique_binned_mz_sparse(sub_m)
        retention_times = np.unique(sub_m[:,[1,2]], axis=0)[:,1]

        return cls(
            gpf, 
            sub_m_p,
            scans, 
            bin_indices,
            retention_times,
            discretize,
        )

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
