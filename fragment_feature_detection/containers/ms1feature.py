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

import numpy as np
import h5py
import pandas as pd

logger = logging.getLogger(__name__)


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
