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

from .gpfrun import GPFRun
from .scanwindow import ScanWindow
from ..discretization import MzDiscretize
from ..config import Config

logger = logging.getLogger(__name__)


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
