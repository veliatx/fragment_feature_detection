from typing import Any, Dict, Union, Literal, Optional, Type
from pathlib import Path
import logging 
from collections import defaultdict

import pandas as pd
from pyteomics import mzml
import numpy as np
from tqdm import tqdm 
import h5py

from fragment_feature_detection.config import Config 

logger = logging.getLogger(__name__)

class MzMLParser(mzml.MzML):
    _ms1_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="1"]]'
    )
    _ms2_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="2"]]'
    )
    _default_iter_path = (
        '//spectrum[./*[local-name()="cvParam"]]'
    )
    _use_index = False
    _iterative = False 

    def __init__(self, *args: Any, keep_level: Literal[1, 2] = 2, **kwargs: Any):
        if keep_level == 1:
            self._default_iter_path = self._ms1_iter_path 
        elif keep_level == 2: 
            self._default_iter_path = self._ms2_iter_path
        super().__init__(*args, **kwargs)
    
    @classmethod 
    def to_ms2_h5(
        cls: Type["MzMLParser"], 
        path: Union[Path, str], 
        h5_fh: Optional[Union[Path, str]] = None, 
        config: Config = Config,
        chunk_size: int = 2500,
    ) -> None:
        """ """
        if not h5_fh:
            h5_fh = Path(path).with_suffix('.h5')
        
        obj = cls(source=str(path), keep_level=2)

        def write_chunks() -> None:
            """ """
            nonlocal mz_arrays
            existing_groups = [
                m for m in f.keys() if isinstance(f[m], h5py.Group)
            ]

            for m, a in mz_arrays.items():
                if len(a) == 0: 
                    continue
                if m not in existing_groups:
                    n_columns = len(a[0])
                    group = f.create_group(m)
                    dataset = group.create_dataset(
                        'ms2_long',
                        shape=(0, n_columns),
                        maxshape=(None, n_columns),
                        compression='gzip',
                        dtype="float32"
                    )
                dataset = f[m]['ms2_long']
                dataset.resize((dataset.shape[0] + len(a), dataset.shape[1]))
                dataset[-1*len(a):] = np.vstack(a)
            
            f.flush()
            del mz_arrays
            mz_arrays = defaultdict(list)

        f = h5py.File(h5_fh, "w")

        mz_arrays = defaultdict(list)

        with tqdm(disable=(not config.tqdm_enabled)) as pbar:
            while True:
                try:
                    scan = cls.parse_scan(next(obj), keep_level=2)
                    mz_array = scan['m/zArray']
                    intensity_array = scan['IntensityArray']
                    if config.ms2_preprocessing.filter_spectra:
                        percentile_intensity = np.percentile(scan['IntensityArray'], config.ms2_preprocessing.filter_spectra_percentile)
                        mz_array = mz_array[intensity_array > percentile_intensity]
                        intensity_array = intensity_array[intensity_array > percentile_intensity]

                    scan_key = str(scan['MS2TargetMass'])
                    base_scan_data = [scan['MS2TargetMass'], scan['ScanNum'], scan['RetentionTime']]

                    if config.ms2_preprocessing.include_gpf_bounds:
                        extra_data = [scan['lowerMass'], scan['higherMass']]
                        mz_arrays[scan_key].extend(
                            [
                                base_scan_data + [mz, intensity] + extra_data for mz, intensity in zip(
                                    mz_array, intensity_array
                                )
                            ]
                        )
                    else:
                        mz_arrays[scan_key].extend(
                            [
                                base_scan_data + [mz, intensity] for mz, intensity in zip(
                                    mz_array, intensity_array
                                )
                            ]
                        )

                    pbar.update(1)
                    if pbar.n % chunk_size == 0:
                        write_chunks()
                except StopIteration:
                    write_chunks()
                    break
                except Exception as e:
                    logger.exception(e)

    @classmethod
    def to_ms2_long(cls: Type["MzMLParser"], path: Union[Path, str], config: Config = Config) -> np.ndarray:
        """ """
        obj = cls(source=str(path), keep_level=2)

        mz_arrays = []

        with tqdm(disable=(not config.tqdm_enabled)) as pbar:
            while True:
                try:
                    scan = cls.parse_scan(next(obj), keep_level=2)
                    mz_array = scan['m/zArray']
                    intensity_array = scan['IntensityArray']
                    if config.ms2_preprocessing.filter_spectra:
                        percentile_intensity = np.percentile(scan['IntensityArray'], config.ms2_preprocessing.filter_spectra_percentile)
                        mz_array = mz_array[intensity_array > percentile_intensity]
                        intensity_array = intensity_array[intensity_array > percentile_intensity]

                    base_scan_data = [scan['MS2TargetMass'], scan['ScanNum'], scan['RetentionTime']]

                    if config.ms2_preprocessing.include_gpf_bounds:
                        extra_data = [scan['lowerMass'], scan['higherMass']]
                        mz_arrays.extend(
                            [
                                base_scan_data + [mz, intensity] + extra_data for mz, intensity in zip(
                                    mz_array, intensity_array
                                )
                            ]
                        )
                    else:
                        mz_arrays.extend(
                            [
                                base_scan_data + [mz, intensity] for mz, intensity in zip(
                                    mz_array, intensity_array
                                )
                            ]
                        )
                    pbar.update(1)
                except StopIteration:
                    break
                except Exception as e:
                    logger.exception(e)

        return np.vstack(mz_arrays)

    @classmethod
    def to_ms2_df(cls: Type["MzMLParser"], path: Union[Path, str], config: Config = Config) -> pd.DataFrame:
        """ """
        obj = cls(source=str(path), keep_level=2)

        scans = []
        
        with tqdm(disable=(not config.tqdm_enabled)) as pbar:
            while True:
                try:
                    scan = cls.parse_scan(next(obj), keep_level=2)
                    scans.append(scan)
                    pbar.update(1)
                except StopIteration:
                    break
                except Exception as e:
                    logger.exception(e)
                

        return pd.DataFrame.from_records(scans)

    @staticmethod
    def parse_scan(s: Dict[str, Any], keep_level: Literal[1, 2] = 1) -> Union[Dict[str, Any], None]:
        """ """
        if s['ms level'] != keep_level:
            return 
        
        scan = {}

        scan['ScanNum'] = s['index'] + 1
        scan['RetentionTime'] = s['scanList']['scan'][0]['scan start time']
        scan['m/zArray'] = s['m/z array']
        scan['IntensityArray'] = s['intensity array']

        if keep_level == 2:
            scan['MS2TargetMass'] = s['precursorList']['precursor'][0]['isolationWindow'][
                'isolation window target m/z'
            ]
            scan['PrecursorDetected'] = s['precursorList']['precursor'][0]['selectedIonList'][
                'selectedIon'
            ][0]['selected ion m/z']

            scan['lowerMass'] = (
                scan['PrecursorDetected'] - s['precursorList']['precursor'][0]['isolationWindow'][
                    'isolation window lower offset'
                ]
            )
            scan['higherMass'] = (
                scan['PrecursorDetected'] + s['precursorList']['precursor'][0]['isolationWindow'][
                    'isolation window upper offset'
                ]
            )

            scan['IsolationSize'] = scan['higherMass'] - scan['lowerMass']

        elif keep_level == 1:
            scan['lowerMass'] = s['scanList']['scan'][0]['scanWindowList']['scanWindow'][0]['scan window lower limit']
            scan['higherMass'] = s['scanList']['scan'][0]['scanWindowList']['scanWindow'][0]['scan window upper limit']

        return scan




