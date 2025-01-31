from typing import Any

import pandas as pd
from pyteomics import mzml
import numpy as np


class MS2OnlyMzML(mzml.MzML):
    """Reads ms2 scans from mzml."""

    _default_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="2"]]'
    )
    _use_index = False
    _iterative = False


class MS1OnlyMzML(mzml.MzML):
    """Reads ms1 scans from mzml."""

    _default_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="1"]]'
    )
    _use_index = False
    _iterative = False


class MzMLParser(mzml.MzML):
    _use_index = False
    _iterative = False 

    def __init__(self, *args: Any, parse_ms2: bool = True, parse_ms1: bool = True, **kwargs: Any):
        pass
