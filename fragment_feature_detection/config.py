import configparser
import logging
from pathlib import Path
from typing import Any, Union


def format_logger(logger: logging.Logger, level: int = logging.INFO) -> None:
    """Configure basic formatting for a logger.

    Args:
        logger: Logger instance to configure
    """
    formatter = logging.Formatter(
        fmt="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    logger.setLevel(level)


class AttributeDict(dict):
    """Dictionary with attribute-style access"""

    def __getattr__(self, item: str) -> Any:
        """ """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")

    def __setattr__(self, key: Any, value: Any) -> None:
        """ """
        self[key] = value

    def __repr__(self) -> str:
        """ """
        items = [f"{k}={v}" for k, v in self.items()]
        return f"{self.__class__.__name__}({', '.join(items)})"


class Config:
    random_seed = 42
    tqdm_enabled = False
    lazy = True
    downcast_intensities = True

    h5 = AttributeDict(
        **{
            "scan_window_save_m_unfilter": False,
            "scan_window_filter_during_load": True,
        }
    )

    discretization = AttributeDict(
        **{
            "bin_width": "ppm",
            "mz_low": 200.0,
            "mz_high": 2000.0,
            "steps": 2,
            "tolerance": 20.0 / 1e6,
        }
    )

    tuning = AttributeDict(
        **{
            "run_tune": True,
            "optuna": True,
            "window_sampling_n": 50,
            "window_sampling_fraction": 0.0125,
            "exclude_scan_window_edges": 10,
            "optuna_hyperparameter_grid": {
                "l1_ratio": (0.1, 0.9),
                "alpha_W": (0.0001, 0.1, "log"),
                "alpha_H": (0.0001, 0.1, "log"),
            },
            "hyperparameter_grid": {
                "alpha_W": [0.0001, 0.0005, 0.001, 0.005, 0.01],
                "alpha_H": [0.0001, 0.0005, 0.001, 0.005, 0.01],
                "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "objective_params": [
                "test_reconstruction_errors",
                "neglog_ratio_train_test_reconstruction_error",
                "sample_orthogonality",
                "weight_orthogonality",
                "fraction_window_component",
            ],
            "n_iter": 500,
            "n_components": 20,
            "components_in_window": 8.0,
            "component_sigma": 2.0,  # In units of scans (95% of peak density falls within 8 scans.)
            "n_splits": 3,
            "test_fraction": 0.2,
            "splitter_type": "Mask",
        }
    )

    ms2_preprocessing = AttributeDict(
        **{
            "include_gpf_bounds": True,
            "filter_spectra": False,
            "filter_spectra_percentile": 50,
        }
    )

    scan_filter = AttributeDict(
        **{
            "scan_overlap": 30,
            "scan_width": 150,
        }
    )

    nmf = AttributeDict(
        **{
            "n_components": 20,
            "alpha_W": 0.00001,
            "alpha_H": 0.0375,
            "l1_ratio": 0.75,
            "max_iter": 500,
            "solver": "cd",
        }
    )

    fitpeaks = AttributeDict(
        **{
            "mu_bounds": None,
            "sigma_bounds": [0.1, 10.0],
        }
    )

    feature_matching = AttributeDict(
        **{
            "match_ms1": True,
            "alpha": 0.1,
            "c_cutoff": 0.025,
            "extend_w_fraction": 0.5,
        }
    )

    feature_output = AttributeDict(
        **{
            "top_n_rescaled_peaks": 200,  # Number of peaks to report for weight/reweighted spectra.
            "matching_coefficient_threshold": 0.1,  # Minimum reporting threshold.
            "top_n_ms1_matches": 4,  # Number of matches to report before ties.
        }
    )

    def __repr__(self):
        """Return string representation of Config showing all attributes"""
        items = []

        attributes = dict(**vars(self.__class__))
        attributes.update(vars(self))

        for key, value in attributes.items():
            if not key.startswith("_"):
                if isinstance(value, AttributeDict):
                    items.append(f"{key}={value}")
                else:
                    items.append(f"{key}={repr(value)}")
        attribute_sep = "\n  "
        return f"{self.__class__.__name__}(\n  {(', '+attribute_sep).join(items)}\n)"

    @classmethod
    def from_ini(cls, ini_path: Union[str, Path]) -> "Config":
        """Load configuration from an INI file.

        Args:
            ini_path (str): Path to the INI file

        Returns:
            Config: New config instance with values from INI file
        """
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(ini_path)

        instance = cls()

        for section in config.sections():
            if hasattr(instance, section):
                section_dict = getattr(instance, section)
                # Update existing AttributeDict sections
                for key, value in config[section].items():
                    # Convert string values to appropriate types
                    try:
                        typed_value = eval(value)
                    except ValueError:
                        typed_value = value
                    section_dict[key] = typed_value
            else:
                # Is general setting section:
                for key, value in config[section].items():
                    try:
                        typed_value = eval(value)
                    except ValueError:
                        typed_value = value
                    setattr(instance, key, typed_value)

        return instance

    def to_ini(self, ini_path: Union[Path, str]) -> None:
        """Save current configuration to an INI file.

        Args:
            ini_path (str): Path where to save the INI file
        """
        config = configparser.ConfigParser()

        attributes = dict(**vars(self.__class__))
        attributes.update(vars(self))

        config["general"] = {}
        for key, value in attributes.items():
            if (
                not key.startswith("_")
                and not isinstance(value, classmethod)
                and not callable(value)
            ):
                if isinstance(value, AttributeDict):
                    config[key] = {}
                    for k, v in value.items():
                        config[key][k] = repr(v)
                else:
                    config["general"][key] = repr(value)

        with open(ini_path, "w") as f:
            config.write(f)


class Constants:
    isotope_mu = 1.008
    proton_mu = 1.0073
