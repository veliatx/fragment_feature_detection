class AttributeDict(dict):
    """Dictionary with attribute-style access"""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value


class Config:
    random_seed = 42
    tqdm_enabled = True

    discretization = AttributeDict(
        **{
            'bin_width': "ppm",
            "mz_low": 200.0, 
            "mz_high": 2000.0,
            "steps": 2,
            "tolerance": 60.0 / 1e6,
        }
    )

    tuning = AttributeDict(
        **{
            "window_sampling_fraction": 0.0125,
            "exclude_scan_window_edges": 10,
        }
    )

    ms2_preprocessing = AttributeDict(
        **{
            "include_gpf_bounds": True,
            "filter_spectra": False,
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
            "solver": "cd",  # "mu",
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
            "alpha": 0.1,
        }
    )
