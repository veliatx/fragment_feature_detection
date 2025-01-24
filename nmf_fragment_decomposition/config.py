
class Config:
    tqdm_enabled = False

    scan_filter = {
        "scan_overlap": 30,
    }

    nmf = {
        "n_components": 20,
        "alpha_W": 0.00001,
        "alpha_H": 0.0375,
        "l1_ratio": 0.75,
        "max_iter": 500,
        "solver": "cd", # "mu",
    }

    fitpeaks = {
        "mu_bounds": None,
        "sigma_bounds": [0.1, 10.0],
    }
