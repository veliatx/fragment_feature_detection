from typing import (
    Union, List, Optional, Any, Tuple, Dict, Callable
)
import numbers
from collections import defaultdict
import warnings
import time
import logging 

import numpy as np 
import optuna 

from sklearn.decomposition import NMF
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.utils._param_validation import Interval
from sklearn.base import BaseEstimator, clone
from sklearn.utils.parallel import Parallel, delayed

logger = logging.getLogger(__name__)


class MzBinMaskingSplitter:
    """Custom splitter that randomly masks bins in scanwindows"""

    def __init__(
        self, 
        n_splits: int = 5, 
        mask_fraction: float = 0.2, 
        random_state: int = 42, 
        mask_signal: bool = False,
        balance_mask_signal: bool = True,
    ):
        self._n_splits = n_splits
        self._mask_fraction = mask_fraction
        self._random_state = random_state
        self._balance_mask_signal = balance_mask_signal
        self._mask_signal = mask_signal

    def split(self, X: List[np.ndarray], y=None):
        """Generate train-test masks by randomly hiding mass bins in scanwindows"""
        rng = np.random.default_rng(seed=self._random_state)
        for _ in range(self._n_splits):
            unmasked = []
            masked = []
            for m in X:
                if not self._mask_signal:
                    sub_mask = rng.random(m.shape) < self._mask_fraction
                else:
                    flat_mask = np.zeros_like(m.flatten(), dtype=bool)
                    non_zero_idxes = np.where((m > 0).flatten())[0]
                    mask_non_zero_idxes = rng.random(non_zero_idxes.shape) < self._mask_fraction                        
                    flat_mask[non_zero_idxes[mask_non_zero_idxes]] = True
                    if self._balance_mask_signal:
                        zero_idxes = np.where((m == 0).flatten())[0]
                        mask_zero_idxes = rng.random(zero_idxes.shape) < (flat_mask.sum() / flat_mask.size)
                        flat_mask[zero_idxes[mask_zero_idxes]] = True
                    sub_mask = flat_mask.reshape(m.shape)
                unmasked.append(~sub_mask)
                masked.append(sub_mask)
            yield unmasked, masked

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._n_splits


class NMFMaskWrapper(BaseEstimator):
    """ """

    def __init__(self, n_splits: int = 5, mask_fraction: float = 0.2, mask_signal: bool = True, **nmf_kwargs: Dict[str, Any]):
        self._rng = np.random.default_rng(seed=nmf_kwargs.get('random_seed', 42))
        self.mask_fraction = mask_fraction 
        self.n_splits = n_splits
        self.mask_signal = mask_signal
        self._nmf_kwargs = nmf_kwargs

    def fit_model(self, m: np.ndarray):
        """ """
        W, H, model = fit_nmf_matrix_custom_init(
            m, return_model=True, **self._nmf_kwargs,
        )
        return W, H, model

    def fit(self, X, y=None):
        """ """
        self._models = []
        self._reconstruction_errors = []
        self._train_reconstruction_errors = []
        self._weight_orthogonality = []
        self._sample_orthogonality = []
        self._test_samples = []
        self._train_samples = []

        for train_idxes, test_idxes in MzBinMaskingSplitter(
            n_splits=self.n_splits, 
            mask_fraction=self.mask_fraction, 
            mask_signal=self.mask_signal,
            random_state=self._nmf_kwargs.get('random_seed', 42)
        ).split(X):
            for m, train_mask, test_mask in zip(X, train_idxes, test_idxes):
                m_train = m.copy()
                m_train[test_mask] = 0.0
                W, H, model = self.fit_model(m_train)
                
                non_zero_components = (~np.isclose(W.sum(axis=0), 0.0)) & (~np.isclose(H.sum(axis=1), 0.0))
                if non_zero_components.sum() >= 2:
                    H_nonzero = H[non_zero_components,:]
                    W_nonzero = W[:,non_zero_components]
                    orthogonality_H = H_nonzero @ H_nonzero.T 
                    orthogonality_W = W_nonzero.T @ W_nonzero
                    weight_identity_matrix_approximation = orthogonality_H / np.linalg.norm(H_nonzero, axis=1)[..., np.newaxis]
                    sample_identity_matrix_approximation = orthogonality_W / np.linalg.norm(W_nonzero, axis=0)[..., np.newaxis]
                    weight_deviation_identity = np.linalg.norm(
                        weight_identity_matrix_approximation - np.eye(H_nonzero.shape[0])
                    )
                    sample_deviation_identity = np.linalg.norm(
                        sample_identity_matrix_approximation - np.eye(W_nonzero.shape[1])
                    )
                else:
                    weight_deviation_identity = 0.0
                    sample_deviation_identity = 0.0

                m_reconstructed = W@H

                self._models.append(model)

                mse = np.nanmean(( m[test_mask] - m_reconstructed[test_mask] ) ** 2)
                mse_train = np.nanmean(( m[train_mask] - m_reconstructed[train_mask] ) ** 2)

                self._weight_orthogonality.append(weight_deviation_identity)
                self._sample_orthogonality.append(sample_deviation_identity)
                self._reconstruction_errors.append(mse)
                self._train_reconstruction_errors.append(mse_train)
                self._test_samples.append(test_mask.sum())
                self._train_samples.append(train_mask.sum())
        
        return self 

    def score(self, X, y=None):
        """ """
        return -1*np.nanmean(self._reconstruction_errors)

    def get_params(self, deep: bool = True):
        """Override get_params to allow access to nmf hyperparameters of interest."""
        return self._nmf_kwargs

    def set_params(self, **params: Dict[str, Any]):
        """Override set_params to update the parameters in _nmf_kwargs"""
        for param, value in params.items():
            if param in self._nmf_kwargs.keys():
                self._nmf_kwargs[param] = value 
            else:
                setattr(self, param, value)
        return self

class RandomizedSearchReconstructionCV(BaseSearchCV):

    _required_parameters = ["estimator", "param_distributions"]

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter 
        self.random_state = random_state 
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates: Callable):
        """ """
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state,
            )
        )

    def fit(self, X: Union[List[Any], np.ndarray], y=None, **params: Dict[str, Any]) -> "RandomizedSearchReconstructionCV":
        """ """
        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        n_splits = self.estimator.n_splits

        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params: ParameterSampler, more_results: Optional[Dict[str, Any]] = None) -> None:
                """ """
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        parameters=parameters,
                    )
                    for parameters in candidate_params
                )
                out = [i for j in out for i in j]

                if len(out) < 1:
                    raise ValueError('No fits were performed.')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('Returned fits not consistent with size of parameter space.')
                
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                
                if more_results:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)
                
                nonlocal results 

                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results,
                )

                return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_score"].argmin()
        self.best_score_ = results["mean_test_score"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.cv_results = results 
        self.n_splits_ = n_splits

        return self 

class OptunaSearchReconstructionCV(BaseSearchCV):

    _required_parameters = ["estimator", "param_distributions"]

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_sampling,
        objective_params,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.param_sampling = param_sampling
        self.n_iter = n_iter 
        self.random_state = random_state 
        self.objective_params = objective_params
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates: Callable):
        """ """
        evaluate_candidates(self.param_sampling, self.n_iter, random_state=self.random_state)

    def fit(self, X: Union[List[Any], np.ndarray], y=None, **params: Dict[str, Any]) -> "RandomizedSearchReconstructionCV":
        """ """
        base_estimator = clone(self.estimator)

        n_splits = self.estimator.n_splits

        results = {}
            
        def evaluate_candidates(candidate_params: Dict[str, Tuple[Any]], more_results: Optional[Dict[str, Any]] = None) -> None:
            """ """
            n_candidates = len(candidate_params.keys())

            def objective(trial: optuna.Trial):
                """ """

                params = {}
                for k, v in candidate_params.items():
                    # Suggest categorical variable.
                    if isinstance(v[0], str):
                        params[k] = trial.suggest_categorical(k, v)
                    # Suggest continuous variable. 
                    elif isinstance(v[0], (float, int)):
                        params[k] = trial.suggest_float(k, *v[:2], log=True if (len(v) > 2 and v[2] == 'log') else False)

                results = _fit_and_score(
                    clone(base_estimator),
                    X, 
                    params,
                )

                for k, v in results[0].items():
                    if isinstance(v, (int, float)):
                        trial.set_user_attr(
                            k, np.array([r[k] for r in results])
                        )
                    elif k in ('train_scores', 'test_scores') and isinstance(v, dict):
                        for sk, _ in v.items():
                            trial.set_user_attr(
                                f'{k}_{sk}', np.array([r[k][sk] for r in results])
                            )
                    
                try:
                    return tuple([-1.*np.nanmean([r['test_score'][k] for r in results]) for k in self.objective_params])
                except:
                    logger.info('Only one objective defined.')

                return -1.*np.nanmean([r['test_score'] for r in results])

            study = optuna.create_study(directions=['maximize' for _ in self.objective_params])
            study.optimize(objective, n_trials=self.n_iter, n_jobs=self.n_jobs)

            out = [t.user_attrs for t in study.trials]
            for i, (t, o) in enumerate(zip(study.trials, out)):
                o.update(
                    t
                )
                            
            if len(out) < 1:
                raise ValueError('No fits were performed.')
            elif len(out) != n_candidates * n_splits:
                raise ValueError('Returned fits not consistent with size of parameter space.')
            
            all_candidate_params.extend(candidate_params)
            all_out.extend(out)
            
            if more_results:
                for key, value in more_results.items():
                    all_more_results[key].extend(value)
            
            nonlocal results 

            results = self._format_results(
                all_candidate_params, n_splits, all_out, all_more_results,
            )

            return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_score"].argmin()
        self.best_score_ = results["mean_test_score"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.cv_results = results 
        self.n_splits_ = n_splits

        return self 


def _fit_and_score(
    estimator: BaseEstimator, 
    X: Union[List[Any], np.ndarray], 
    parameters: Dict[str, Any] = {},
    **kwargs: Dict[str, Any],
) -> Tuple[float, float]:
    """ """
    start_time = time.time()
    fit_error = None
    score_time = 0.0 
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            estimator.set_params(**parameters)
            estimator.fit(X)
    except Exception as e:
        fit_time = time.time() - start_time
        fit_error = str(e)
        return [
            {
                'estimator': estimator._models[i],
                'fit_time': fit_time,
                'score_time': score_time,
                'fit_error': fit_error,
            } for i in range(estimator.n_splits)
        ]
    fit_time = time.time() - start_time 

    results = [
        {
            'test_scores': -1.*estimator._reconstruction_errors[i],
            'train_scores': -1.*estimator._train_reconstruction_errors[i],
            'n_test_samples': estimator._test_samples[i],
            'estimator': estimator._models[i],
            'fit_time': fit_time,
            'score_time': score_time,
            'fit_error': fit_error,
        } for i in range(estimator.n_splits)
    ]

    if hasattr(estimator, '_weight_orthogonality') and hasattr(estimator, '_sample_orthogonality'):
        for i, r in enumerate(results):
            r['test_scores'] = {
                'score': r['test_scores'],
                'weight_orthogonality': estimator._weight_orthogonality[i],
                'sample_orthogonality': estimator._sample_orthogonality[i]
            }
            r['train_scores'] = {
                'score': r['train_scores'],
                'weight_orthogonality': estimator._weight_orthogonality[i],
                'sample_orthogonality': estimator._sample_orthogonality[i]
            }

    return results 


def fit_nmf_matrix_custom_init(
    m: np.ndarray,
    n_components: int = 20,
    alpha_W: float = 0.00001,
    alpha_H: float = 0.0375,
    l1_ratio: float = 0.75,
    max_iter: int = 500,
    solver: str = "cd",
    random_seed: int = 42, 
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    return_model: bool = False,
    **nmf_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Optional[NMF]]:
    """Fits NMF model with optional custom initialization.
    
    Args:
        m (np.ndarray): Input data matrix
        max_components (int): Maximum number of components
        alpha_W (float): L1/L2 regularization parameter for W matrix
        alpha_H (float): L1/L2 regularization parameter for H matrix
        l1_ratio (float): Ratio of L1 vs L2 regularization
        max_iter (int): Maximum number of iterations
        W_init (np.ndarray): Initial W matrix
        H_init (np.ndarray): Initial H matrix
        return_model (bool): Whether to return fitted model
        **nmf_kwargs (Any): Additional keyword arguments for NMF
        
    Returns:
        W (np.ndarray): W matrix
        H (np.ndarray): H matrix
        model (Optional[NMF]): Fitted NMF model if return_model is True
    """
    model = NMF(
        n_components=min(n_components, min(m.shape)),
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        solver=solver,
        init=(
            "custom"
            if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray)
            else nmf_kwargs.get("init", "nndsvd")
        ),
        random_state=random_seed,
        **nmf_kwargs,
    )
    if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray):
        W = model.fit_transform(m, W=W_init, H=H_init)
    else:
        W = model.fit_transform(m)

    if return_model:
        return W, model.components_, model
    return W, model.components_

def tune_hyperparameters_randomizedsearchcv(
    ms: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_seed: int = 42,
    nmf_param_grid: Dict[str, Any] = {},
) -> None: 
    """ """
    rng = np.random.default_rng(seed=random_seed)
    cv_errors = []

    for i in range(n_splits):
        pass
