from typing import (
    Union, List, Optional, Any, Tuple, Dict, Callable, Literal
)
import numbers
from collections import defaultdict
import warnings
import time
import base64 
import pickle
import tempfile
import logging 

import numpy as np 
import optuna 

from sklearn.decomposition import NMF
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.utils._param_validation import Interval
from sklearn.base import BaseEstimator, clone
from sklearn.utils.parallel import Parallel, delayed

from utils import calculate_hoyer_sparsity
from config import Config

logger = logging.getLogger(__name__)

_FIT_AND_SCORE = [
    '_test_reconstruction_errors',
    '_weight_orthogonality',
    '_sample_orthogonality',
    '_nonzero_component_fraction',
    '_neglog_ratio_train_test_reconstruction_error',
    '_mean_weight_sparsity',
    '_mean_sample_sparsity',
    '_fraction_window_component',
]

class OptimizationParameters:

    _components_in_window = 5.0
    _components = 20.0
    _scan_width = 150.0
    _component_sigma = 4.0

    def __init__(self):
        """ """
        self.set_params()

    def set_params(self) -> None:
        """ """
        self.scores = {
            'weight_orthogonality': 0.0,
            'sample_orthogonality': 0.0,
            'nonzero_component_fraction': self._components_in_window / self._components,
            'mean_weight_sparsity': -1.0,
            'mean_sample_sparsity': -1.0,
            # 'fraction_window_component': (self._components_in_window * 4 * self._component_sigma) / self._scan_width,
            'fraction_window_component': 0.0,
        }
    
    def score(self, param: str) -> float:
        """ """
        pass
    
    @classmethod 
    def from_config(cls, config: Config = Config) -> 'OptimizationParameters':
        """ """
        return cls()


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
    
    @staticmethod
    def mask_train_matrix(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ """
        m[mask] = 0.0
        return m
    
    @staticmethod 
    def mean_squared_error(
        m: np.ndarray, 
        m_reconstructed: np.ndarray, 
        mask: np.ndarray, 
        *args: Any,
    ) -> float:
        """ """
        return np.nanmean((m[mask] - m_reconstructed[mask]) ** 2)

class MzBinSampleSplitter:
    """Custom splitter that samples scans in scanwindows"""

    def __init__(
        self, 
        n_splits: int = 5, 
        mask_fraction: float = 0.2, 
        random_state: int = 42, 
        **kwargs: Dict[str, Any],
    ):
        self._n_splits = n_splits
        self._mask_fraction = mask_fraction
        self._random_state = random_state

    def split(self, X: List[np.ndarray], y=None):
        """Generate train-test masks by randomly hiding mass bins in scanwindows"""
        rng = np.random.default_rng(seed=self._random_state)
        for _ in range(self._n_splits):
            unmasked = []
            masked = []
            for m in X:
                sub_mask = rng.random(m.shape[0]) < self._mask_fraction
                unmasked.append(~sub_mask)
                masked.append(sub_mask)
            yield unmasked, masked

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._n_splits
    
    @staticmethod
    def mask_train_matrix(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ """
        return m[~mask]
    
    @staticmethod 
    def mean_squared_error(
        m: np.ndarray, 
        m_reconstructed: np.ndarray, 
        mask: np.ndarray, 
        model: NMF,
    ) -> float:
        """ """
        if model.components_.sum() == 0.0 or m[mask].sum() == 0.0:
            return np.nanmean((m[mask] - np.zeros_like(m[mask])) ** 2)
        mt = model.transform(m[mask])
        m_reconstructed = mt @ model.components_

        return np.nanmean((m[mask] - m_reconstructed) ** 2)

class NMFMaskWrapper(BaseEstimator):
    """ """

    def __init__(
        self, 
        splitter_type: Literal['Sample', 'Masking'] = 'Masking',
        n_splits: int = 5, 
        mask_fraction: float = 0.2, 
        mask_signal: bool = True, 
        balance_mask_signal: bool = True,
        **nmf_kwargs: Dict[str, Any],
    ):
        self.splitter_type = splitter_type
        self._rng = np.random.default_rng(seed=nmf_kwargs.get('random_state', 42))
        self.mask_fraction = mask_fraction 
        self.n_splits = n_splits
        self.mask_signal = mask_signal
        self.balance_mask_signal = balance_mask_signal
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
        self._test_reconstruction_errors = []
        self._train_reconstruction_errors = []
        self._neglog_ratio_train_test_reconstruction_error = []
        self._nonzero_component_fraction = []
        self._fraction_window_component = []
        self._weight_orthogonality = []
        self._sample_orthogonality = []
        self._test_samples = []
        self._train_samples = []
        self._mean_weight_sparsity = []
        self._mean_sample_sparsity = []

        if self.splitter_type == "Mask":
            splitter = MzBinMaskingSplitter(
                n_splits=self.n_splits, 
                mask_fraction=self.mask_fraction, 
                mask_signal=self.mask_signal,
                balance_mask_signal=self.balance_mask_signal,
                random_state=self._nmf_kwargs.get('random_state', 42)
            )
        elif self.splitter_type == "Sample": 
            splitter = MzBinSampleSplitter(
                n_splits=self.n_splits, 
                mask_fraction=self.mask_fraction, 
                random_state=self._nmf_kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(
                f"Invalid splitter type: '{self.splitter_type}'. Must be either 'Masking' or 'Sample'."
            )
        
        for train_idxes, test_idxes in splitter.split(X):
            for m, train_mask, test_mask in zip(X, train_idxes, test_idxes):
                m_train = m.copy()
                m_train = splitter.mask_train_matrix(m_train, test_mask)
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
                    # We usually want to maximize sparsity, so making these negative here because the sign gets 
                    # automatically inverted in _fit_and_score. 
                    weight_sparsity = -1.*np.apply_along_axis(calculate_hoyer_sparsity, axis=0, arr=W_nonzero).mean()
                    sample_sparsity = -1.*np.apply_along_axis(calculate_hoyer_sparsity, axis=1, arr=H_nonzero).mean()
                else:
                    weight_deviation_identity = 0.0
                    sample_deviation_identity = 0.0
                    weight_sparsity = 0.0
                    sample_sparsity = 0.0

                m_reconstructed = W@H

                self._models.append(model)
                
                mse = splitter.mean_squared_error(
                    m, 
                    m_reconstructed,
                    test_mask,
                    model,
                )
                mse_train = splitter.mean_squared_error(
                    m, 
                    m_reconstructed,
                    train_mask,
                    model,
                )

                self._nonzero_component_fraction.append((W.sum(axis=0) > 0).sum() / W.shape[1])
                self._mean_weight_sparsity.append(weight_sparsity)
                self._mean_sample_sparsity.append(sample_sparsity)
                self._weight_orthogonality.append(np.log2(weight_deviation_identity + 1.))
                self._sample_orthogonality.append(np.log2(sample_deviation_identity + 1.))
                self._neglog_ratio_train_test_reconstruction_error.append(-1.0*np.log2(mse_train / mse))
                self._test_reconstruction_errors.append(mse)
                self._train_reconstruction_errors.append(mse_train)
                self._test_samples.append(test_mask.sum())
                self._train_samples.append(train_mask.sum())
                self._fraction_window_component.append((W.sum(axis=1) > 0).sum() / W.shape[0])
        
        return self 

    def score(self, X, y=None):
        """ """
        return -1*np.nanmean(self._reconstruction_errors)

    def get_params(self, deep: bool = True):
        """Override get_params to allow access to nmf hyperparameters of interest."""
        return {**super().get_params(deep=deep), **self._nmf_kwargs}
    
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

            def evaluate_candidates(candidate_params: ParameterSampler, more_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                """ """
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        parameters=parameters,
                        extra_scores=_FIT_AND_SCORE,
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

    _prune_at_zero = [
        '_weight_orthogonality',
        '_sample_orthogonality',
        '_nonzero_component_fraction',
        '_mean_weight_sparsity',
        '_mean_sample_sparsity',
        '_fraction_window_component',
    ]

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
        n_jobs: Optional[int] = None,
        verbose: int =0,
        pre_dispatch: str = "2*n_jobs",
        random_state: Optional[int] = None,
        error_score: float = np.nan,
        return_train_score: bool =False,
        prune_bad_parameter_spaces: bool = False,
        optimization_parameters: OptimizationParameters = OptimizationParameters(),
    ):
        self.param_sampling = param_sampling
        self.n_iter = n_iter 
        self.random_state = random_state 
        self.objective_params = objective_params
        self.prune_bad_parameter_spaces = prune_bad_parameter_spaces
        self.optimization_parameters = optimization_parameters
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
        evaluate_candidates(self.param_sampling)

    def fit(self, X: Union[List[Any], np.ndarray], y=None, **params: Dict[str, Any]) -> "RandomizedSearchReconstructionCV":
        """ """
        base_estimator = clone(self.estimator)

        n_splits = self.estimator.n_splits

        results = {}
        best_trials = []
        
        all_more_results = defaultdict(list)

        def evaluate_candidates(
            candidate_params: Dict[str, Tuple[Any]], 
            more_results: Optional[Dict[str, Any]] = None, 
        ) -> Dict[str, Any]:
            """ """

            def objective(trial: optuna.Trial):
                """ """
                try:
                        
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
                        parameters=params,
                        extra_scores=_FIT_AND_SCORE,
                    )

                    # This is incompatible with optuna trial pruning. 

                    # if self.prune_bad_parameter_spaces:
                    #     for k in self._prune_at_zero:
                    #         if (
                    #             k.lstrip('_') in results[0]['test_scores'].keys() and \
                    #             k.lstrip('_') in self.objective_params and \
                    #             np.nanmean([r['test_scores'][k.lstrip('_')] for r in results]) == 0.0
                    #         ):
                    #             raise Exception('zero parameter space explored.')

                    if any([r['fit_error'] for r in results]):
                        raise Exception('One or more fits failed in cv.')

                    trial.set_user_attr('out', base64.b64encode(pickle.dumps(results)).decode("utf-8"))

                    try:
                        output_parameters = []
                        for k in self.objective_params:
                            parameter_value = np.nanmean([r['test_scores'][k] for r in results])
                            if self.prune_bad_parameter_spaces and f'_{k}' in self._prune_at_zero and parameter_value == 0.0:
                                parameter_value = -10.0
                            if k in self.optimization_parameters.scores.keys():
                                parameter_value = -1.0 * np.abs(parameter_value + self.optimization_parameters.scores[k])
                            output_parameters.append(parameter_value)
                        return tuple(output_parameters)
                    except:
                        logger.info('Only one objective defined.')

                    return np.nanmean([r['test_scores'] for r in results])
                    
                except Exception as e:
                    print([r['fit_error'] for r in results])
                    print(e)
                    trial.set_user_attr("error", str(e))
                    trial.set_user_attr("out", base64.b64encode(pickle.dumps([None] * int(base_estimator.n_splits))).decode("utf-8"))
                    raise optuna.TrialPruned()

            def optimize_optuna_study(study_name: str, storage_string: str, objective: Callable, n_trials: int) -> None:
                """ """
                study = optuna.create_study(study_name=study_name, storage=storage_string, load_if_exists=True)
                study.optimize(objective, n_trials=n_trials)

            with tempfile.NamedTemporaryFile(dir='/dev/shm', suffix=".db") as temp_db:
                storage_string = f"sqlite:///{temp_db.name}"
                study = optuna.create_study(storage=storage_string, directions=['maximize' for _ in self.objective_params])
                
                with Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch) as parallel:
                    parallel(
                        delayed(optimize_optuna_study)(
                            study.study_name,
                            storage_string,
                            objective,
                            (self.n_iter // self.n_jobs) + 1,
                        ) for _ in range(self.n_jobs)
                    )
            all_out = [
                r for t in study.trials for r in pickle.loads(base64.b64decode(t.user_attrs['out']))
                if not t.user_attrs.get('error', None)
            ]
                            
            if len(all_out) < 1:
                raise ValueError('No fits were performed.')
            elif len(all_out) < self.n_iter * n_splits:
                raise ValueError('Returned fits not consistent with size of parameter space.')
            
            all_candidate_params = [t.params for t in study.trials if not t.user_attrs.get('error', None)]
            
            if more_results:
                for key, value in more_results.items():
                    all_more_results[key].extend(value)
            
            nonlocal results 

            results = self._format_results(
                all_candidate_params, n_splits, all_out, all_more_results,
            )

            nonlocal best_trials 

            best_trials = [{'params': t.params, 'scores': dict(zip(self.objective_params, t._values))} for t in study.best_trials]

            return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_score"].argmin()
        self.best_score_ = results["mean_test_score"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]
        self.best_trials_ = best_trials

        self.cv_results = results 
        self.n_splits_ = n_splits

        return self 


def _fit_and_score(
    estimator: BaseEstimator, 
    X: Union[List[Any], np.ndarray], 
    extra_scores: Optional[List[str]] = None,
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
            'test_scores': -1.*estimator._test_reconstruction_errors[i],
            'train_scores': -1.*estimator._train_reconstruction_errors[i],
            'n_test_samples': estimator._test_samples[i],
            'estimator': estimator._models[i],
            'fit_time': fit_time,
            'score_time': score_time,
            'fit_error': fit_error,
        } for i in range(estimator.n_splits)
    ]

    if extra_scores and all(hasattr(estimator, es) for es in extra_scores):
        for i, r in enumerate(results):
            r['test_scores'] = {
                'score': r['test_scores'],
            }
            r['test_scores'].update(
                {es.lstrip('_'): -1.0*getattr(estimator, es)[i] for es in extra_scores}
            )
            r['train_scores'] = {
                'score': r['train_scores'],
            }
            r['train_scores'].update(
                {es.lstrip('_'): -1.0*getattr(estimator, es)[i] for es in extra_scores}
            )

    return results 


def fit_nmf_matrix_custom_init(
    m: np.ndarray,
    n_components: int = 20,
    alpha_W: float = 0.00001,
    alpha_H: float = 0.0375,
    l1_ratio: float = 0.75,
    max_iter: int = 500,
    solver: str = "cd",
    random_state: int = 42, 
    init: Optional[str] = None,
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
    init = (
        "custom"
        if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray)
        else (init if init else ('nndsvd' if solver == 'cd' else 'nndsvda'))
    )
    model = NMF(
        n_components=min(n_components, min(m.shape)),
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        solver=solver,
        init=init,
        random_state=random_state,
        **nmf_kwargs,
    )
    if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray):
        W = model.fit_transform(m, W=W_init, H=H_init)
    else:
        W = model.fit_transform(m)

    if return_model:
        return W, model.components_, model
    return W, model.components_


def pick_parameters_optuna_harmonic_mean(
    bcv: BaseSearchCV,
    parameter_names: List[str],
    constant: float = 0.001,
) -> Dict[str, Any]:
    """ """
    if not all([f'mean_test_{p}' in bcv.cv_results.keys() for p in parameter_names]):
        raise ValueError('All parameter names must be a valid parameter name in cv_results.')
    
    parameters = {
        p: bcv.cv_results[f'mean_test_{p}'] for p in parameter_names
    }
 
    # min-max scale all parameters.
    parameters = {
        p: (v - v.min()) / (v - v.min()).max() for p,v in parameters.items()
    }

    harmonic_mean = len(parameters) / ( 
        1. / (np.vstack(
            list(parameters.values()),
        )+constant)
    ).sum(axis=0)
    
    return harmonic_mean


def tune_hyperparameters_randomizedsearchcv(
    ms: np.ndarray,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    nmf_param_grid: Dict[str, Any] = {},
) -> None: 
    """ """
    rng = np.random.default_rng(seed=random_state)
    cv_errors = []

    for i in range(n_splits):
        pass
