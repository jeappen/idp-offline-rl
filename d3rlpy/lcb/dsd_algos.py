"""Any DSD based algorithms"""
import logging

import numpy as np

from .ksd.kernels import exp_hamming_kernel
from .ksd.ksd import KSD
from .ksd.ksd_utils import ksd_est_biased, ksd_est
from .ksd.model import TabularModel
from .offvi import OfflineVI
from .qling import Qlearning
from .util import LARGE_CONST, SMALL_CONST


class OfflineVIwDSD(OfflineVI):
    """Offline Value Iteration with Discrete Stein Discrepancy."""

    def __init__(self, *args, kernel_fun=None,
                 gamma_b=1.0, biased_estimator=True, sqrt_penalty=False,
                 **kwargs):
        """Initialize the model for DSD calculation.

        Args:
            kernel_fun: Kernel function for the DSD
            gamma_b: Gamma parameter for scaling the DSD (Clip value)
                    if None, set to dataset max,
                    if < 0 set to 0 for unvisited states
            biased_estimator: Whether to use the biased estimator for the DSD
            gamma_dataset: Whether to set the gamma parameter from the dataset
            sample_data_for_gamma: Whether to sample data for the gamma parameter
                        (only if gamma_dataset is True and large dataset)
            sqrt_penalty: Whether to take the square root of the penalty
        """
        super().__init__(*args, **kwargs)
        # self.logger.setLevel(logging.INFO)
        self.logger.info("Initializing OfflineVIwDSD")
        self.penalty_range = 1e3
        self.kernel_fun = exp_hamming_kernel if kernel_fun is None else kernel_fun
        self.gamma_b = gamma_b
        self.biased_estimator = biased_estimator
        self.sqrt_penalty = sqrt_penalty
        if self.gamma_b is None:
            # raise NotImplementedError, "Need efficient way to calculate penalty"
            logging.info("Trying to calculate gamma from the dataset")
        elif self.gamma_b < 0:
            self.logger.info("Setting gamma to 0 for unvisited states")
        self.additional_info.update({'gamma_b': self.gamma_b, 'biased_estimator': self.biased_estimator,
                                     'kernel_fun': self.kernel_fun.__name__})

    def _init_dsd_model(self, p_mat=None):
        """Initialize the model used for the DSD calculation"""
        if p_mat is None:
            # Note: Assumes the transition probabilities are known from the environment
            print("Using known transition probabilities in penalty calculation")
            self.model = TabularModel(env=self._env)
        else:
            self.model = TabularModel(env=self._env, p_mat=p_mat)
        self.ksd = KSD(neg_fun=self.model.neg, score_fun=self.model.score,
                       kernel_fun=self.kernel_fun, neg_inv_fun=self.model.neg_inv)  # Use null model

    def _calc_ksd(self, samples, sa=None, partial_kappa=False):
        """Calculate the KSD for the given samples"""
        kappa_vals = self.ksd.compute_kappa(samples=samples, sa=sa)

        # Compute U-statistics and bootstrap intervals
        if self.biased_estimator:
            ksd_stats = ksd_est_biased([kappa_vals])
        else:
            ksd_stats, _ = ksd_est([kappa_vals])

        # ksd_boot_list, ksd_thres_list = ksd_boot([kappa_vals])
        # ksd_pvals = ksd_pvalue(ksd_boot_list, ksd_stats)

        ksd_stat = ksd_stats[0]
        # ksd_thres = ksd_thres_list[0]
        # ksd_pval = ksd_pvals[0]
        # ksd_pred = 1 * (ksd_stat > ksd_thres)  # 0 for p, 1 for q
        # Logging info
        if ksd_stat + SMALL_CONST < 0:
            self.info['ksd_min'] = min(self.info.get('ksd_min', 0), ksd_stat)
            self.logger.warning(f"KSD is negative {ksd_stat}, {samples}, {sa},"
                                f"kappa {kappa_vals}, samples {samples.shape}, "
                                f"sa {sa.shape if sa is not None else None}")
            self.info['ksd_neg'] = self.info.get('ksd_neg', 0) + 1
        else:
            self.info['ksd_max'] = max(self.info.get('ksd_max', 0), ksd_stat)
            self.info['ksd_non_neg'] = self.info.get('ksd_non_neg', 0) + 1
            if ksd_stat > LARGE_CONST:
                self.info['ksd_large'] = self.info.get('ksd_large', 0) + 1
                self.logger.warning(f"KSD is unbounded {ksd_stat}, {samples}, {sa},"
                                    f"kappa {kappa_vals}, samples {samples.shape}, "
                                    f"sa {sa.shape if sa is not None else None}")
        if self.sqrt_penalty:
            ksd_stat = np.sqrt(ksd_stat)
        return ksd_stat

    def _dataset_statistics(self, known_p=False):
        """Now we have the entire dataset, calculate the gamma (DSD clip value)"""
        self._init_gamma(known_p=known_p)
        if known_p:
            # Use the known transition probabilities
            self.logger.info("Using known transition probabilities in penalty calculation")
            self._init_dsd_model()

    def _init_gamma(self, known_p=False):
        """Initialize the gamma parameter from the dataset only once
            using fact self.gamma_b is None initially"""
        if self.gamma_b is None:
            # Get entire dataset statistics
            # Reset estimator for ksd computation
            self.estimatePR.current_t = -1
            fm_mode = self.estimatePR.fixed_model
            if self._dataset.dataset_size > 1:
                # Use minibatch approximation of DSD max to prevent storing the entire thing in memory
                self.logger.info("Large dataset, not calculating gamma all at once.")
                max_penalty = 0
                for t in range(self.estimatePR.T):
                    max_penalty = max(max_penalty, self._calc_penalty(t, known_p=known_p).max())
            else:
                # Use the entire dataset in-memory to calculate gamma
                self.estimatePR.fixed_model = True
                self.estimatePR.estimate()
                max_penalty = self._calc_penalty(-1).max()
            # Reset the estimator
            self.estimatePR.current_t = -1
            self.estimatePR.fixed_model = fm_mode
            self.estimatePR.estimate()
            self.gamma_b = max_penalty
            self.logger.info(f"Initialized gamma_b as {self.gamma_b} from dataset")
        elif self.gamma_b < 0:
            self.logger.info("Setting gamma to 0 for unvisited states")
            self.gamma_b = 0

    def _calc_penalty(self, t, known_p=False, *args, **kwargs):
        """Calculate the penalty term for the given time step for each state-action pair"""
        if not known_p:
            # Use the estimated transition probabilities, todo: add holdout options here
            # Maybe cumulative uses t-1 model for t penalty
            self._init_dsd_model(p_mat=self.estimatePR.get_batch_pt(t, cumulative=True))

        if self.holdout:
            # Use the holdout set to estimate the penalty
            ind = self.estimatePR.get_batch_mt_holdout(t).astype(int)
        else:
            ind = self.estimatePR.get_batch_mt(t).astype(int)
        s, a, s_next = np.nonzero(ind)
        # Group all s_next for each (s,a) pair
        sa_grouping = {}
        for sas in zip(s, a, s_next):
            sa_tuple = sas[:2]
            _s_next = sas[-1]
            if sa_tuple in sa_grouping:
                sa_grouping[sa_tuple].append(_s_next)
            else:
                sa_grouping[sa_tuple] = [_s_next]

        # Now calculate penalty terms
        if self.gamma_b is not None and self.gamma_b > 0:
            # Will set a default penalty for unvisited (s,a) pairs
            penalty = np.ones((self.nb_states, self.nb_actions)) * LARGE_CONST
        else:
            # gamma_b = 0
            # No penalty for unvisited (s,a) pairs
            penalty = np.zeros((self.nb_states, self.nb_actions))

        # Calculate DSD for each (s,a) pair in mini-batch
        for (s, a), s_next in sa_grouping.items():
            # Tile s_next to match the number of times it appears in the dataset
            _s_tiled = []
            # If only one s_next, tile it at least twice to avoid single samples when using unbiased estimator
            clip_count = 2 if not self.biased_estimator and len(s_next) == 1 else 1
            for _s_next in s_next:
                sas_count = ind[(s, a, _s_next)].clip(clip_count)  # NOTE: clip when using unbiased estimator
                _s_tiled.append(np.tile(_s_next, (sas_count, 1)))
            _s_tiled = np.vstack(_s_tiled)
            ksd_stat = self._calc_ksd(_s_tiled, sa=np.tile((s, a), (_s_tiled.shape[0], 1)))
            penalty[(s, a)] = np.clip(self._scale_penalty(ksd_stat), 0, self.penalty_range)
        if self.gamma_b is not None and self.gamma_b > 0:
            # Use gamma_b to clip penalty for unvisited (s,a) pairs
            penalty = np.minimum(penalty, self.gamma_b)

        return penalty


class QlearningwDSD(Qlearning, OfflineVIwDSD):
    """Qlearning algorithm with Discrete Stein Discrepancy."""

    def __init__(self, *args, kernel_fun=None,
                 gamma_b=1.0, alpha_b=1.0, beta_b=0.0,
                 biased_estimator=True, sqrt_penalty=False,
                 alpha_schedule=None,
                 **kwargs):
        """Initialize the model for DSD calculation.

        Args:
            kernel_fun: Kernel function for the DSD
            gamma_b: Gamma parameter for scaling the DSD (Clip value)
                    if None, set to dataset max,
                    if < 0 set to 0 for unvisited states
            alpha_b: Alpha parameter for scaling the DSD (Linear scaling)
            beta_b: Beta parameter for scaling the DSD (Linear scaling)
            biased_estimator: Whether to use the biased estimator for the DSD
            gamma_dataset: Whether to set the gamma parameter from the dataset
            sample_data_for_gamma: Whether to sample data for the gamma parameter
                        (only if gamma_dataset is True and large dataset)
            sqrt_penalty: Whether to take the square root of the penalty
            alpha_schedule: Schedule for alpha_b ('exp', gamma, start_alpha) or ('lin', iter_to_stop)
        """
        super().__init__(*args, **kwargs)
        # self.logger.setLevel(logging.INFO)
        self.logger.info("Initializing QlearningwDSD")
        self.penalty_range = 1e3
        self.kernel_fun = exp_hamming_kernel if kernel_fun is None else kernel_fun
        self.gamma_b = gamma_b
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        self.biased_estimator = biased_estimator
        self.sqrt_penalty = sqrt_penalty
        if self.gamma_b is None:
            # raise NotImplementedError, "Need efficient way to calculate penalty"
            logging.info("Trying to calculate gamma from the dataset")
        elif self.gamma_b < 0:
            self.logger.info("Setting gamma to 0 for unvisited states")
        self.additional_info.update({'gamma_b': self.gamma_b, 'alpha_b': self.alpha_b, 'beta_b': self.beta_b,
                                     'biased_estimator': self.biased_estimator, 'kernel_fun': self.kernel_fun.__name__,
                                     'sqrt_penalty': self.sqrt_penalty})
        self._model_free = False  # Need to store the model for the DSD calculation
        self._alpha_schedule = alpha_schedule

    def _alpha_schedule_fn(self, t):
        """Return the alpha value for the given iteration"""
        if self._alpha_schedule is None:
            return self.alpha_b
        elif self._alpha_schedule[0] == 'exp':
            # Exponential schedule
            return min(self._alpha_schedule[2] * (self._alpha_schedule[1] ** min(t, 100)), self.alpha_b)
        elif self._alpha_schedule[0] == 'lin':
            # Linear schedule
            return min((self.alpha_b / self._alpha_schedule[1]) * t, self.alpha_b)
        else:
            raise NotImplementedError(f"Unknown alpha schedule {self._alpha_schedule}")

    # def _calc_penalty_qling(self, t, learning_tuple=None, **kwargs):
    #     """Calculate the penalty term for all s,a as a matrix using DSD."""
    #     return self._calc_penalty(t, known_p=kwargs.get('known_p', False))
    def _dataset_statistics_q(self, **kwargs):
        return self._dataset_statistics(known_p=kwargs.get('known_p', False))

    # def _init_penalty(self, t, **kwargs):
    #     """To initialize the penalty term before running the algorithm."""
    #     # Keeping the penalty fixed throughout a minibatch, uses the DSD penalty function
    #     self._penalty = self._calc_penalty(t, known_p=kwargs.get('known_p', False))
    #     return self._penalty

    # def sample_sarsa_tuple(self, sample_index=None):
    #     """Sample a tuple of (s,a,r,s') from the sampling dataset"""
    #     # Only sample from dataset used to compute penalty (the current minibatch)
    #     # TODO: Check this range
    #     # sample_range = (sample_index - self.estimatePR.batch_size, sample_index)
    #     transition = self._sampling_dataset.sample(sample_range=None)
    #     return int(transition.observation[0]), transition.action, transition.reward, \
    #         int(transition.next_observation[0]), transition.terminal

    def _update_sampling_dataset(self, t):
        """Update the sampling dataset for the Q-learning update"""
        # Only sample from dataset used to compute penalty
        # TODO: Do any SPMCMC sampling here to reduce the dataset
        self._sampling_dataset = self.estimatePR._dataset

    def _calc_penalty_qling(self, t, H=1, learning_tuple=None, **kwargs):
        """Calculate the penalty term for all s,a as a matrix."""
        if not kwargs.get('known_p', False):
            # Use the estimated transition probabilities, todo: add holdout options here
            # Maybe cumulative uses t-1 model for t penalty
            self._init_dsd_model(p_mat=self.estimatePR.get_batch_pt(t, cumulative=True))
        # Adjust learning rate based on QL-LCB
        s, a, r, s_dash, done = learning_tuple
        n = self.n_sa[s, a]  # Number of times (s,a) pair has been visited
        alpha_b = self._alpha_schedule_fn(t)
        # TODO: See if single value KSD makes sense
        # TODO: Try a alpha schedule from low to high to prevent using penalty too early
        ksd_val = self._calc_ksd(np.array([[s_dash]]), sa=np.array([[s, a]]))
        self._penalty[s, a] = max(self._penalty[s, a] * np.sqrt((n - 1) / n), alpha_b * ksd_val + self.beta_b)
        return self._penalty
