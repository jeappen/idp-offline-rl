"""To calculate transition and reward models used by the algorithms"""
import logging

import numpy as np

from d3rlpy.lcb.util import MDPDataset


class EstimatePR:
    """Estimate the transition probabilities and the reward function."""

    def __init__(self, nb_states, nb_actions, dataset: MDPDataset, T=10, discrete_actions=True, fixed_model=False):
        """Initialize the estimator.

        Args:
            nb_states (int): number of states
            nb_actions (int): number of actions
            dataset (MDPDataset): dataset to use for estimation
            T (int): time horizon
            discrete_actions (bool): whether the actions are discrete
            fixed_model (bool): whether to use the whole dataset for the transition model and reward model
        """
        self._nb_states = nb_states
        self._nb_actions = nb_actions
        self._dataset = dataset
        self.T = T
        self._discrete = discrete_actions
        self.fixed_model = fixed_model

        self.empirical_r_t = None  # empirical reward function
        self.empirical_m_t = None  # empirical state-action counts
        self.empirical_p_t = None  # empirical transition probabilities
        self.empirical_m_t_holdout = None  # holdout empirical state-action counts
        self.logger = logging.getLogger(__name__)

    @property
    def batch_size(self):
        return int(self._dataset.dataset_size / (self.T + 1))

    def estimate(self, holdout=0., shuffle_dataset=True):
        """Estimate the transition probabilities and the reward function.

        Args:
            holdout (float): fraction of the dataset to hold out for validation
            shuffle_dataset (bool): whether to shuffle the dataset before estimation
        """
        assert self._discrete, "Not implemented yet for continuous actions"

        empirical_r_t = np.zeros((self.T + 1, self._nb_states, self._nb_actions))
        empirical_m_t = np.zeros((self.T + 1, self._nb_states, self._nb_actions, self._nb_states))

        # holdout a fraction of the dataset for validation, use the rest for estimation
        empirical_m_t_holdout = np.zeros((self.T + 1, self._nb_states, self._nb_actions, self._nb_states))

        batch_size = self.batch_size
        for t, trans_mb in enumerate(self._dataset.get_dataset_batches(batch_size=batch_size, shuffle=shuffle_dataset)):
            if t > self.T:
                # Last batch might have inconsistent size
                self.logger.warning("Dataset is too large for the given time horizon"
                                    f"\nmb_size{len(trans_mb)} t{t} batch_size{batch_size} T {self.T}"
                                    f"\ndataset_size{self._dataset.dataset_size} vs {self.T * batch_size}")
                break
            if len(trans_mb) != batch_size:
                raise ValueError(f"Batch size {batch_size} is not constant across time {t} when "
                                 f"T={self.T} and trans mb size {len(trans_mb)}")

            # initialize R_sa, use sum to combine (s, a, s') counts to (s, a) for R(s,a) estimation
            empirical_r_t[t] = empirical_r_t[t - 1] * empirical_m_t[t - 1].sum(axis=-1)
            empirical_m_t[t] = empirical_m_t[t - 1]
            empirical_m_t_holdout[t] = empirical_m_t_holdout[t - 1]

            for i, transition in enumerate(trans_mb):
                o, a, rw, o_next, done = transition.observation, transition.action, transition.reward, \
                    transition.next_observation, transition.terminal

                if self._discrete:
                    o = int(o[0])
                    o_next = int(o_next[0])

                empirical_r_t[t, o, a] += rw
                if i < holdout * batch_size:
                    empirical_m_t_holdout[t, o, a, o_next] += 1
                else:
                    empirical_m_t[t, o, a, o_next] += 1
            # Combine (s, a, s') counts to (s, a) for R(s,a) estimation
            empirical_r_t[t] /= empirical_m_t[t].sum(axis=-1).clip(1)

        self.empirical_r_t = empirical_r_t
        self.empirical_m_t = empirical_m_t
        self.empirical_p_t = empirical_m_t / empirical_m_t.sum(axis=-1, keepdims=True).clip(1)
        self.empirical_m_t_holdout = empirical_m_t_holdout

        if self.fixed_model:
            # Use whole dataset for transition model and reward model
            self.empirical_p_t = self.empirical_p_t[-1]
            self.empirical_r_t = self.empirical_r_t[-1]

    def get_batch_mt(self, t):
        """Get the batch of state counts for a given time step t."""
        assert self.empirical_m_t is not None, "Call estimate() first"
        if t > 0:
            return self.empirical_m_t[t] - self.empirical_m_t[t - 1]
        else:
            return self.empirical_m_t[t]

    def get_batch_mt_holdout(self, t):
        """Get the batch of state counts for a given time step t."""
        assert self.empirical_m_t_holdout is not None, "Call estimate() first"
        if t > 0:
            return self.empirical_m_t_holdout[t] - self.empirical_m_t_holdout[t - 1]
        else:
            return self.empirical_m_t_holdout[t]

    def get_batch_pt(self, t, cumulative=False):
        """Get the batch of transition probabilities for a given time step t.

        Args:
            t (int): time step
            cumulative (bool): whether to use the cumulative estimated transition probabilities
        """
        assert self.empirical_p_t is not None, "Call estimate() first"
        if self.fixed_model:
            # Model is fixed for all time steps
            return self.empirical_p_t.copy()
        if cumulative or t == 0:
            # Use the cumulative transition probabilities estimate for the current minibatch
            return self.empirical_p_t[t].copy()
        else:
            # Get the empirical transition probabilities only for the current minibatch
            m_t = self.empirical_m_t[t].sum(axis=-1)
            m_tminus1 = self.empirical_m_t[t - 1].sum(axis=-1)
            return self.get_batch_mt(t) / (m_t - m_tminus1).clip(1).reshape(self._nb_states, self._nb_actions, 1)

    def get_batch_rt(self, t, cumulative=False):
        """Get the batch of rewards for a given time step t.

        Args:
            t (int): time step
            cumulative (bool): whether to use the cumulative estimated rewards
        """
        assert self.empirical_r_t is not None, "Call estimate() first"
        if self.fixed_model:
            # Model is fixed for all time steps
            return self.empirical_r_t.copy()
        if cumulative or t == 0:
            # Use the cumulative reward estimate for the current minibatch
            return self.empirical_r_t[t].copy()
        else:
            # Get the empirical reward only for the current minibatch
            m_t = self.empirical_m_t[t].sum(axis=-1).clip(1)
            m_tminus1 = self.empirical_m_t[t - 1].sum(axis=-1).clip(1)
            sum_of_mb_rw = self.empirical_r_t[t] * m_t - self.empirical_r_t[t - 1] * m_tminus1
            return sum_of_mb_rw / (m_t - m_tminus1)


class OnlineEstimatePR(EstimatePR):
    """Estimate the transition probabilities and the reward function online, no precomputing and more memory efficient."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_mt = None
        self.current_mt_holdout = None
        self.current_pt = None
        self.current_rt = None
        self.prev_mt = None
        self.prev_mt_holdout = None
        self.prev_rt = None
        self.current_t = -1

    def estimate_online(self, t, holdout=0., shuffle_dataset=True):
        assert self.current_t == t - 1, "Not in sync with time step"
        assert self._discrete, "Not implemented yet for continuous actions"

        if self.current_t == -1:
            # Starting from first batch
            self.prev_mt = np.zeros((self._nb_states, self._nb_actions, self._nb_states))
            # holdout a fraction of the dataset for validation, use the rest for estimation
            self.prev_mt_holdout = np.zeros((self._nb_states, self._nb_actions, self._nb_states))
            self.prev_pt = np.zeros((self._nb_states, self._nb_actions, self._nb_states))
            self.prev_rt = np.zeros((self._nb_states, self._nb_actions))
        else:
            self.prev_mt = self.current_mt
            self.prev_mt_holdout = self.current_mt_holdout
            self.prev_pt = self.current_pt
            self.prev_rt = self.current_rt

        batch_size = self.batch_size
        trans_mb = self._dataset.get_dataset_batch(t, batch_size=batch_size, shuffle=shuffle_dataset)
        if t > self.T:
            # Last batch might have inconsistent size
            self.logger.warning("Dataset is too large for the given time horizon"
                                f"\nmb_size{len(trans_mb)} t{t} batch_size{batch_size} T {self.T}"
                                f"\ndataset_size{self._dataset.dataset_size} vs {self.T * batch_size}")
            return

        if len(trans_mb) != batch_size:
            raise ValueError(f"Batch size {batch_size} is not constant across time {t} when "
                             f"T={self.T} and trans mb size {len(trans_mb)}")

        # initialize R_sa, use sum to combine (s, a, s') counts to (s, a) for R(s,a) estimation
        current_rt = self.prev_rt * self.prev_mt.sum(axis=-1)
        current_mt = self.prev_mt.copy()
        current_mt_holdout = self.prev_mt_holdout.copy()

        for i, transition in enumerate(trans_mb):
            o, a, rw, o_next, done = transition.observation, transition.action, transition.reward, \
                transition.next_observation, transition.terminal

            if self._discrete:
                o = int(o[0])
                o_next = int(o_next[0])

            current_rt[o, a] += rw
            if i < holdout * batch_size:
                current_mt_holdout[o, a, o_next] += 1
            else:
                current_mt[o, a, o_next] += 1
        # Combine (s, a, s') counts to (s, a) for R(s,a) estimation
        current_rt /= current_mt.sum(axis=-1).clip(1)

        self.current_t = t
        self.current_rt = current_rt
        self.current_mt = current_mt
        if not (self.fixed_model and self.current_t < self.T):
            # Only update probability if not fixed model or if fixed model and at the last time step
            self.current_pt = current_mt / current_mt.sum(axis=-1, keepdims=True).clip(1)
        self.current_mt_holdout = current_mt_holdout

    def estimate(self, holdout=0., shuffle_dataset=True):
        """If fixed model mode"""
        if self.fixed_model:
            # Use whole dataset for transition model and reward model
            for t in range(self.T + 1):
                self.estimate_online(t, holdout=holdout, shuffle_dataset=shuffle_dataset)
            pass
            self.current_t = -1  # Reset time step so that get_batch_* functions work
        else:
            self.estimate_online(0)
            print("Only doing t=0 estimation since not Fixed model, use estimate_online instead")

    def get_batch_mt(self, t):
        """Get the batch of state counts for a given time step t."""
        assert not (self.fixed_model and self.current_mt is None), "Call estimate() first for fixed model"
        if self.current_t == t - 1:
            self.estimate_online(t)
        if self.current_t != t and not self.fixed_model:
            raise ValueError("Not in sync with time step")
        return self.current_mt - self.prev_mt

    def get_batch_mt_holdout(self, t):
        """Get the batch of state counts for a given time step t."""
        assert not (self.fixed_model and self.current_mt_holdout is None), "Call estimate() first for fixed model"
        if self.current_t == t - 1:
            self.estimate_online(t)
        if self.current_t != t and not self.fixed_model:
            raise ValueError("Not in sync with time step")
        return self.current_mt_holdout - self.prev_mt_holdout

    def get_batch_pt(self, t, cumulative=False):
        """Get the batch of transition probabilities for a given time step t.

                Args:
                    t (int): time step
                    cumulative (bool): whether to use the cumulative estimated transition probabilities
                """
        assert not (self.fixed_model and self.current_pt is None), "Call estimate() first for fixed model"
        if self.current_t == t - 1:
            self.estimate_online(t)
        if self.current_t != t and not self.fixed_model:
            raise ValueError("Not in sync with time step")
        if self.fixed_model or cumulative or t == 0:
            # Use the cumulative transition probabilities estimate for the current minibatch
            return self.current_pt.copy()
        else:
            # Get the empirical transition probabilities only for the current minibatch
            m_t = self.current_mt.sum(axis=-1)
            m_tminus1 = self.prev_mt.sum(axis=-1)
            sum_of_mb_p = (self.current_mt - self.prev_mt)
            return sum_of_mb_p / (m_t - m_tminus1).clip(1).reshape(self._nb_states, self._nb_actions, 1)

    def get_batch_rt(self, t, cumulative=False):
        """Get the batch of rewards for a given time step t.

        Args:
            t (int): time step
            cumulative (bool): whether to use the cumulative estimated rewards
        """
        assert not (self.fixed_model and self.current_rt is None), "Call estimate() first for fixed model"
        if self.current_t == t - 1:
            self.estimate_online(t)
        if self.current_t != t and not self.fixed_model:
            raise ValueError("Not in sync with time step")
        if self.fixed_model or cumulative or t == 0:
            # Use the cumulative reward estimate for the current minibatch
            return self.current_rt.copy()
        else:
            # Get the empirical reward only for the current minibatch
            m_t = self.current_mt.sum(axis=-1).clip(1)
            m_tminus1 = self.prev_mt.sum(axis=-1).clip(1)
            sum_of_mb_rw = self.current_rt * m_t - self.prev_rt * m_tminus1
            return sum_of_mb_rw / (m_t - m_tminus1)
