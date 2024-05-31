"""Offline Value Iteration baselines."""

import logging

import numpy as np

from d3rlpy.envs import MDPToolboxEnv
from d3rlpy.lcb.estimators import EstimatePR
from d3rlpy.lcb.util import evaluate_policy_on_env, SAVE_PENALTY_FREQ


class OfflineVI:
    """Offline VI algorithm."""

    def __init__(self, env, dataset, eval_env=None, alpha_b=1., beta_b=0.,
                 estimator=None, holdout=0., **kwargs):
        """Initialize the offline VI algorithm.

        Args:
            alpha_b: Alpha parameter for scaling the penalty (Linear scaling)
            beta_b: Beta parameter for scaling the penalty (Linear scaling)
        """
        self.info = None
        self.estimatePR = None
        self._env = env
        self._dataset = dataset
        self.holdout = holdout
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        # Choose the estimator function for the transition probabilities and the reward function
        if estimator is None:
            self.estimator = EstimatePR
        else:
            self.estimator = estimator
        self._eval_env = eval_env if eval_env is not None else env
        self._init_Pmat()
        self.logger = logging.getLogger(__name__)
        self.additional_info = {'alpha_b': self.alpha_b,
                                'beta_b': self.beta_b}  # Additional info to be logged such as special values

    def _init_Pmat(self):
        """Init a Probability transition matrix if not estimating the transition probabilities.
            Useful to vectorize the computation.
        """
        if isinstance(self._env, MDPToolboxEnv):
            P = self._env.env.P
        else:
            P = self._env.P
        self.P_mat = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                for p, s_next, r, _ in P[s][a]:
                    self.P_mat[s, a, s_next] += p

    @property
    def nb_states(self):
        # number of possible states
        return self._env.observation_space.n

    @property
    def nb_actions(self):
        # number of actions from each state
        return self._env.action_space.n

    def _calc_penalty(self, t, v_max, L, **kwargs):
        """Calculate the penalty term for all s,a as a matrix."""
        # Sum m_t over all next states (s,a,s') to get m_t(s,a)
        penalty_val = v_max * np.sqrt(L / self.estimatePR.get_batch_mt(t).sum(axis=-1).clip(1))
        return self._scale_penalty(penalty_val)

    def _scale_penalty(self, penalty):
        """Scale the penalty to be within the range of the model"""
        return self.alpha_b * penalty + self.beta_b

    # Calculate Q
    def calc_q_pi(self, V_pi, model, gamma, penalty, t=-1, use_estimated_p=False, cumulative_model=True):
        """Given a state-value function V_pi and reward estimates, calculate the Q-function for a given model.

        Args:
            V_pi: state-value function
            model: model of the environment
            gamma: discount factor
            penalty: penalty term for each state-action pair
            t: time step
            use_estimated_p: whether to use the estimated transition probabilities
            cumulative_model: whether to use the cumulative model for the transition probabilities

        Returns:
            Q_pi: Q-function for the given state-value function V_pi
        """
        assert self.estimatePR is not None, "Call estimate() first"
        Q_pi = self.estimatePR.get_batch_rt(t, cumulative=cumulative_model)
        Q_pi -= penalty
        if use_estimated_p:
            Q_pi += gamma * np.matmul(self.estimatePR.get_batch_pt(t, cumulative=cumulative_model), V_pi)
        else:
            # use the true transition probabilities
            Q_pi += gamma * np.matmul(self.P_mat, V_pi)
        return Q_pi

    def calc_v_pi(self, q_pi):
        """Given a Q-function, calculate the state-value function V_pi."""
        return np.max(q_pi, axis=-1)

    def calc_policy(self, q_pi):
        """Given a Q-function, calculate the policy."""
        return np.argmax(q_pi, axis=-1)

    def _dataset_statistics(self, **kwargs):
        """To calculate any statistics about the dataset before running the algorithm."""
        pass

    def log_policy_perf(self, curr_t, policy_to_test, policy_perf_store):
        """Helper function to log the performance of the policy."""

        def off_policy_fn(obs):
            return policy_to_test[obs]

        off_vi_stat = evaluate_policy_on_env(off_policy_fn, self._eval_env, gymnasium=True, num_episodes=1000)
        policy_perf_store.append([curr_t, off_vi_stat['mean'], off_vi_stat['std']])

    def offline_value_iteration(self, gamma=0.99, N=100, delta=0.01, known_P=True, Lc=2000, V_max=None,
                                shuffle_dataset=True, final_eval=False, plot_perf_interval=None,
                                store_intermediate_policy=True, estimator_kwargs=None):
        """Perform offline value iteration.

        Args:
            gamma (float): discount factor
            N (int): number of data points
            delta (float): confidence parameter
            known_P (bool): whether the transition probabilities are known
            Lc (int): Constant for the penalty term (meant to be large for
            V_max (float): maximum value of the state-value function
            shuffle_dataset (bool): whether to shuffle the dataset before training
            final_eval (bool): whether to evaluate the policy at the end of the algorithm
            plot_perf_interval (int): number of steps for logging the performance of the policy, if None, no plotting
            store_intermediate_policy (bool): whether to store the intermediate policies (to redraw plots)
            estimator_kwargs (dict): additional arguments for the estimator
        """
        T = int(np.log(N) / (1 - gamma))
        L = Lc * np.log(2 * (T + 1) * self.nb_states * self.nb_actions / delta)
        if V_max is None:
            V_max = 1 / (1 - gamma)
        if estimator_kwargs is None:
            estimator_kwargs = {}

        # R_sa = np.zeros((self.nb_states, self.nb_actions))
        Q_sa = np.zeros((self.nb_states, self.nb_actions))
        V_s = np.zeros(self.nb_states)
        self.info = {}

        self.estimatePR = self.estimator(self.nb_states, self.nb_actions, self._dataset, T=T, **estimator_kwargs)
        self.estimatePR.estimate(holdout=self.holdout, shuffle_dataset=shuffle_dataset)
        self._dataset_statistics(known_p=known_P)

        # Set initial policy to random (TODO: check if it is better to set deterministic)
        policy = np.random.randint(self.nb_actions, size=self.nb_states)

        saved_penalty = {}
        saved_visitation = {}
        policy_perf = []
        intermediate_policy = []

        # initial policy performance
        if plot_perf_interval is not None:
            self.log_policy_perf(0, policy, policy_perf)
            if store_intermediate_policy:
                intermediate_policy.append((0, policy.copy()))

        for t in range(1, T + 1):
            penalty = self._calc_penalty(t, v_max=V_max, L=L, known_p=known_P)
            # Compute Q using the current policy and V_s
            model = self._env.env.P if known_P else None
            Q_sa = self.calc_q_pi(V_s, model, gamma, penalty, t=t, use_estimated_p=not known_P)
            # Compute V_mid and check if it is better
            V_mid = self.calc_v_pi(Q_sa)
            pi_mid = self.calc_policy(Q_sa)
            # Replace V_s and policy if V_mid is better
            # Weirdly, may work better in many states when sign swapped. Possible imitation learning like effect?
            replace_ind = V_mid > V_s
            policy[replace_ind] = pi_mid[replace_ind]
            V_s[replace_ind] = V_mid[replace_ind]
            if t % SAVE_PENALTY_FREQ == 0:
                saved_penalty[t] = penalty.copy()
                # Also save visitation counts
                saved_visitation[t] = self.estimatePR.get_batch_mt(t).copy()
                self.logger.debug(f"saved visitation at t={t} {saved_visitation[t].sum()}")

            if plot_perf_interval is not None and t % (T // plot_perf_interval) == 0:
                self.log_policy_perf(t, policy, policy_perf)
                if store_intermediate_policy:
                    intermediate_policy.append((t, policy.copy()))

        if final_eval or plot_perf_interval is not None:
            self.log_policy_perf(T, policy, policy_perf)
        # Log info for the optimization
        info = {'T': T, 'L': L, 'V_max': V_max, 'Q_sa': Q_sa,
                'dataset_size': self._dataset.dataset_size,
                'final_penalty': penalty, 'saved_penalty': saved_penalty,
                'saved_visitation': saved_visitation, 'policy_perf': policy_perf,
                'intermediate_policy': intermediate_policy,
                'known_P': known_P, 'holdout': self.holdout, 'shuffle_dataset': shuffle_dataset,
                'estimator_kwargs': estimator_kwargs, 'gamma': gamma, 'delta': delta}
        info.update(self.additional_info)
        self.info.update(info)

        return policy, V_s, self.info


class OfflineVIwNoPenalty(OfflineVI):
    """Offline Value Iteration without penalty."""

    def _calc_penalty(self, t, v_max, L, **kwargs):
        return np.zeros((self.nb_states, self.nb_actions))
