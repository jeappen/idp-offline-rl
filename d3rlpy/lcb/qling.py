"""Q-learning baselines."""

import logging

import numpy as np

from d3rlpy.envs import MDPToolboxEnv
from d3rlpy.lcb.estimators import OnlineEstimatePR
from d3rlpy.lcb.util import evaluate_policy_on_env, SAVE_PENALTY_FREQ


class Qlearning:
    """Qlearning algorithm."""

    def __init__(self, env, dataset, eval_env=None, alpha_b=1., beta_b=0.,
                 estimator=None, holdout=0., **kwargs):
        """Initialize the qling algorithm.

        Args:
            alpha_b: Alpha parameter for scaling the penalty (Linear scaling)
            beta_b: Beta parameter for scaling the penalty (Linear scaling)
        """
        self.log_stdelta = None
        self.n_sa = None
        self.info = None
        self.estimatePR = None
        self._env = env
        self._dataset = dataset
        self._sampling_dataset = dataset  # Dataset to sample from for Qlearning update
        self.holdout = holdout
        # Choose the estimator function for the transition probabilities and the reward function
        if estimator is None:
            self.estimator = OnlineEstimatePR
        else:
            self.estimator = estimator
        self._eval_env = eval_env if eval_env is not None else env
        self._init_Pmat()
        self.logger = logging.getLogger(__name__)
        self._model_free = True  # To save memory, only store the model if needed
        self._update_q_with_v = False  # Whether to update the Q-function with the V-function
        self.alpha_b = alpha_b
        self.beta_b = beta_b
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

    def _calc_penalty_qling(self, t, H=None, learning_tuple=None, **kwargs):
        """Calculate the penalty term for all s,a as a matrix."""
        # Keeping penalty computation light since it is done for each sampled (s,a) pair
        # Update the penalty term here, now fixed for each minibatch
        return self._penalty

    def _scale_penalty(self, penalty):
        """Scale the penalty to be within the range of the model"""
        return self.alpha_b * penalty + self.beta_b

    # Calculate Q
    def update_q_pi(self, learning_tuple, Q_pi, eta, gamma, penalty, V_pi=None, t=-1, cumulative_model=True):
        """Given a action-value function, calculate the updated Q-function.

        Args:
            learning_tuple: tuple of (s,a,r,s') from the sampling dataset to update the Q-function
            Q_pi: action-value function
            eta: learning rate
            gamma: discount factor
            penalty: penalty term for each state-action pair
            t: time step
            cumulative_model: whether to use the cumulative model for the transition probabilities

        Returns:
            Q_pi: Q-function for the given state-value function V_pi
        """
        assert self.estimatePR is not None, "Call estimate() first"
        s, a, r, s_dash, done = learning_tuple
        if V_pi is not None:
            v_s_dash = V_pi[s_dash]
        else:
            v_s_dash = np.max(Q_pi[s_dash])
        Q_pi[s, a] = (1 - eta) * Q_pi[s, a] + eta * (r + gamma * v_s_dash - penalty[s, a])
        return Q_pi

    def sample_sarsa_tuple(self, sample_index=None):
        """Sample a tuple of (s,a,r,s') from the sampling dataset"""
        transition = self._sampling_dataset.sample(sample_range=None)
        return int(transition.observation[0]), transition.action, transition.reward, \
            int(transition.next_observation[0]), transition.terminal

    def _update_sampling_dataset(self, t):
        """Update the sampling dataset for the Q-learning update"""
        # self._sampling_dataset = self._dataset
        pass

    def calc_v_pi(self, q_pi, v_pi=None):
        """Given a Q-function, calculate the state-value function V_pi."""
        if v_pi is not None:
            return np.maximum(np.max(q_pi, axis=-1), v_pi)
        else:
            return np.max(q_pi, axis=-1)

    def calc_policy(self, q_pi):
        """Given a Q-function, calculate the policy."""
        return np.argmax(q_pi, axis=-1)

    def _dataset_statistics_q(self, **kwargs):
        """To calculate any statistics about the dataset before running the algorithm."""
        pass

    def _init_penalty(self, t, **kwargs):
        """To initialize the penalty term before running the algorithm."""
        self._penalty = np.zeros((self.nb_states, self.nb_actions))
        return self._penalty

    def log_policy_perf(self, curr_t, policy_to_test, policy_perf_store):
        """Helper function to log the performance of the policy."""

        def off_policy_fn(obs):
            return policy_to_test[obs]

        off_vi_stat = evaluate_policy_on_env(off_policy_fn, self._eval_env, gymnasium=True, num_episodes=1000)
        policy_perf_store.append([curr_t, off_vi_stat['mean'], off_vi_stat['std']])

    def _update_statistics(self, learning_tuple, t):
        """Update any statistics after each update"""
        if t == 0:
            self.n_sa = np.zeros((self.nb_states, self.nb_actions))
        else:
            s, a, _, _, _ = learning_tuple
            self.n_sa[s, a] += 1

    def update_eta(self, t, H=1, learning_tuple=None, eta=None):
        if eta is None:
            # Adjust learning rate based on QL-LCB
            s, a, r, s_dash, done = learning_tuple
            n = self.n_sa[s, a]  # Number of times (s,a) pair has been visited
            return (H + 1) / (H + n)
        else:
            return eta

    def run(self, eta=0.5, T=None, gamma=0.99, N=100, delta=0.01, known_P=True, V_max=None,
            shuffle_dataset=True, final_eval=False, plot_perf_interval=None,
            store_intermediate_policy=True, estimator_kwargs=None, samples_per_iteration=None,
            update_q_with_v=False, estimator_T=None):
        """Perform Q-Learning.

        Args:
            eta (float): learning rate
            T (int): number of iterations
            gamma (float): discount factor
            N (int): number of data points
            delta (float): confidence parameter
            known_P (bool): whether the transition probabilities are known
            shuffle_dataset (bool): whether to shuffle the dataset before training
            final_eval (bool): whether to evaluate the policy at the end of the algorithm
            plot_perf_interval (int): number of steps for logging the performance of the policy, if None, no plotting
            store_intermediate_policy (bool): whether to store the intermediate policies (to redraw plots)
            estimator_kwargs (dict): additional arguments for the estimator
            samples_per_iteration (int): number of samples per iteration
            update_q_with_v (bool): whether to update Q with V_pi
            estimator_T (int): number of batches for the estimator
        """
        if T is None:
            # TODO: Set T appropriately
            T = int(np.log(N) / (1 - gamma))
        if estimator_T is None:
            estimator_T = T
        estimator_T = min(estimator_T, N - 1)
        self.log_stdelta = np.log(self.nb_states * T / delta) / ((1 - gamma) ** 2)
        H = np.ceil((4 * (1 - gamma)) * self.log_stdelta)  # since (1 - gamma)**2 in denominator of self.log_stdelta
        # L = Lc * np.log(2 * (T + 1) * self.nb_states * self.nb_actions / delta)
        # if V_max is None:
        #     V_max = 1 / (1 - gamma)
        if estimator_kwargs is None:
            estimator_kwargs = {}
        print("Fixing model for Q-learning")
        estimator_kwargs['fixed_model'] = estimator_kwargs.get('fixed_model', True)

        # R_sa = np.zeros((self.nb_states, self.nb_actions))
        Q_sa = np.zeros((self.nb_states, self.nb_actions))
        V_s = np.zeros(self.nb_states) if self._update_q_with_v or update_q_with_v else None
        self.n_sa = np.zeros((self.nb_states, self.nb_actions))
        self.info = {}

        self.estimatePR = self.estimator(self.nb_states, self.nb_actions, self._dataset, T=estimator_T,
                                         **estimator_kwargs)
        if not self._model_free:
            self.estimatePR.estimate(holdout=self.holdout, shuffle_dataset=shuffle_dataset)
        self._dataset_statistics_q(known_p=known_P)
        self._update_sampling_dataset(t=0)

        # Set initial policy to random
        policy = np.random.randint(self.nb_actions, size=self.nb_states)
        samples_per_iteration = self.estimatePR.batch_size if samples_per_iteration is None else samples_per_iteration

        saved_penalty = {}
        saved_visitation = {}
        policy_perf = []
        intermediate_policy = []

        # initial policy performance
        if plot_perf_interval is not None:
            self.log_policy_perf(0, policy, policy_perf)
            if store_intermediate_policy:
                intermediate_policy.append((0, policy.copy()))

        self._update_statistics(None, t=0)  # Init statistics for t=0
        for t in range(1, T + 1):
            penalty = self._init_penalty(t, known_p=known_P)
            self._update_sampling_dataset(t=t)
            for mb_i in range(samples_per_iteration):
                # sample_range reflects the index of the sample in the
                learning_tuple = self.sample_sarsa_tuple(sample_index=t * self.estimatePR.batch_size)
                self._update_statistics(learning_tuple, t=t)  # To keep track of count
                # Update penalty (if needed)
                penalty = self._calc_penalty_qling(t, H=H, learning_tuple=learning_tuple, known_p=known_P)
                # Compute Q using the current policy and V_s
                q_update_eta = self.update_eta(t, H=H, learning_tuple=learning_tuple, eta=eta)  # Update learning rate
                Q_sa = self.update_q_pi(learning_tuple, Q_sa, q_update_eta, gamma, penalty, V_pi=V_s, t=t)
                # Compute policy
                policy = self.calc_policy(Q_sa)
                if V_s is not None:
                    V_s = self.calc_v_pi(Q_sa, V_s)

            if not self._model_free and estimator_T == T and t % SAVE_PENALTY_FREQ == 0:
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
        info = {'T': T, 'Q_sa': Q_sa, 'eta': eta, estimator_T: estimator_T,
                'samples_per_iteration': samples_per_iteration,
                'dataset_size': self._dataset.dataset_size,
                'final_penalty': penalty, 'saved_penalty': saved_penalty,
                'saved_visitation': saved_visitation, 'policy_perf': policy_perf,
                'intermediate_policy': intermediate_policy, 'final_n_sa': self.n_sa,
                'gamma': gamma, 'holdout': self.holdout, 'shuffle_dataset': shuffle_dataset}
        info.update(self.additional_info)
        self.info.update(info)

        return policy, Q_sa, self.info


class QlearningLCB(Qlearning):
    """Qlearning algorithm with LCB."""

    def __init__(self, *args, cb=10, **kwargs):
        """Initialize the model for DSD calculation.

        Args:
            cb: Confidence parameter for LCB
        """
        super().__init__(*args, **kwargs)
        self.cb = cb
        self.additional_info.update({'cb': cb})
        self._update_q_with_v = True

    def _calc_penalty_qling(self, t, H=1, learning_tuple=None, **kwargs):
        """Calculate the penalty term for all s,a as a matrix."""
        # Adjust learning rate based on QL-LCB
        s, a, r, s_dash, done = learning_tuple
        n = self.n_sa[s, a]  # Number of times (s,a) pair has been visited
        self._penalty[s, a] = self.cb * np.sqrt((H / n) * self.log_stdelta)
        return self._scale_penalty(self._penalty)

    # def _calc_penalty_qling(self, t, learning_tuple=None, **kwargs):
    #     """Calculate the penalty term for all s,a as a matrix using DSD."""
    #     return self._calc_penalty(t, known_p=kwargs.get('known_p', False))

    # def _init_penalty(self, t, **kwargs):
    #     """To initialize the penalty term before running the algorithm."""
    #     # Keeping the penalty fixed throughout a minibatch, uses the DSD penalty function
    #     self._penalty = self._calc_penalty(t, known_p=kwargs.get('known_p', False))
    #     return self._penalty
