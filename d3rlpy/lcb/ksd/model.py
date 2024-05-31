"""To hold models used by the Stein Discrepancy calculations"""

from copy import deepcopy

import numpy as np

EPS_CONST = 1e-6

from d3rlpy.envs.env import MDPToolboxEnv

import logging


class TabularModel:
    """To model transition probabilities for a tabular setting"""

    def __init__(self, env=None, p_mat=None):
        """ Initialize the model

        Args:
            env: environment to model
            p_mat: provide a transition probability matrix if available
        """
        self.logger = logging.getLogger(__name__)
        self._env = env
        if p_mat is None:
            self.logger.debug("Creating transition probability matrix")
            self._P_matrix()  # create transition probability matrix for vectorization
        else:
            self.logger.debug("Using provided transition probability matrix")
            self.P_mat = p_mat

    def _P_matrix(self):
        """Creates the transition probability matrix for vectorization"""
        P_mat = np.zeros((self.nb_states, self.nb_actions, self.nb_states))  # sxaxs (only suitable for small problems)
        if isinstance(self._env, MDPToolboxEnv):
            # Now P is an action x state x state matrix
            P_mat = np.moveaxis(self._env.P, [0, 1], [1, 0])
        else:
            env_P = self._env.env.P
            for s in env_P:
                for action in env_P[s]:
                    for p, next_s, *rest in env_P[s][action]:
                        P_mat[s, action, next_s] += p
        self.P_mat = P_mat
        return self.P_mat

    @property
    def nb_states(self):
        """number of possible states"""
        return self._env.observation_space.n

    @property
    def nb_actions(self):
        """number of actions from each state"""
        return self._env.action_space.n

    def check_valid_input(self, x):
        """For tabular models, the input must be a integer."""
        return isinstance(x, float)

    def neg(self, x, i=0):
        """
        Negative Cyclic permutation of the i-th coordinate of x.

        Args:
            x: array((num_samples, m)) , next state
            i: int, coordinate to be permuted (doesn't matter if x is 1D, i.e. tabular setting)
        """
        self.check_valid_input(x)
        x = np.atleast_2d(x)
        assert i < x.shape[1]

        l = np.array(range(self.nb_states))

        res = deepcopy(x)
        res[:, 0] = l[list(map(lambda e: (e + 1) % len(l), x.squeeze(-1)))]  # Negative cyclic permutation
        return res

    def neg_inv(self, x, i=0):
        """
        Cyclic permutation of the i-th coordinate of x.

        Args:
            x: array((num_samples, m)) , next state
            i: int, coordinate to be permuted (doesn't matter if x is 1D, i.e. tabular setting)
        """
        self.check_valid_input(x)
        x = np.atleast_2d(x)
        assert i < x.shape[1]

        l = np.array(range(self.nb_states))

        res = deepcopy(x)
        res[:, 0] = l[list(map(lambda e: (e - 1), x.squeeze(-1)))]  # cyclic permutation
        return res

    def score(self, x, sa=None, known_P=True):
        """
                Computes the (difference) score function.

                Args:
                    x: array((num_samples, m)) , next state
                    sa: array((num_samples, 2)), state-action pairs
                    known_P: bool, whether the transition probabilities are known

                Returns:
                    res: array((num_samples, m)), del p / p
        """
        # Note: Modfied for conditional probabilities
        if not known_P:
            raise NotImplementedError("Not implemented yet")
        else:
            assert self._env is not None, "Initialize env first"

        assert self.P_mat is not None, "Initialize P_mat first"
        assert sa is not None, "Need state-action pairs for conditional probabilities"

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        # Get the transition probabilities
        P_value = self.P_mat[sa[:, 0], sa[:, 1], x]

        # Get the transition probabilities for the negative cyclic permutation
        x = x[:, np.newaxis]
        P_neg = self.P_mat[sa[:, 0], sa[:, 1], self.neg(x).squeeze(-1)]

        # Compute the score. Clip to satisfy the positivity constraint on pmf
        res = 1 - P_neg / P_value.clip(min=EPS_CONST)

        return res[:, np.newaxis]
