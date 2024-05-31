"""Online Algorithms"""
import numpy as np


class ValueIteration:
    """Runs value iteration on a given MDP, example used 'Frozen-Lake-v0'"""

    def __init__(self, env, gamma=0.9, theta=0.01):
        self.env = env
        self.gamma = gamma
        self.theta = theta

    @property
    def nb_states(self):
        """number of possible states"""
        return self.env.observation_space.n

    @property
    def nb_actions(self):
        """number of actions from each state"""
        return self.env.action_space.n

    def flatten_mdp(self, policy, model):
        """Incorporate policy into the MDP to get a 'flattened' MRP"""
        P_pi = np.zeros([self.nb_states, self.nb_states])  # transition probability matrix (s) to (s')
        R_pi = np.zeros([self.nb_states])  # exp. reward from state (s) to any next state
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                for p_, s_, r_, _ in model[s][a]:
                    # p_ - transition probability from (s,a) to (s')
                    # s_ - next state (s')
                    # r_ - reward on transition from (s,a) to (s')
                    P_pi[s, s_] += policy[s, a] * p_  # transition probability (s) -> (s')
                    Rsa = p_ * r_  # exp. reward from (s,a) to any next state
                    R_pi[s] += policy[s, a] * Rsa  # exp. reward from (s) to any next state
        assert np.alltrue(
            (np.sum(P_pi, axis=-1) + 1e-5).astype(int) == np.ones([self.nb_states]))  # rows should sum to 1
        return P_pi, R_pi

    def calc_Q_pi(self, V_pi, model, lmbda):
        """Calculate Q_pi(s,a) for all (s,a) from the MDP model"""
        Q_pi = np.zeros([self.nb_states, self.nb_actions])
        for s in range(self.nb_states):
            for a in range(self.nb_actions):
                for p_, s_, r_, _ in model[s][a]:
                    # p_ - transition probability from (s,a) to (s')
                    # s_ - next state (s')
                    # r_ - reward on transition from (s,a) to (s')
                    Rsa = p_ * r_  # expected reward for transition s,a -> s_
                    Vs_ = V_pi[s_]  # state-value of s_
                    Q_pi[s, a] += Rsa + lmbda * p_ * Vs_
        return Q_pi

    def run(self, n_iter=10, n_eval=100):
        """Run value iteration"""

        old_V_pi = V_pi = np.zeros([self.nb_states])
        policy = np.ones([self.nb_states, self.nb_actions]) / self.nb_actions  # random policy, 25% each action

        for n in range(n_iter):
            old_V_pi = V_pi.copy()
            # flatten MDP
            P_pi, R_pi = self.flatten_mdp(policy, self.env.env.P)

            # evaluate policy
            for k in range(n_eval):
                V_pi = R_pi + self.gamma * P_pi @ V_pi

            # iterate policy
            Q_pi = self.calc_Q_pi(V_pi, self.env.env.P, self.gamma)
            a_max = np.argmax(Q_pi, axis=-1)  # could distribute actions between all max(q) values
            policy *= 0  # clear
            policy[range(self.nb_states), a_max] = 1  # pick greedy action

        def policy_fn(state):
            return policy[state].argmax()

        info = {'V_pi': V_pi, 'Q_pi': Q_pi,
                'v_pi_gap': np.max(np.abs(V_pi - old_V_pi))}  # max difference between old and new V_pi
        return policy_fn, info
