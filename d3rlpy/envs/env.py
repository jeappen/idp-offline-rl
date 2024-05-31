from itertools import product

import gymnasium as gym
import numpy as np


class dummy_env:
    """For compatibility with TabularModel"""

    def __init__(self, P, R):
        """Converts P and R matrix into PR[s,a] = (p, s', r, done)"""
        self.n_states = P.shape[0]
        self.n_actions = P.shape[1]
        self.P = {i: {j: [] for j in range(self.n_actions)} for i in range(self.n_states)}
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_prime in range(self.n_states):
                    self.P[s][a].append((P[s, a, s_prime], s_prime, R[s, a, s_prime], False))


class MDPToolboxEnv(gym.Env):
    """MDP Toolbox environment as gym environment."""

    def __init__(self, P=None, R=None, spec_id="", max_episode_steps=100):
        self.t = None
        self.state = None
        self.max_episode_steps = max_episode_steps
        if self.spec is None and spec_id != "":
            self.spec = gym.envs.registration.EnvSpec(f"{spec_id}-v0", entry_point='d3rlpy.envs:MDPToolboxEnv')
        if P is not None and R is not None:
            self.initialize_PR(P, R)

    def initialize_PR(self, P, R):
        """Initialize the environment with transition probabilities and rewards."""
        self.P = P
        self.R = R
        P_mat_sas = np.moveaxis(P, [0, 1], [1, 0])  # sxaxs (To be consistent with the tabular model)
        R_mat_sas = np.moveaxis(R, [0, 1], [1, 0])  # sxaxs or sxa (To be consistent with the tabular model)
        self.env = dummy_env(P_mat_sas, R_mat_sas)
        self.n_states = P.shape[1]
        self.n_actions = P.shape[0]
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.reset()

    def reset(self, reset_state=0, seed=None, options=None):
        """Reset the environment."""
        self.state = reset_state
        self.t = 0
        super().reset(seed=seed, options=options)
        return self.state, {}

    def step(self, action):
        """Take an action in the environment."""
        assert self.state is not None, "Call reset() first"
        assert self.action_space.contains(action), f"Invalid action: {action} is not in {self.action_space}"
        old_state = self.state
        self.state = np.random.choice(self.n_states, p=self.P[action, self.state])
        if len(self.R.shape) == 2:
            reward = self.R[action, old_state]
        else:
            reward = self.R[action, old_state, self.state]
        self.t += 1
        trunc = self.t >= self.max_episode_steps
        term = False
        info = {}
        return self.state, reward, term, trunc, info


_np = np


def randDense(states, actions, mask=None, beta_mode=False):
    """Generate random dense ``P`` and ``R``. See ``mdptoolbox.example.rand`` for details.

    Includes beta distribution sampling option.
    """
    # definition of transition matrix : square stochastic matrix
    P = _np.zeros((actions, states, states))
    # definition of reward matrix (values between -1 and +1)
    R = _np.zeros((actions, states, states))
    for action in range(actions):
        for state in range(states):
            # create our own random mask if there is no user supplied one
            if mask is None:
                m = _np.random.random(states)
                r = _np.random.random()
                m[m <= r] = 0
                m[m > r] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            # Make sure that there is atleast one transition in each state
            if m.sum() == 0:
                m[_np.random.randint(0, states)] = 1
            if beta_mode:
                P[action][state] = m * _np.random.beta(0.5, 0.5, states)  # Bimodal
            else:
                P[action][state] = m * _np.random.random(states)  # Uniform
            P[action][state] = P[action][state] / P[action][state].sum()
            R[action][state] = (m * (2 * _np.random.random(states) -
                                     _np.ones(states, dtype=int)))
    return (P, R)


def deepSea(states):
    """Generate ''P'' and ''R'' for the deep sea chain MDP problem.
    """
    actions = 2
    # definition of transition matrix : square stochastic matrix
    P = _np.zeros((actions, states, states))
    # definition of reward matrix (values between -1 and +1)
    R = _np.zeros((actions, states, states))
    for state in range(states):
        # left transition unless at 0
        left_transition = _np.zeros(states)
        left_transition[max(0, state - 1)] = 1
        P[0][state] = left_transition

        # right transition unless at end loop back to 0
        right_transition = _np.zeros(states)
        if state == states - 1:
            right_transition[0] = 1 - 1 / float(states)
        else:
            right_transition[state + 1] = 1 - 1 / float(states)
        right_transition[max(0, state - 1)] = 1 / float(states)
        P[1][state] = right_transition
    R[1][states - 2][states - 1] = 1
    return (P, R)


class PortfolioActionEnv(gym.Env):
    """Portfolio environment which takes action-dependent transitions as gym environment."""

    def __init__(self, num_assets,
                 risk_free_return=0.1,
                 state_transition_kernel=None,
                 asset_discretization_steps=10,
                 action_discretization_steps=10,
                 price_lb=1.0, price_ub=10.0,
                 max_steps=np.inf,
                 positivity_constraint=True, beta_mode=False):
        self.num_assets = num_assets
        self.tau = risk_free_return
        self.p = state_transition_kernel
        self.asset_delta = asset_discretization_steps
        self.action_delta = action_discretization_steps
        self.price_lb = price_lb
        self.price_ub = price_ub
        self.max_steps = max_steps
        if self.spec is None:
            envname = f"PortEnv-{'Hi' if price_ub >= 100 else 'Lo'}{'-beta' if beta_mode else ''}-v0"
            self.spec = gym.envs.registration.EnvSpec(envname,
                                                      entry_point='d3rlpy.envs.env:PortfolioActionEnv')

        print('Creating state and action spaces...')

        # Set up internal representations of portfolio
        self.states = list(product(
            *[np.linspace(price_lb, price_ub, self.asset_delta)
              for _ in range(num_assets)]
        ))
        self.state_dict = {s: i for i, s in enumerate(self.states)}
        self.num_to_state_dict = {v: k for k, v in self.state_dict.items()}

        self.actions = list(product(
            *[np.linspace(0, 1, self.action_delta)
              for _ in range(num_assets)]
        ))
        self.actions = [action for action in self.actions if sum(action) == 1.0]
        self.action_dict = {a: i for i, a in enumerate(self.actions)}
        self.num_to_action_dict = {v: k for k, v in self.action_dict.items()}

        print('Computing action dep probs...')

        if self.p == None:
            probs = {}
            for state in self.states:
                for action in self.actions:
                    # make mask
                    states = len(self.states)
                    m = np.random.random(states)
                    r = np.random.random()
                    m[m <= r] = 0
                    m[m > r] = 1
                    if m.sum() == 0:
                        m[np.random.randint(0, states)] = 1

                    if beta_mode:
                        dist = np.random.beta(0.5, 0.5, len(self.states))
                    else:
                        dist = np.random.random(len(self.states))
                    if positivity_constraint:
                        # For DSD we need to ensure that the probabilities are positive
                        dist = dist.clip(min=1e-2)
                    probs[(state, action)] = dist / np.sum(dist)

            def transition_probs(state, action):
                return probs[(state, action)]

            self.p = transition_probs

        #         print('Computing rewards...')

        #         self.rewards = {}
        #         for state in self.states:
        #             np_state = np.array(state)
        #             for action in self.actions:
        #                 np_action = np.array(action)
        #                 gain = 0.0
        #                 for next_state in self.states:
        #                     np_next_state = np.array(next_state)
        #                     raw_return = (np_next_state - np_state) / np_state
        #                     gain += (np_action.dot(raw_return)) * \
        #                         self.p(state)[self.state_dict[next_state]]
        #                 self.rewards[state, action] = gain

        #         print('Computing excesses and shortfalls...')

        #         self.excess, self.shortfall = {}, {}
        #         for state in self.states:
        #             for action in self.actions:
        #                 self.excess[state, action] = max(0, self.rewards[state, action] - self.tau)
        #                 self.shortfall[state, action] = max(0, self.tau - self.rewards[state, action])

        # Set up gym attributes
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self._init_prob()
        print('Done with init.')

    def _init_prob(self):
        """Converts P and R matrix into PR[s,a] = (p, s', r, done)"""

        class dummy_env:
            # For compatibility with TabularModel
            def __init__(self, P):
                self.P = P

        self.n_states = self.num_states
        self.n_actions = self.num_actions
        self.P = {i: {j: [] for j in range(self.n_actions)} for i in range(self.n_states)}

        for s_i in range(self.n_states):
            s = self.num_to_state_dict[s_i]
            for a_i in range(self.n_actions):
                a = self.num_to_action_dict[a_i]

                next_p = self.p(s, a)
                for s_prime_i in range(self.n_states):
                    s_prime = self.num_to_state_dict[s_prime_i]
                    self.P[s_i][a_i].append((next_p[s_prime_i], s_prime_i, self.reward(s, a, s_prime), False))
        self.env = dummy_env(self.P)

    def reward(self, state, action, next_state):
        np_state = np.array(state)
        np_action = np.array(action)
        np_next_state = np.array(next_state)
        return np_action.dot((np_next_state - np_state) / np_state)

    def reset(self):
        self.time_step = 0
        self.state = self.observation_space.sample()
        return self.state, {}

    def sample_next_state(self, state, action):
        transition_probs = self.p(state, action)
        next_state_num = np.random.choice(np.arange(self.num_states), p=transition_probs)
        return next_state_num

    def step(self, action):
        assert action in self.action_space
        action = self.num_to_action_dict[action]
        state = self.num_to_state_dict[self.state]
        next_state_num = self.sample_next_state(state, action)
        next_state = self.num_to_state_dict[next_state_num]
        reward = self.reward(state, action, next_state)
        excess = max(0, reward - self.tau)
        shortfall = max(0, self.tau - reward)
        self.state = next_state_num
        done = (self.time_step >= self.max_steps)
        self.time_step += 1
        return self.state, reward, False, done, {}


import mdptoolbox.example

P, R = mdptoolbox.example.forest()

gym.register('MDPToolbox-Forest-v0', entry_point='d3rlpy.envs:MDPToolboxEnv', kwargs={'P': P, 'R': R})

P, R = mdptoolbox.example.forest(S=100)

gym.register('MDPToolbox-Forest-Large-v0', entry_point='d3rlpy.envs:MDPToolboxEnv', kwargs={'P': P, 'R': R})

P, R = mdptoolbox.example.forest(S=1000)

gym.register('MDPToolbox-Forest-Huge-v0', entry_point='d3rlpy.envs:MDPToolboxEnv', kwargs={'P': P, 'R': R})

P, R = mdptoolbox.example.forest(S=1000)

gym.register('MDPToolbox-Random-Huge-v0', entry_point='d3rlpy.envs:MDPToolboxEnv', kwargs={'P': P, 'R': R})
