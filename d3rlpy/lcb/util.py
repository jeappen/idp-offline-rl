import logging
import pickle
from typing import Any, Callable

import gym as old_gym
import gymnasium as gymnasium_module
import matplotlib.pyplot as plt
import numpy as np
import pickle5
import seaborn as sns

import d3rlpy
from d3rlpy.metrics.scorer import AlgoProtocol
from d3rlpy.preprocessing.stack import StackedObservation

SAVE_PENALTY_FREQ = 300
LARGE_CONST = 1e5
SMALL_CONST = 1e-5

MED_CONST = 1e-1
SMALL_SIZE = 8  # For text in the plots
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
HUGE_SIZE = 14
HUMONGO_SIZE = 20
HUMONGOROUS_SIZE = 28

# Experiment ids
value_iter_extension = 'valiter'
dsd_extension = 'dsd'
vanilla_extension = 'vanilla'
lcb_extension = 'lcb-tuned'
lcb_paper_extension = 'lcb-paper'
cql_extension = 'cql'
qling_extension = 'qling'
qling_dsd_extension = 'qling-dsd'
qling_lcb_extension = 'qling-lcb'

# Col Names
dataset_col_name = 'Dataset Size (N)'
main_exp_identifier = 'IDP'
main_exp_identifier_ext = dsd_extension

# Plotting
ax_map = lambda k: np.power(10, np.linspace(1, 4, 7)).astype(int).tolist().index(k)


def get_extensions_and_names(abbrv=False):
    """Get list of experiment ids and corresponding names."""
    extension_ids = [value_iter_extension, lcb_paper_extension, lcb_extension, dsd_extension, vanilla_extension]
    names_list = ['Value Iteration', 'VI-LCB (Paper)', 'VI-LCB (Tuned)', 'IDP-VI (Ours)', 'VI-Vanilla']
    if abbrv:
        names_list = ['VI', 'VI-LCB (P)', 'VI-LCB (T)', 'IDP-VI (O)', 'VI-Vanilla']
    extension_ids.extend([cql_extension, qling_extension, qling_dsd_extension, qling_lcb_extension])
    if abbrv:
        names_list.extend(['CQL', 'QL', 'IDP-Q (O)', 'QL LCB'])
    else:
        names_list.extend(['CQL', 'Q-Learning', 'IDP-Q (Ours)', 'Q-Learning LCB'])

    return extension_ids, names_list


def get_color_map():
    """Dictionary mapping experiment ids to colors."""
    # blue, orange, green, red, purple, brown, pink, gray, yellow, cyan
    #     new_colors = mcolors.TABLEAU_COLORS
    new_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                  'tab:olive', 'tab:cyan']
    color_list = ([None, ] + new_colors[:4]  # + ['tab:brown']#new_colors[-4:]
                  + [None, ] + new_colors[:3] + [None, ])
    map_ext_to_color = dict(zip(get_extensions_and_names()[1], color_list))
    #     map_ext_to_color['IDP-VI (Ours)'] = 'tab:purple'
    map_ext_to_color['VI-LCB (Paper)'] = 'tab:gray'
    return map_ext_to_color


class MDPDataset(d3rlpy.dataset.MDPDataset):
    """MDP dataset for LCB with helper functions."""

    def __init__(self, dataset_size=None, *args, **kwargs):
        """Initialize MDPDataset.

        Args:
            dataset_size (int): number of transitions in the dataset.
        """
        super().__init__(*args, **kwargs)
        self._shuffle = None
        self._transitions = None
        self._indices = None
        self.dataset_size = dataset_size

    def extend(self, dataset) -> None:
        """Extend the dataset with another dataset."""
        super().extend(dataset)
        self.dataset_size += dataset.dataset_size

    def _build_transitions(self):
        transitions = []
        for episode in self.episodes:
            transitions += episode.transitions
        # Convert to numpy array for faster indexing
        self._transitions = np.array(transitions)
        # Note: Transition also needs the index of step in the episode for Bernstein penalty (VI-LCB)
        self._indices = np.arange(len(self._transitions))
        if self._shuffle:
            np.random.shuffle(self._indices)

    def get_dataset_batches(self, batch_size=100, shuffle=True, ignore_done=True):
        """Get batches of transitions of the dataset."""
        if not ignore_done:
            raise NotImplementedError("Not implemented yet")
        if self._transitions is None:
            self._shuffle = shuffle
            self._build_transitions()

        current_batch = []
        current_ind = 0
        while current_ind < len(self._transitions):
            current_batch = self._transitions[self._indices[current_ind:current_ind + batch_size]]
            current_ind += batch_size
            yield current_batch

        return current_batch

    def get_dataset_batch(self, current_batch_ind=0, batch_size=100, shuffle=True, ignore_done=True):
        """Get batches of transitions of the dataset."""
        if not ignore_done:
            raise NotImplementedError("Not implemented yet")
        if self._transitions is None:
            self._shuffle = shuffle
            self._build_transitions()

        current_ind = current_batch_ind * batch_size
        return self._transitions[self._indices[current_ind:current_ind + batch_size]]

    def sample(self, sample_range=None, shuffle=True, ignore_done=True):
        """Get batches of transitions of the dataset.

        Args:
            sample_range (tuple): range of indices to sample from.
            shuffle (bool): flag to shuffle the dataset.
            ignore_done (bool): flag to ignore done transitions. (Not implemented yet)
        """
        if not ignore_done:
            raise NotImplementedError("Not implemented yet")
        if self._transitions is None:
            self._shuffle = shuffle
            self._build_transitions()
        if sample_range is None:
            sample_range = (0, len(self._transitions))

        transition = self._transitions[self._indices[np.random.randint(*sample_range)]]

        return transition


def evaluate_policy_on_env(
    policy,
    env,
    num_episodes=10,
    max_episode_steps=None,
    logger=None,
    render=False,
    callback=None,
    gymnasium=False,
    gamma=1.0
):
    """Evaluate policy on the environment.

    Args:
        policy (d3rlpy.policy.base.Policy): policy to evaluate.
        env (gym.Env): gym-like environment.
        num_episodes (int): number of episodes to evaluate.
        max_episode_steps (int): maximum steps for each episode.
        logger (logging.Logger): logger used in evaluation.
        render (bool): flag to render the environment.
        callback (Callable): callback function called after each episode.
        gymnasium (bool): flag to use gymnasium environment.

    Returns:
        dict: evaluation results.

    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("start evaluation")

    scores = []
    for i in range(num_episodes):
        score = 0.0
        observation = env.reset()
        if gymnasium:
            observation, info = observation
        done = False
        step = 0
        discount_factor = gamma
        while not done:
            if render:
                env.render()
            action = policy(observation)
            if gymnasium:
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # HACK: quick fix for gymnasium
            else:
                observation, reward, done, _ = env.step(action)
            score += discount_factor * reward
            step += 1
            discount_factor *= gamma
            if max_episode_steps is not None and step >= max_episode_steps:
                break
        scores.append(score)
        if callback is not None:
            callback(i, score)
    scores = np.array(scores)

    logger.info("evaluation ended for %d episodes", num_episodes)
    logger.info("average: %f", scores.mean())
    logger.info("median: %f", np.median(scores))
    logger.info("min: %f", scores.min())
    logger.info("max: %f", scores.max())
    logger.info("std: %f", scores.std())

    return {
        "mean": scores.mean(),
        "median": np.median(scores),
        "min": scores.min(),
        "max": scores.max(),
        "std": scores.std(),
    }


def evaluate_on_environment_scorer(
    env: old_gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False,
    gymnasium: bool = False, logger=None
) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.

    Returns:
        scoerer function.


    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("start evaluation")

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            if gymnasium:
                observation, info = observation
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        if (isinstance(env.observation_space, gymnasium_module.spaces.Discrete) or
                            isinstance(env.observation_space, old_gym.spaces.Discrete)):
                            # Need batch dimension for torch model input
                            observation = np.array([observation])
                        action = algo.predict([observation])[0]

                if gymnasium:
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated  # HACK: quick fix for gymnasium
                else:
                    observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)

        scores = np.array(episode_rewards)

        logger.info("evaluation ended")
        logger.info("average: %f", scores.mean())
        logger.info("median: %f", np.median(scores))
        logger.info("min: %f", scores.min())
        logger.info("max: %f", scores.max())
        logger.info("std: %f", scores.std())
        return float(np.mean(episode_rewards))

    return scorer


def plot_heatmap_penalty(off_info, off_vi_dsd, off_info_dsd, num_episodes=30):
    full_visit_mode = False

    for t_ind, penalty in off_info['saved_penalty'].items():

        fig, ax = plt.subplots(3, 1, figsize=(5, 10))
        fig.supxlabel('Actions')
        fig.supylabel('States')

        ax[0].title.set_text('Visitation')
        if full_visit_mode:
            visitation = off_vi_dsd.estimatePR.empirical_m_t[-1].sum(axis=-1).clip(0, 100)
        else:
            visitation = off_vi_dsd.estimatePR.get_batch_mt(t_ind).sum(axis=-1).clip(0, 100)
        sns.heatmap(visitation, ax=ax[0])

        ax[1].title.set_text('LCB Penalty')
        a = penalty  # off_info_dsd['final_penalty']
        sns.heatmap(a, ax=ax[1])

        ax[2].title.set_text('DSD Penalty')
        a = off_info_dsd['saved_penalty'][t_ind]
        sns.heatmap(a, ax=ax[2])

        plt.savefig(f'img/ds{num_episodes}_t{t_ind}_final_penalty{"_full" if full_visit_mode else ""}.png')


class CreateOfflineDataset:
    """Create an offline dataset from an environment and a policy."""

    def __init__(self, env, episode_length, n_episodes, seed, epsilon=0.3, proportion_optimal=1.0,
                 proportion_eps_optimal=1.0):
        """Initialize the dataset generator.

        Args:
            env: environment to use
            episode_length: length of each episode
            n_episodes: number of episodes to generate
            seed: seed for the environment
            epsilon: probability of taking a random action in epsilon-greedy data collection
            proportion_optimal: proportion of optimal data to use in the dataset (if integer, is number of episodes)
            proportion_eps_optimal: proportion of epsilon-greedy data to use in the dataset (if integer, is number of episodes)
            """
        self._dataset = None
        self._episode_length = episode_length
        self._n_episodes = n_episodes
        self._dataset_size = episode_length * n_episodes  # Max size of the dataset
        self._env = env
        self._epsilon = epsilon
        self._proportion_optimal = proportion_optimal
        self._proportion_eps_optimal = proportion_eps_optimal
        assert self._proportion_optimal >= 0.0, "Proportion of optimal data must be greater than 0 (None)"
        assert self._proportion_eps_optimal >= 0.0, "Proportion of epsilon-greedy data must be greater than 0 (None)"

    def _generate_dataset_helper(self, policy, seed, gymnasium=False, proportion=1.0, dataset2add2=None):
        if proportion >= SMALL_CONST:
            if isinstance(proportion, int):
                # Use a fixed number of optimal episodes
                generated_dataset = self._generate_dataset(policy, seed, gymnasium,
                                                           num2sample=proportion)
            else:
                # Use a proportion of optimal episodes
                generated_dataset = self._generate_dataset(policy, seed, gymnasium,
                                                           num2sample=int(proportion * self._n_episodes))
            if dataset2add2 is not None:
                dataset2add2.extend(generated_dataset)
            return generated_dataset
        return None

    def generate_full_dataset(self, policy, seed, gymnasium=False):
        """Generate a dataset using a policy and an environment."""
        random_policy = lambda obs: self._env.action_space.sample()
        dataset = self._generate_dataset(random_policy, seed, gymnasium)
        self._generate_dataset_helper(policy, seed, gymnasium, proportion=self._proportion_optimal,
                                      dataset2add2=dataset)
        # TODO: Append random policy from other states to the dataset
        eps_random_policy = lambda obs: policy(obs) if np.random.random() < self._epsilon else \
            self._env.action_space.sample()
        self._generate_dataset_helper(eps_random_policy, seed, gymnasium,
                                      proportion=self._proportion_eps_optimal, dataset2add2=dataset)
        self._dataset = dataset
        return self._dataset

    def _generate_dataset(self, policy, seed, gymnasium=False, num2sample=None):
        """Generate a dataset using a policy and an environment."""
        if num2sample is None:
            num2sample = self._n_episodes

        # TODO: add a seed to the environment
        num_episodes = 0
        transitions = []
        for i in range(num2sample):
            obs = self._env.reset()
            if gymnasium:
                obs, info = obs
            for j in range(self._episode_length):
                action = policy(obs)
                if gymnasium:
                    next_obs, reward, terminated, truncated, info = self._env.step(action)
                    done = terminated or truncated  # HACK: quick fix for gymnasium
                else:
                    next_obs, reward, done, _ = self._env.step(action)
                if j == self._episode_length - 1:
                    # To handle non-terminating environments
                    done = True
                if not isinstance(obs, np.ndarray):
                    obs = np.array([obs])
                if not isinstance(next_obs, np.ndarray):
                    next_obs = np.array([next_obs])
                # NOTE: on episode end, the next obs is not used usually so dataset throws it away
                # Fixed this with a patch to d3rlpy
                transitions.append((obs, action, reward, next_obs, done))
                obs = next_obs[0]
                if done:
                    break
            num_episodes += 1

        # return transitions
        o, a, rw, o_next, done = list(map(np.array, list(zip(*transitions))))

        # Now o_next is used in the dataset (Possible improvement: use it to compute the Q-values)
        return MDPDataset(len(transitions), o, a, rw, done, next_observations=o_next)


def log_results(stat_list, env_name='FrozenLake-v1',
                exp_id=""):
    assert stat_list is not None
    import csv
    with open(f'log/{exp_id}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['name', 'mean', 'std'])
        for stat in stat_list:
            writer.writerow([stat['name'], stat['mean'], stat['std']])


from collections import Counter


class LoadExperimentOutput(object):
    """Tools to load experiment output"""

    def load_with_postfix(self, pickle_output_filenames, postfix_groups=None):
        """Consolidate all pickle files into a single dictionary"""
        hparam_concat = {}
        if postfix_groups is None:
            # No postfix
            postfix_groups = [None] * len(pickle_output_filenames)
        for pickle_output_filename, postfix_group in zip(pickle_output_filenames, postfix_groups):
            with open(pickle_output_filename, 'rb') as f:
                try:
                    loaded_result = pickle.load(
                        f)  # dict of results {10: {seed: seed_results}, 100: {seed: seed_results}}
                except ValueError:
                    # Use pickle5 for python 3.7
                    loaded_result = pickle5.load(
                        f)  # dict of results {10: {seed: seed_results}, 100: {seed: seed_results}}
                # Add postfix to each key
                if postfix_group is not None:
                    ext, name = postfix_group
                    for loadk in loaded_result:
                        # Over episode numbers
                        # Update the keys of only the main experiment
                        updated_main_result = {f"{k}-{ext}": v for k, v in loaded_result[loadk].items() if
                                               main_exp_identifier_ext in k}
                        not_main_result = {k: v for k, v in loaded_result[loadk].items() if
                                           main_exp_identifier_ext not in k}
                        loaded_result[loadk] = {**updated_main_result, **not_main_result}
                # Combine over pickle files
                for k in loaded_result:
                    if k in hparam_concat:
                        hparam_concat[k].update(loaded_result[k])
                    else:
                        hparam_concat[k] = loaded_result[k]
        return hparam_concat


class HParamLogger(LoadExperimentOutput):
    """Tools to get Hparam details of experiments"""

    def _load_hparams_and_count(self, pickle_output_filenames, env_name='FrozenLake-v1', extra_exp_prefix="",
                                postfix_groups=None, run_subset=None):
        """Search all pickle files for hyperparameter used
            postfix_groups: List of (extension postfixes, name postfix) to add to each key
        """
        hparam_concat = self.load_with_postfix(pickle_output_filenames, postfix_groups)
        selected_hparams = {}
        for loadk in hparam_concat:
            selected_hparams[loadk] = {}
            for k, v in hparam_concat[loadk].items():
                if isinstance(v, tuple) and isinstance(v[2], dict):
                    ext = k.split('_')[1]
                    param_dict_for_expt = {param_name: param_val for param_name, param_val in v[2].items()
                                           if isinstance(param_val, int) or isinstance(param_val, float)}
                    # Add L_c value if L there
                    if 'L' in param_dict_for_expt and lcb_extension in k:
                        param_dict_for_expt['L_c'] = self._get_lc_value(v[2])
                    if ext not in selected_hparams[loadk]:
                        selected_hparams[loadk][ext] = {}
                    # Update each param counter
                    for param_name, param_val in param_dict_for_expt.items():
                        if param_name in selected_hparams[loadk][ext]:
                            selected_hparams[loadk][ext][param_name].update([param_val])
                        else:
                            selected_hparams[loadk][ext][param_name] = Counter([param_val])
                else:
                    # These two will never have reqd output format
                    if 'cql' not in k and k != 'extra_info':
                        print(f"mismatched output format{k}")
        return selected_hparams

    def _get_lc_value(self, param_list_for_lcb):
        """Helper function just to get the L_c value from L using other parameters"""
        n_states, n_actions, _ = next(iter(param_list_for_lcb['saved_visitation'].values())).shape
        if 'delta' not in param_list_for_lcb:
            delta = 0.01
        else:
            delta = param_list_for_lcb['delta']
        T = param_list_for_lcb['T']
        lc_multiplier = np.log(2 * (T + 1) * n_states * n_actions / delta)
        return param_list_for_lcb['L'] / lc_multiplier

    def get_parameters_of_interest(self, extra_ext=None):
        # Set up parameters of interest
        all_ext, _ = get_extensions_and_names()
        if extra_ext is None:
            extra_qling_ext = [qling_dsd_extension + "-knownp"]
            extra_ext = extra_qling_ext
        interested_param_list = {k: [] for k in all_ext + extra_ext}
        interested_param_list.update(
            {ql: ['eta'] for ql in [qling_dsd_extension, qling_extension, qling_lcb_extension] + extra_qling_ext})
        interested_param_list[qling_lcb_extension] += ['cb']
        for k in all_ext + extra_ext:
            if main_exp_identifier_ext in k:
                interested_param_list[k] += ['alpha_b', 'beta_b', 'gamma_b']
        interested_param_list[lcb_extension] = ['L_c', 'V_max']
        interested_param_list[lcb_paper_extension] = ['L_c', 'V_max']
        return interested_param_list

    def _filter_params_and_display(self, param_counter_output, interested_param_list=None):
        """Only display the few hparams of interest"""
        if interested_param_list is None:
            interested_param_list = self.get_parameters_of_interest()
        saved_params = {}
        for k in param_counter_output:
            if interested_param_list is None:
                saved_params[k] = {expt: {pk: pv.most_common(2) for pk, pv in param_counter_output[k][expt].items()}
                                   for expt in param_counter_output[k]}
            else:
                saved_params[k] = {expt: {pk: pv.most_common(2) for pk, pv in param_counter_output[k][expt].items() if
                                          expt in interested_param_list and k in interested_param_list[expt]}
                                   for expt in param_counter_output[k]}
        return saved_params

    def _filter_params_and_display_agg(self, param_counter_output, interested_param_list=None):
        """Only display the few hparams of interest"""
        if interested_param_list is None:
            interested_param_list = self.get_parameters_of_interest()
        saved_params = {}
        for k in param_counter_output:
            if interested_param_list is None:
                params_counters = param_counter_output[k]
            else:
                params_counters = {expt: {pk: pv for pk, pv in param_counter_output[k][expt].items()
                                          if expt in interested_param_list and pk in interested_param_list[expt]}
                                   for expt in param_counter_output[k]}
            # Now merge all counters for a given expt
            for expt in params_counters:
                if expt in saved_params:
                    for pk, pv in params_counters[expt].items():
                        # merge counter for given param pk
                        saved_params[expt][pk].update(pv)
                else:
                    saved_params[expt] = params_counters[expt]

        return saved_params

    def load_pickle_files_get_params(self, pickle_output_filenames, env_name='FrozenLake-v1', extra_exp_prefix="",
                                     postfix_groups=None):
        hparam_counters = self._load_hparams_and_count(pickle_output_filenames, env_name=env_name,
                                                       extra_exp_prefix=extra_exp_prefix, postfix_groups=postfix_groups)
        aggregated_counters = self._filter_params_and_display_agg(hparam_counters)
        return aggregated_counters

    def get_topk_agg(self, counter_dict, k=1):
        """Helper function to print the best params from aggregate"""
        output_dict = {}
        for expt in counter_dict:
            if expt not in output_dict:
                output_dict[expt] = {}
            for param in counter_dict[expt]:
                output_dict[expt][param] = counter_dict[expt][param].most_common(1)[0]
        return output_dict
