"""Helper scripts to run LCB algorithm."""
import itertools

import gymnasium as gym
import mdptoolbox
import torch.random
from sklearn.model_selection import train_test_split

from d3rlpy.algos import DiscreteCQL
from d3rlpy.envs import MDPToolboxEnv
from d3rlpy.lcb.dsd_algos import OfflineVIwDSD, QlearningwDSD
from d3rlpy.lcb.offvi import OfflineVI, OfflineVIwNoPenalty
from d3rlpy.lcb.online import ValueIteration
from d3rlpy.lcb.plot import plot_all_results, plot_training_iter_std
from d3rlpy.lcb.qling import Qlearning, QlearningLCB
from d3rlpy.lcb.util import *
from d3rlpy.lcb.util import log_results


# Compare other DSD kernel
def run_all_for_env(num_episodes=300, lc_range=None, alpha_range=None, beta_range=None, gamma_range=None,
                    known_p=False, seed_range=None, env_creator=None, v_max_range=None, separate_seed_plots=False,
                    plot_perf_interval=10, max_ep_length=200, dsd_kwargs=None, extra_exp_prefix="",
                    dataset_kwargs=None, vi_kwargs=None, lcb_kwargs=None, vanilla_kwargs=None, lcb_paper_kwargs=None,
                    eval_env_creator=None, eval_env_kwargs=None, eval_num_episodes=1000,
                    env_kwargs=None, fair_eval_mode=False, include_cql=False, num_cql_epochs=30,
                    include_qling=False, qling_kwargs=None, eta_range=None,
                    run_subset=None, estimator=None, estimator_kwargs=None, cb_range=None, delta_range=None,
                    alpha_search=False, test_mode=False, skip_plotting=False):
    """Train on FrozenLake dataset using Offline VI with DSD penalty
    Use separate plots for each seed when plotting for Random MDPs

    Args:
        fair_eval_mode (bool): If True, evaluate vanilla multiple times (num_seeds again) and average the results
        run_subset (list): List of algorithms to run. If None, run all algorithms. Check extension_ids for valid values
        alpha_search (bool): If True, try scaling all penalties by alpha
    """

    if seed_range is None:
        seed_range = [0]
    if lc_range is None:
        lc_range = [1e-3]

    if alpha_range is None:
        alpha_range = [1]

    if beta_range is None:
        beta_range = [0]

    if gamma_range is None:
        gamma_range = [None]

    if v_max_range is None:
        v_max_range = [1]

    if eta_range is None:
        eta_range = [0.5]

    if cb_range is None:
        cb_range = [10]

    if delta_range is None:
        delta_range = [0.01]

    if test_mode:
        # For quick testing
        lc_range = [1]  # np.power(10, np.linspace(-3, 3, 13))
        alpha_range = [1]  # np.power(10, np.linspace(-1, 1, 3))
        beta_range = [1]  # np.hstack((np.power(10, np.linspace(-1, 0, 2)), [0]))
        gamma_range = [1]  # np.hstack((np.power(10, np.linspace(-1, 1, 3)), [None]))
        eta_range = [0.1]
        cb_range = [1]
        seed_range = [1]  # list(range(3))
        v_max_range = [1]  # [1, 50, 100]
        alpha_search = False
        qling_kwargs = {'T': 100, 'samples_per_iteration': 1}

    # Make all output directories in case they are missing
    from pathlib import Path
    Path("datasets/").mkdir(parents=True, exist_ok=True)  # For pickled datasets
    Path("data/").mkdir(parents=True, exist_ok=True)  # For pickled results (use for plotting)
    Path("img/").mkdir(parents=True, exist_ok=True)  # For plots
    Path("log/").mkdir(parents=True, exist_ok=True)  # For final scores

    dsd_kwargs = {} if dsd_kwargs is None else dsd_kwargs
    dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
    vi_kwargs = {} if vi_kwargs is None else vi_kwargs
    lcb_kwargs = {} if lcb_kwargs is None else lcb_kwargs
    lcb_paper_kwargs = {} if lcb_paper_kwargs is None else lcb_paper_kwargs
    vanilla_kwargs = {} if vanilla_kwargs is None else vanilla_kwargs
    qling_kwargs = {'T': 100} if qling_kwargs is None else qling_kwargs
    env_kwargs = {'map_name': "4x4"} if env_kwargs is None else env_kwargs
    estimator_kwargs = {} if estimator_kwargs is None else estimator_kwargs

    exp_id = f"{'test_' if test_mode else ''}{'known_p' if known_p else 'unk_p'}_nep_{num_episodes}_lc{len(lc_range)}" \
             f"_alpha{len(alpha_range)}_beta{len(beta_range)}_gamma{len(gamma_range)}_vmax{len(v_max_range)}" \
             f"_seed{len(seed_range)}"

    best_lcb_policy = best_dsd_policy = best_vanilla_policy = best_cql_policy = best_qling_policy = \
        best_qling_dsd_policy = best_qling_lcb_policy = None

    extension_ids = [value_iter_extension, lcb_paper_extension, lcb_extension, dsd_extension, vanilla_extension]
    if include_cql:
        extension_ids.append(cql_extension)
    if include_qling:
        extension_ids.append(qling_extension)
        extension_ids.append(qling_dsd_extension)
        extension_ids.append(qling_lcb_extension)
    all_extensions, all_names = get_extensions_and_names()
    map_ext_to_names = dict(zip(all_extensions, all_names))
    names_list = [map_ext_to_names[k] for k in extension_ids]

    seed_results = {}  # Results for each seed
    # Allow option to run only a subset of experiments
    if run_subset is None:
        run_subset = extension_ids
    else:
        # Filter out expts not in run_subset
        assert all([k in extension_ids for k in run_subset]), "Invalid run subset"
        if value_iter_extension not in run_subset:
            # Need value iteration always for dataset anyway
            run_subset.append(value_iter_extension)
        if lcb_paper_extension in run_subset and lcb_extension not in run_subset:
            # Running all LCB expts if included
            run_subset.append(lcb_extension)
        extensions_and_names = zip(extension_ids, names_list)
        extensions_and_names_to_run = [k for k in extensions_and_names if k[0] in run_subset]
        extension_ids, names_list = zip(*extensions_and_names_to_run)

        # Now change exp_id
        hash_of_runset = hash(tuple(run_subset))
        exp_id = f"{exp_id}_runset{hash_of_runset}"

    # Allow scaling all penalties by a factor alpha
    all_penalties_alpha_range = set([1])
    if alpha_search:
        all_penalties_alpha_range = all_penalties_alpha_range.union(alpha_range)
    dataset_col_name = 'Dataset Size (N)'
    map_ext_to_names = {extension_ids[i]: names_list[i] for i in range(len(extension_ids))}
    dataset_sizes = []
    for seed in seed_range:
        np.random.seed(seed)
        torch.manual_seed(seed)
        # TODO: Also get mean of seed range and max in hyperparam range
        if env_creator is None:
            env = gym.make('FrozenLake-v1', desc=None, is_slippery=True, **env_kwargs)
        else:
            env = env_creator(**env_kwargs)

        if eval_env_creator is None:
            eval_env = env
        else:
            eval_env_kwargs = env_kwargs if eval_env_kwargs is None else eval_env_kwargs
            eval_env = eval_env_creator(**eval_env_kwargs)
        env_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else env.unwrapped.__class__.__name__
        seed_exp_id = f"seed_{seed}_{exp_id}_{env_name}"
        if len(extra_exp_prefix) != 0:
            seed_exp_id = f"{extra_exp_prefix}_{seed_exp_id}"
        if isinstance(env, MDPToolboxEnv):
            def run_vi_and_get_policy(env, gamma=0.9):
                vi = mdptoolbox.mdp.ValueIteration(env.P, env.R, gamma)
                vi.run()
                return lambda state: vi.policy[int(state)]  # result is (0, 0, 0)

            policy_fn = run_vi_and_get_policy(env)
            vi_info = {}

        else:
            policy_fn, vi_info = ValueIteration(env).run(**vi_kwargs)
        creator = CreateOfflineDataset(env, max_ep_length, num_episodes, seed, **dataset_kwargs)
        ds = creator.generate_full_dataset(policy_fn, seed, gymnasium=True)
        ds.dump(f"datasets/{seed_exp_id}.h5")  # For reproducibility
        dataset_sizes.append(ds.dataset_size)
        vi_stat = evaluate_policy_on_env(policy_fn, eval_env, gymnasium=True, num_episodes=eval_num_episodes)
        print(f"Value Iteration Statistics over {eval_num_episodes} evaluation episodes: {vi_stat}")

        seed_results[f"{seed}_{value_iter_extension}"] = (None, None, vi_info, vi_stat)

        if separate_seed_plots:
            print(f"Plotting for seed {seed}")
            # Reset the best policy for each seed
            best_lcb_policy = best_dsd_policy = best_vanilla_policy = best_cql_policy = best_qling_policy = \
                best_qling_dsd_policy = best_qling_lcb_policy = None

        # Paper LCB
        if lcb_paper_extension in run_subset:
            off_vi = OfflineVI(env, ds, eval_env=eval_env, estimator=estimator)
            off_policy, off_V_s, off_info = off_vi.offline_value_iteration(N=ds.dataset_size, Lc=2000,
                                                                           V_max=None, known_P=known_p,
                                                                           plot_perf_interval=plot_perf_interval,
                                                                           estimator_kwargs=estimator_kwargs)
            off_vi_stat = evaluate_policy_on_env(lambda obs: off_policy[obs], eval_env, gymnasium=True,
                                                 num_episodes=eval_num_episodes)
            seed_results[f"{seed}_{lcb_paper_extension}"] = (off_policy, off_V_s, off_info, off_vi_stat)

        # Tuned LCB
        if lcb_extension in run_subset:
            for lc_range_val, v_max, alpha_val in itertools.product(lc_range, v_max_range, all_penalties_alpha_range):
                off_vi = OfflineVI(env, ds, eval_env=eval_env, estimator=estimator, alpha_b=alpha_val)
                off_policy, off_V_s, off_info = off_vi.offline_value_iteration(N=ds.dataset_size, Lc=lc_range_val,
                                                                               V_max=v_max, known_P=known_p,
                                                                               plot_perf_interval=plot_perf_interval,
                                                                               estimator_kwargs=estimator_kwargs)
                off_vi_stat = evaluate_policy_on_env(lambda obs: off_policy[obs], eval_env, gymnasium=True,
                                                     num_episodes=eval_num_episodes)
                # Get best policy
                if best_lcb_policy is None or off_vi_stat['mean'] > best_lcb_stat['mean']:
                    best_lcb_policy = off_policy
                    best_lcb_stat = off_vi_stat
                    seed_results[f"{seed}_{lcb_extension}"] = (off_policy, off_V_s, off_info, off_vi_stat)
                    print(
                        f"New best policy found with Lc={lc_range_val} v_max={v_max} and mean score {off_vi_stat['mean']}")

        # Tuned DSD:  Iterate over all combinations of alpha, beta, gamma values
        if dsd_extension in run_subset:
            for alpha_range_val, beta_range_val, gamma_range_val in itertools.product(alpha_range, beta_range,
                                                                                      gamma_range):
                off_vi_dsd = OfflineVIwDSD(env, ds, alpha_b=alpha_range_val, beta_b=beta_range_val,
                                           gamma_b=gamma_range_val,
                                           eval_env=eval_env, estimator=estimator,
                                           **dsd_kwargs)
                off_policy_dsd, off_V_s_dsd, off_info_dsd = off_vi_dsd.offline_value_iteration(N=ds.dataset_size,
                                                                                               known_P=known_p,
                                                                                               plot_perf_interval=plot_perf_interval,
                                                                                               estimator_kwargs=estimator_kwargs)
                off_vi_dsd_stat = evaluate_policy_on_env(lambda obs: off_policy_dsd[obs], eval_env, gymnasium=True,
                                                         num_episodes=eval_num_episodes)
                # Get best policy
                if best_dsd_policy is None or off_vi_dsd_stat['mean'] > best_dsd_stat['mean']:
                    best_dsd_policy = off_policy_dsd
                    best_dsd_stat = off_vi_dsd_stat
                    seed_results[f"{seed}_{dsd_extension}"] = (
                        off_policy_dsd, off_V_s_dsd, off_info_dsd, off_vi_dsd_stat)
                    print(f"New best DSD policy found with alpha={alpha_range_val}, beta={beta_range_val},"
                          f" gamma={gamma_range_val} and mean score {off_vi_dsd_stat['mean']}")

        # Vanilla
        num_vanilla_eval = len(seed_range) if fair_eval_mode else 1
        if vanilla_extension in run_subset:
            for _ in range(num_vanilla_eval):
                off_vi_no_pen = OfflineVIwNoPenalty(env, ds, eval_env=eval_env, estimator=estimator)
                off_policy_no_pen, off_V_s_no_pen, off_info_no_pen = off_vi_no_pen.offline_value_iteration(
                    N=ds.dataset_size,
                    known_P=known_p,
                    plot_perf_interval=plot_perf_interval,
                    estimator_kwargs=estimator_kwargs)

                off_vi_no_pen_stat = evaluate_policy_on_env(lambda obs: off_policy_no_pen[obs], eval_env,
                                                            gymnasium=True,
                                                            num_episodes=eval_num_episodes)
                if best_vanilla_policy is None or off_vi_no_pen_stat['mean'] > best_vanilla_stat['mean']:
                    best_vanilla_policy = off_policy_no_pen
                    best_vanilla_stat = off_vi_no_pen_stat
                    seed_results[f"{seed}_{vanilla_extension}"] = (
                        off_policy_no_pen, off_V_s_no_pen, off_info_no_pen, best_vanilla_stat)
                    print(f"New best vanilla policy found with mean score {best_vanilla_stat['mean']}")

        # CQL (A DNN based approach)
        if include_cql and cql_extension in run_subset:
            num_epochs = num_cql_epochs
            # split train and test episodes
            train_episodes, test_episodes = train_test_split(ds, test_size=0.1)
            for _ in range(num_vanilla_eval):

                cql = DiscreteCQL(use_gpu=False)
                off_info = cql.fit(train_episodes,
                                   eval_episodes=test_episodes,
                                   n_epochs=num_epochs,
                                   scorers={'environment': evaluate_on_environment_scorer(env, gymnasium=True)})

                policy_fn = lambda state: cql.predict([np.array([state])])[0]
                off_vi_stat = evaluate_policy_on_env(policy_fn, eval_env, gymnasium=True,
                                                     num_episodes=eval_num_episodes)

                if best_cql_policy is None or off_vi_stat['mean'] > best_cql_stat['mean']:
                    # Convert neural policy into tabular policy for pickling
                    best_cql_policy = np.zeros(env.observation_space.n, dtype=int)
                    for state in range(env.observation_space.n):
                        best_cql_policy[state] = int(policy_fn(state))
                    best_cql_stat = off_vi_stat
                    off_q_fn = cql.predict_value

                    seed_results[f"{seed}_{cql_extension}"] = (best_cql_policy, off_q_fn, off_info, best_cql_stat)
                    print(f"New best cql policy found with mean score {best_cql_stat['mean']}")

        # Q-learning Experiments (Tabular)
        if include_qling and qling_extension in run_subset:
            # Set an appropriate T for the environment matching other experiments
            for saved_results in seed_results.values():
                if saved_results[2] is not None and 'T' in saved_results[2]:
                    print(f"Using T={saved_results[2]['T']} for Q-Learning")
                    qling_kwargs['T'] = saved_results[2]['T']
            if qling_dsd_extension in run_subset:
                # Tuned DSD:  Iterate over all combinations of alpha, beta, gamma values
                for alpha_range_val, beta_range_val, delta_val, eta_val in itertools.product(alpha_range, beta_range,
                                                                                             delta_range,
                                                                                             eta_range):
                    qling_dsd = QlearningwDSD(env, ds,
                                              alpha_b=alpha_range_val, beta_b=beta_range_val,
                                              # gamma_b=gamma_range_val,
                                              estimator=estimator,
                                              eval_env=eval_env,
                                              **dsd_kwargs)
                    offq_policy_dsd, offq_V_s_dsd, offq_info_dsd = qling_dsd.run(eta=eta_val,
                                                                                 delta=delta_val,
                                                                                 N=ds.dataset_size,
                                                                                 known_P=known_p,
                                                                                 plot_perf_interval=plot_perf_interval,
                                                                                 estimator_kwargs=estimator_kwargs,
                                                                                 **qling_kwargs)
                    qling_dsd_stat = evaluate_policy_on_env(lambda obs: offq_policy_dsd[obs], eval_env, gymnasium=True,
                                                            num_episodes=eval_num_episodes)
                    # Get best policy
                    if best_qling_dsd_policy is None or qling_dsd_stat['mean'] > best_qling_dsd_stat['mean']:
                        best_qling_dsd_policy = offq_policy_dsd
                        best_qling_dsd_stat = qling_dsd_stat
                        seed_results[f"{seed}_{qling_dsd_extension}"] = (
                            offq_policy_dsd, offq_V_s_dsd, offq_info_dsd, qling_dsd_stat)
                        print(f"New best Qling DSD policy found with alpha={alpha_range_val}, beta={beta_range_val},"
                              f" eta={eta_val} and mean score {qling_dsd_stat['mean']}")

            # Vanilla Q-Learning
            for eta_val in eta_range:
                qlinq = Qlearning(env, ds, estimator=estimator, eval_env=eval_env, **dsd_kwargs)
                offq_policy, offq_V_s, offq_info = qlinq.run(eta=eta_val, N=ds.dataset_size,
                                                             known_P=known_p,
                                                             plot_perf_interval=plot_perf_interval,
                                                             **qling_kwargs)
                qlinq_stat = evaluate_policy_on_env(lambda obs: offq_policy[obs], eval_env, gymnasium=True,
                                                    num_episodes=eval_num_episodes)
                # Get best policy
                if best_qling_policy is None or qlinq_stat['mean'] > best_qlinq_stat['mean']:
                    best_qling_policy = offq_policy
                    best_qlinq_stat = qlinq_stat
                    seed_results[f"{seed}_{qling_extension}"] = (
                        offq_policy, offq_V_s, offq_info, qlinq_stat)
                    print(f"New best Qling policy found with eta={eta_val} and mean score {qlinq_stat['mean']}")

            if qling_lcb_extension in run_subset:
                # Tuned QLCB:  Iterate over all combinations of alpha, beta, gamma values
                for cb_range_val, eta_val, alpha_val in itertools.product(cb_range, eta_range,
                                                                          all_penalties_alpha_range):
                    qling_lcb = QlearningLCB(env, ds,
                                             cb=cb_range_val,
                                             estimator=estimator,
                                             alpha_b=alpha_val,
                                             eval_env=eval_env,
                                             **dsd_kwargs)
                    offq_policy_dsd, offq_V_s_dsd, offq_info_dsd = qling_lcb.run(eta=eta_val,
                                                                                 N=ds.dataset_size,
                                                                                 known_P=known_p,
                                                                                 plot_perf_interval=plot_perf_interval,
                                                                                 estimator_kwargs=estimator_kwargs,
                                                                                 **qling_kwargs)
                    qling_lcb_stat = evaluate_policy_on_env(lambda obs: offq_policy_dsd[obs], eval_env, gymnasium=True,
                                                            num_episodes=eval_num_episodes)
                    # Get best policy
                    if best_qling_lcb_policy is None or qling_lcb_stat['mean'] > best_qling_lcb_stat['mean']:
                        best_qling_lcb_policy = offq_policy_dsd
                        best_qling_lcb_stat = qling_lcb_stat
                        seed_results[f"{seed}_{qling_lcb_extension}"] = (
                            offq_policy_dsd, offq_V_s_dsd, offq_info_dsd, qling_lcb_stat)
                        print(f"New best Qling LCB policy found with alpha={alpha_range_val}, beta={beta_range_val},"
                              f" eta={eta_val} and mean score {qling_lcb_stat['mean']}")

        if separate_seed_plots:
            # Plot the results for each seed
            stat_list = [vi_stat]
            if lcb_paper_extension in run_subset:
                paper_lcb_stat = seed_results[f"{seed}_{lcb_paper_extension}"][3]
                stat_list.append(paper_lcb_stat)
                stat_list.append(best_lcb_stat)
            if dsd_extension in run_subset:
                stat_list.append(best_dsd_stat)
            if vanilla_extension in run_subset:
                stat_list.append(best_vanilla_stat)
            if include_cql and cql_extension in run_subset:
                stat_list.append(best_cql_stat)
            if include_qling and qling_extension in run_subset:
                stat_list.append(best_qlinq_stat)
            if qling_dsd_extension in run_subset:
                stat_list.append(best_qling_dsd_stat)
            if qling_lcb_extension in run_subset:
                stat_list.append(best_qling_lcb_stat)

            if not skip_plotting:
                plot_all_results(stat_list, exp_id=seed_exp_id,
                                 env_name=env_name, names_list=names_list)

    plot_prefix = f"{num_episodes} Eps, N={np.mean(dataset_sizes):.2}"
    seed_results['extra_info'] = {'num_episodes': num_episodes, 'dataset_size': np.mean(dataset_sizes),
                                  'plot_perf_interval': plot_perf_interval, 'max_ep_length': max_ep_length,
                                  'fair_eval_mode': fair_eval_mode, 'env_name': env_name}

    # Final evaluation
    if separate_seed_plots:
        # Get mean and std over seeds
        run_results = seed_results
        # filter out results for each case from seed{num}_{extension}
        grouped_results = {k_extension: [v for k, v in seed_results.items() if k.split("_")[1] == k_extension] for
                           k_extension in extension_ids}
        # Bar plot of mean and std over seeds
        summary_results = {}
        for k_extension in grouped_results:
            # Replace with mean and std of grouped
            summary_results[k_extension] = {'mean': np.mean([v[3]['mean'] for v in grouped_results[k_extension]]),
                                            'std': np.std([v[3]['mean'] for v in grouped_results[k_extension]]),
                                            'name': map_ext_to_names[k_extension]}
        global_exp_id = f"{env_name}_N{num_episodes}_mean_seed_{exp_id}"
        if len(extra_exp_prefix) != 0:
            global_exp_id = f"{extra_exp_prefix}_{global_exp_id}"
        # Plot mean and std over seeds
        stat_list = [summary_results[k_extension] for k_extension in
                     extension_ids]
        if not skip_plotting:
            plot_all_results(stat_list, exp_id=global_exp_id,
                             env_name=env_name, plot_name=f"{plot_prefix}, Mean",
                             names_list=names_list)
        # Add dataset size to log
        stat_list = [{'mean': np.mean(dataset_sizes),
                      'std': 0,
                      'name': dataset_col_name}] + stat_list
        log_results(stat_list, exp_id=global_exp_id, env_name=env_name)
        if plot_perf_interval is not None:
            # Group and plot performance interval for each seed
            summary_results = {}
            for k_extension in grouped_results:
                # Replace with mean and std of grouped
                if k_extension == value_iter_extension:
                    # Save value iteration results since they are not over time
                    summary_results[k_extension] = {
                        'mean': np.mean([v[3]['mean'] for v in grouped_results[k_extension]]),
                        'name': map_ext_to_names[k_extension]}
                    continue
                elif k_extension == cql_extension:
                    # Save cql iteration results since they are not over a single epoch
                    summary_results[k_extension] = {
                        'mean': np.mean([v[3]['mean'] for v in grouped_results[k_extension]]),
                        # 'mean_1epoch': np.mean([v[3]['mean'] for v in grouped_results[k_extension]]),
                        'name': map_ext_to_names[k_extension]}
                    continue
                summary_results[k_extension] = {
                    'mean': np.mean([v[2]['policy_perf'] for v in grouped_results[k_extension]], axis=0),
                    'std': np.std([v[2]['policy_perf'] for v in grouped_results[k_extension]], axis=0),
                    'name': map_ext_to_names[k_extension]}
            stat_list = [summary_results[k_extension] for k_extension in
                         extension_ids if not k_extension in [value_iter_extension, cql_extension]]
            cql_baseline_stat = summary_results.get(cql_extension, {"mean": None})['mean']  # Dummy result in case
            if not skip_plotting:
                plot_training_iter_std(stat_list, exp_id=global_exp_id,
                                       env_name=env_name, plot_name=f"{plot_prefix},",
                                       baseline_stat=summary_results[value_iter_extension]['mean'],
                                       cql_baseline_stat=cql_baseline_stat)
    else:
        # Eval on best policies again
        global_exp_id = f"{env_name}_N{num_episodes}_max_seed_{exp_id}"
        if len(extra_exp_prefix) != 0:
            global_exp_id = f"{extra_exp_prefix}_{global_exp_id}"
        off_vi_stat = evaluate_policy_on_env(lambda obs: best_lcb_policy[obs], eval_env, gymnasium=True,
                                             num_episodes=eval_num_episodes)
        off_vi_dsd_stat = evaluate_policy_on_env(lambda obs: best_dsd_policy[obs], eval_env, gymnasium=True,
                                                 num_episodes=eval_num_episodes)
        off_vi_no_pen_stat = evaluate_policy_on_env(lambda obs: best_vanilla_policy[obs], eval_env, gymnasium=True,
                                                    num_episodes=eval_num_episodes)
        print(vi_stat, off_vi_stat, off_vi_dsd_stat, off_vi_no_pen_stat)
        stat_list = [vi_stat, off_vi_stat, off_vi_dsd_stat, off_vi_no_pen_stat]
        run_results = vi_stat, off_vi_stat, off_vi_dsd_stat, off_vi_no_pen_stat, seed_results
        if not skip_plotting:
            plot_all_results(stat_list, exp_id=global_exp_id,
                             env_name=env_name, plot_name=f"{plot_prefix}, Max ")

    return run_results
