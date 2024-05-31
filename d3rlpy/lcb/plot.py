import pickle

import numpy as np
import pickle5
from matplotlib import pyplot as plt, colors as mcolors

from d3rlpy.lcb.util import main_exp_identifier, MED_CONST, SMALL_CONST, get_extensions_and_names, qling_extension, \
    dataset_col_name, value_iter_extension, cql_extension, main_exp_identifier_ext, log_results


def plot_all_results(stat_list=None, env_name='FrozenLake-v1',
                     exp_id="", plot_name="", names_list=None):
    assert stat_list is not None
    if names_list is None:
        names_list = ['Value Iteration', 'Off-VI LCB', 'Off-VI DSD', 'Off-VI Vanilla']
    x_pos = np.arange(len(names_list))
    error = list(map(lambda k: k['std'], stat_list))
    y_val = list(map(lambda k: k['mean'], stat_list))

    # Build the plot
    fig, ax = plt.subplots()
    bars = ax.bar(x_pos, y_val, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10, )
    ax.set_ylabel('Average Return')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_list)
    ax.set_title(f'{plot_name} Performance of Algorithms on {env_name}')
    ax.yaxis.grid(True)
    # ax.set_ylim(0,1)
    ax.bar_label(bars, fmt='%.2f', label_type='center', padding=-55)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f'img/{exp_id}_{env_name}_HPARAM_bar.png')
    plt.show()


def plot_training_iter_std(stat_list=None, env_name='FrozenLake-v1',
                           exp_id="", plot_name="", names_list=None,
                           baseline_stat=None, main_exp_identifier_arg=main_exp_identifier,
                           cql_baseline_stat=None, cql2_baseline_stat=None,
                           xlim=None, ylim=None, vline=None, qling_mode=False):
    """Plot training curves for multiple seeds with std

    Args:
    vline: tuple of floats showing factor of T to show as vertical line
    qling_mode: if true, scale vertical lines appropriately
    """
    plt.clf()
    plt.close('all')
    spec_score_log = "environment_spec.csv"
    fig, ax = plt.subplots(1, 1)  # , figsize=(15, 20), sharex=True, sharey=True)

    # plt.setp(ax, ylim=(-1, 0.5))

    if baseline_stat is not None:
        plt.axhline(y=baseline_stat, color='r', linestyle='--', label='Value Iteration Baseline')
        if ylim is None:
            ylim = np.ceil(baseline_stat + MED_CONST)
    if cql_baseline_stat is not None:
        plt.axhline(y=cql_baseline_stat, color='r', linestyle='-.', label='CQL Baseline')
    if cql2_baseline_stat is not None:
        plt.axhline(y=cql2_baseline_stat, color='r', linestyle='dotted', label='CQL Baseline (1 Epoch)')
    main_lines = []

    def get_halfway_black(c):
        # Make the color half as dark, (0,0,0) means black
        a_color = mcolors.to_rgba(c, 1.0)
        new_color = [0.5 * a_color[0], 0.5 * a_color[1], 0.5 * a_color[2], a_color[3]]
        return mcolors.rgb2hex(new_color, keep_alpha=True)

    max_x = -1
    for stat in stat_list:
        y = stat['mean'][:, 1]
        x = stat['mean'][:, 0]
        if x[-1] > max_x:
            max_x = x[-1]
        y_std = stat['std'][:, 1]
        marker = '^' if main_exp_identifier_arg in stat['name'] else None
        linewidth = 2 if main_exp_identifier_arg in stat['name'] else 1

        if len(main_lines) and main_exp_identifier_arg in stat['name']:
            # Can highlight the main exp after legend to keep colors on graph
            # for main_line in main_lines:
            #     main_line.set_color("black")
            # Keep the main lines the same color
            new_color = main_lines[-1].get_color()
            new_color = get_halfway_black(new_color)
        else:
            # Don't change the color for other lines and first main line
            new_color = None
        plot_op, = ax.plot(x, y, label=stat['name'], linewidth=linewidth, marker=marker, color=new_color)
        if main_exp_identifier_arg in stat['name']:
            main_lines.append(plot_op)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    ax.set_title(f'{plot_name} Training on {env_name}')
    if xlim is None:
        # For tight plots
        xlim = max_x
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)  # Assuming ideal returns are positive

    if vline is not None:
        # Remove grid lines if plotting vlines
        ax.grid(False)
        # HACK: Search Plot name for dataset size
        import re
        ds_size = float(re.search(r'T=([\d\.]+e\+\d+)', plot_name).groups()[0])
        x_scaler = ds_size if qling_mode else max_x  # Max x is T in VI mode since 1 iteration is a minibatch
        for v in vline:
            lineat = int(x_scaler * v)
            vline_text = f"{v}T"
            if np.abs(v - 1) < SMALL_CONST:
                # v is too close to 1
                vline_text = "T"
            if max_x < lineat:
                continue
            plt.vlines(x=lineat, ymin=0, ymax=ylim, color='gray', zorder=1, alpha=0.2)
            plt.text(lineat, ylim / 2, vline_text, verticalalignment='center', ha='center', va='center',
                     rotation='vertical', backgroundcolor='white')

    a = ax
    a.legend()
    fig.supxlabel('Training Iterations')
    fig.supylabel('Return')
    plt.savefig(f"img/trainperf_{exp_id}.png", dpi=300)


def combine_multiple_outputs(seed_results_list, exp_id="combine", extra_exp_prefix="",
                             num_episodes=10, dataset_size=None, plot_perf_interval=10,
                             env_name='FrozenLake-v1', run_subset=None, get_ext_fn=None,
                             **plot_args):
    """To combine different algorithms into a single plot

    Args:
        seed_results_list: List of seed results dictionaries for a particular number
        get_ext_fn: Function to get extension names and ids for each algorithm (extension_ids, names_list)
    """
    # Get mean and std over seeds
    seed_results = {}
    for seed_results_dict in seed_results_list:
        for k, v in seed_results_dict.items():
            if k in seed_results:
                print(f"Warning: Duplicate key {k} found in seed results, replacing with {v}")
            seed_results[k] = v

    if 'extra_info' in seed_results:
        num_episodes = seed_results['extra_info']['num_episodes']
        dataset_size = seed_results['extra_info']['dataset_size']
        plot_perf_interval = seed_results['extra_info']['plot_perf_interval']

    if get_ext_fn is None:
        get_ext_fn = get_extensions_and_names

    if dataset_size is None:
        # Last attempt to get dataset size, search the seed results
        for result in seed_results:
            if isinstance(result[2], dict):
                dataset_size = result[2].get('dataset_size', None)
                if dataset_size is not None:
                    break
        dataset_size = 1e5 if dataset_size is None else dataset_size  # Default to 100k

    plot_prefix = f"{num_episodes} Eps, T={dataset_size:.2}"  # Using T for dataset size from paper
    extension_ids, names_list = get_ext_fn()
    map_ext_to_names = {extension_ids[i]: names_list[i] for i in range(len(extension_ids))}
    # filter out results for each case
    qling_mode = False  # Whether to scale x axis by T
    scaling_x = 1  # Only used in the case of Qlearning plots
    if run_subset is not None:
        extension_ids = [k_extension for k_extension in extension_ids if k_extension in run_subset]
        if qling_extension in run_subset:
            scaling_x = seed_results[f"0_{qling_extension}"][2]['samples_per_iteration']
            print(f" Scaling x by {scaling_x} since plotting qling")
            qling_mode = True

    grouped_results = {k_extension: [v for k, v in seed_results.items() if k.split("_")[1] == k_extension] for
                       k_extension in extension_ids}
    # Filter names and results
    extension_ids = [k_extension for k_extension in extension_ids if len(grouped_results[k_extension]) > 0]
    names_list = [map_ext_to_names[k_extension] for k_extension in extension_ids]
    grouped_results = {k_extension: grouped_results[k_extension] for k_extension in extension_ids}

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
    plot_all_results(stat_list, exp_id=global_exp_id,
                     env_name=env_name, plot_name=f"{plot_prefix}, Mean",
                     names_list=names_list)
    # Add dataset size to log
    stat_list = [{'mean': dataset_size,
                  'std': 0,
                  'name': dataset_col_name}] + stat_list
    log_results(stat_list, exp_id=global_exp_id, env_name=env_name)
    if plot_perf_interval is not None:
        # Group and plot performance interval for each seed
        summary_results = {}
        for k_extension in grouped_results:
            # Replace with mean and std of grouped
            k_extension_root = k_extension.split("-")[0]  # Remove postfix
            if k_extension_root == value_iter_extension:
                # Save value iteration results since they are not over time
                summary_results[k_extension] = {
                    'mean': np.mean([v[3]['mean'] for v in grouped_results[k_extension]]),
                    'name': map_ext_to_names[k_extension]}
                continue
            elif k_extension_root == cql_extension:
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
            if scaling_x != 1:
                # To handle case where qling is plotted and does multiple samples per iteration
                summary_results[k_extension]['mean'][:, 0] *= scaling_x

        stat_list = [summary_results[k_extension] for k_extension in
                     extension_ids if not k_extension.split("-")[0] in [value_iter_extension, cql_extension]]
        cql_baseline_stat = summary_results.get(cql_extension, {"mean": None})['mean']  # Dummy result in case
        # TODO: Plot CQL performance interval appropriately (horizontal line) and 1 epoch
        plot_training_iter_std(stat_list, exp_id=global_exp_id,
                               env_name=env_name, plot_name=f"{plot_prefix},",
                               baseline_stat=summary_results[value_iter_extension]['mean'],
                               cql_baseline_stat=cql_baseline_stat, qling_mode=qling_mode, **plot_args)


def plot_all_pickled_outputs(pickle_output_filenames, env_name='FrozenLake-v1', extra_exp_prefix="", run_subset=None,
                             **plot_args):
    """To combine different pickled outputs into a single plot"""
    hparam_concat = {}
    for pickle_output_filename in pickle_output_filenames:
        with open(pickle_output_filename, 'rb') as f:
            try:
                loaded_result = pickle.load(f)  # dict of results {10: {seed: seed_results}, 100: {seed: seed_results}}
            except ValueError:
                # Use pickle5 for python 3.7
                loaded_result = pickle5.load(f)  # dict of results {10: {seed: seed_results}, 100: {seed: seed_results}}
            for k in loaded_result:
                if k in hparam_concat:
                    hparam_concat[k].update(loaded_result[k])
                else:
                    hparam_concat[k] = loaded_result[k]
    for k in hparam_concat:
        print(f"Plotting for {k}")
        print(f"keys: {hparam_concat[k].keys()}")
        exp_prefix = (f"{extra_exp_prefix}_" if len(extra_exp_prefix) != 0 else "") + f"combine_{k}"
        combine_multiple_outputs([hparam_concat[k]], extra_exp_prefix=exp_prefix,
                                 num_episodes=k, env_name=env_name, run_subset=run_subset, **plot_args)


def plot_all_pickled_outputs_postfix(pickle_output_filenames, postfix_groups=None, env_name='FrozenLake-v1',
                                     extra_exp_prefix="", run_subset=None, **plot_args):
    """To combine different pickled outputs into a single plot with postfixes
    Args:
        pickle_output_filenames: List of pickle output filenames
        postfix_groups: List of (extension postfixes, name postfix) to add to each key
    """
    hparam_concat = {}
    for pickle_output_filename, postfix_group in zip(pickle_output_filenames, postfix_groups):
        with open(pickle_output_filename, 'rb') as f:
            try:
                loaded_result = pickle.load(f)  # dict of results {10: {seed: seed_results}, 100: {seed: seed_results}}
            except ValueError:
                # Use pickle5 for python 3.7
                loaded_result = pickle5.load(f)  # dict of results {10: {seed: seed_results}, 100: {seed: seed_results}}
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
            for k in loaded_result:
                if k in hparam_concat:
                    hparam_concat[k].update(loaded_result[k])
                else:
                    hparam_concat[k] = loaded_result[k]

    def get_ext_fn_w_postfix():
        # Get extension ids and names with postfixes
        extension_ids, names_list = get_extensions_and_names()
        if postfix_groups is not None:
            # Add extensions and names for each of the postfixes
            for postfix_group in postfix_groups:
                if postfix_group is None:
                    continue
                ext, name_postfix = postfix_group
                postfix_extensions = [f"{k}-{ext}" for k in extension_ids]
                postfix_names = [f"{name} - {name_postfix}" for name in names_list]
                extension_ids += postfix_extensions
                names_list += postfix_names
        return extension_ids, names_list

    for k in hparam_concat:
        print(f"Plotting for {k}")
        print(f"keys: {hparam_concat[k].keys()}")
        exp_prefix = (f"{extra_exp_prefix}_" if len(extra_exp_prefix) != 0 else "") + f"postfix_combine_{k}"
        combine_multiple_outputs([hparam_concat[k]], extra_exp_prefix=exp_prefix,
                                 num_episodes=k, env_name=env_name, run_subset=run_subset,
                                 get_ext_fn=get_ext_fn_w_postfix, **plot_args)
