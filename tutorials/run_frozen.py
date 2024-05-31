import gzip

import gymnasium as gym

from d3rlpy.lcb.estimators import OnlineEstimatePR
from d3rlpy.lcb.run import run_all_for_env
from d3rlpy.lcb.util import *


def create_frozen(**env_kwargs):
    return gym.make('FrozenLake-v1', desc=None, is_slippery=True, **env_kwargs)


def run_exp(fair_eval_mode=True, num_seeds=10, extra_exp_prefix="run_exp", known_P=False, dataset="hard",
            run_mode="all", test_mode=False):
    hparam_results = {}
    num_episodes_range = np.power(10, np.linspace(1, 4, 7)).astype(int)
    lc_range = np.power(10, np.linspace(-3, 3, 13))
    alpha_range = np.power(10, np.linspace(-2, 1, 7))
    beta_range = np.hstack((np.power(10, np.linspace(-1, 2, 5)), [0]))
    gamma_range = np.hstack((np.power(10, np.linspace(-1, 1, 3)), [-1, None]))
    eta_range = np.hstack((np.power(10, np.linspace(-1, 0, 3)), [0.5]))
    cb_range = np.power(10, np.linspace(-2, 4, 4))
    seed_range = list(range(num_seeds))
    v_max_range = [1, None]
    if run_mode == "all":
        # Run all experiments (but Qlearning length might be too short)
        run_subset = None
    elif run_mode == "vi":
        # Run only VI experiments (no CQL)
        run_subset = [lcb_paper_extension, lcb_extension, dsd_extension, vanilla_extension]
    elif run_mode == "qling":
        # Run only Qlearning experiments
        run_subset = [cql_extension, qling_extension, qling_dsd_extension, qling_lcb_extension]
    elif run_mode == "qlingknownp":
        # Run only Qlearning experiments with known P for DSD penalty
        run_subset = [qling_extension, qling_dsd_extension]
        known_P = True
    hard_dataset_kwargs = {'epsilon': 0.3, 'proportion_optimal': 1,
                           'proportion_eps_optimal': 0.1}  # hard
    easy_dataset_kwargs = {'epsilon': 0.3, 'proportion_optimal': 1.0,
                           'proportion_eps_optimal': 1.0}  # easy
    rand_dataset_kwargs = {'epsilon': 0.3, 'proportion_optimal': 1,
                           'proportion_eps_optimal': 1}  # rand
    if dataset == "hard":
        dataset_kwargs = hard_dataset_kwargs
    elif dataset == "rand":
        dataset_kwargs = rand_dataset_kwargs
    else:
        dataset_kwargs = easy_dataset_kwargs
    vi_kwargs = {'n_iter': 30}
    env_kwargs = {'map_name': "4x4"}
    qling_kwargs = {'T': 6 * (10 ** 4), 'samples_per_iteration': 1}
    sample_env = create_frozen(**env_kwargs)

    if test_mode:
        # For quick testing
        num_episodes_range = [10]

    for num_episodes in num_episodes_range:
        print(f"Running {num_episodes} episodes")
        final_exp_prefix = f"{'test_' if test_mode else ''}{sample_env.spec.name}_{run_mode}_{dataset}_{extra_exp_prefix}"
        hparam_results[num_episodes] = run_all_for_env(num_episodes, lc_range=lc_range, alpha_range=alpha_range,
                                                       beta_range=beta_range, gamma_range=gamma_range,
                                                       known_p=known_P, separate_seed_plots=True,
                                                       extra_exp_prefix=final_exp_prefix,
                                                       vi_kwargs=vi_kwargs,
                                                       env_creator=create_frozen, fair_eval_mode=fair_eval_mode,
                                                       dataset_kwargs=dataset_kwargs, env_kwargs=env_kwargs,
                                                       seed_range=seed_range, v_max_range=v_max_range, include_cql=True,
                                                       include_qling=True, run_subset=run_subset,
                                                       estimator=OnlineEstimatePR, qling_kwargs=qling_kwargs,
                                                       eta_range=eta_range, cb_range=cb_range, test_mode=test_mode)
        print(f"Saving results for {num_episodes} episodes")
        # Save compressed result file
        with gzip.open(f'data/hparam_results_{final_exp_prefix}_{num_episodes}.pkl.gz', 'wb') as handle:
            pickle.dump(hparam_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return hparam_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--knownp", help="Use ground truth transition model",
                        action="store_true")
    parser.add_argument("--extra-exp-prefix", type=str, default="run_frozen",
                        help="Append to experiment outputs")
    parser.add_argument("--dataset", type=str, default="hard",
                        help="Dataset quality", choices=["hard", "easy", "rand"])
    parser.add_argument("--run-mode", type=str, default="all",
                        help="Which experiments to run", choices=["all", "vi", "qling", "qlingknownp"])
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of seed values")
    parser.add_argument("--test-mode", help="Simple test mode",
                        action="store_true")
    args = parser.parse_args()
    run_exp(num_seeds=args.num_seeds, extra_exp_prefix=args.extra_exp_prefix, known_P=args.knownp, dataset=args.dataset,
            run_mode=args.run_mode, test_mode=args.test_mode)
