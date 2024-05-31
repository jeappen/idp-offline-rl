import gzip

from d3rlpy.envs.env import PortfolioActionEnv
from d3rlpy.lcb.estimators import OnlineEstimatePR
from d3rlpy.lcb.run import run_all_for_env
from d3rlpy.lcb.util import *


def port_env_creator(num_assets=3, asset_discretization_steps=5,
                     action_discretization_steps=5,
                     price_lb=50.0, price_ub=55.0,
                     max_steps=200, **kwargs):
    return PortfolioActionEnv(num_assets, 0.0,
                              state_transition_kernel=None,
                              asset_discretization_steps=asset_discretization_steps,
                              action_discretization_steps=action_discretization_steps,
                              price_lb=price_lb,
                              price_ub=price_ub,
                              max_steps=max_steps)


def run_exp(num_episodes_range=None, max_ep_length=200, num_seeds=5,
            extra_exp_prefix='port_high_rw_var_sparser_200_action_dep',
            known_P=False, dataset="hard", run_mode="all", test_mode=False,
            beta_mode=False):
    if num_episodes_range is None:
        num_episodes_range = np.power(10, np.linspace(1, 4, 7)).astype(int)
    lc_range = np.power(10, np.linspace(-3, 3, 5))
    alpha_range = np.power(10, np.linspace(-1, 1, 3))
    beta_range = np.hstack((np.power(10, np.linspace(-1, 0, 2)), [0]))
    gamma_range = np.hstack((np.power(10, np.linspace(-1, 1, 3)), [-1, None]))

    eta_range = np.hstack((np.power(10, np.linspace(-1, 0, 3)), [0.5]))
    cb_range = np.power(10, np.linspace(-2, 4, 4))
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

    seed_range = list(range(num_seeds))
    v_max_range = [50, None]
    env_kwargs = {"num_assets": 3, "asset_discretization_steps": 5,
                  "action_discretization_steps": 5,
                  "price_lb": 50.0, "price_ub": 100.0,  # Change to 50.0, 55.0 for low variance
                  "max_steps": 200, 'beta_mode': beta_mode}
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
    qling_kwargs = {'T': 6 * (10 ** 4), 'samples_per_iteration': 1}
    sample_env = port_env_creator(**env_kwargs)

    if test_mode:
        # For quick testing
        num_episodes_range = [10]

    for num_episodes in num_episodes_range:
        print(f"\n\nRUNNING {num_episodes} trial\n\n")
        final_exp_prefix = f"{'test_' if test_mode else ''}{sample_env.spec.name}_{run_mode}_{dataset}_{extra_exp_prefix}"
        hparam_results = run_all_for_env(num_episodes, lc_range=lc_range, alpha_range=alpha_range,
                                         beta_range=beta_range, gamma_range=gamma_range,
                                         env_creator=port_env_creator, max_ep_length=max_ep_length,
                                         separate_seed_plots=True, plot_perf_interval=8,
                                         env_kwargs=env_kwargs,
                                         known_p=known_P, seed_range=seed_range, v_max_range=v_max_range,
                                         extra_exp_prefix=final_exp_prefix, include_cql=True,
                                         include_qling=True, run_subset=run_subset,
                                         estimator=OnlineEstimatePR, dataset_kwargs=dataset_kwargs,
                                         qling_kwargs=qling_kwargs, eta_range=eta_range, cb_range=cb_range,
                                         test_mode=test_mode)

        # Save compressed result file
        with gzip.open(f'data/hparam_results_{final_exp_prefix}_{num_episodes}.pkl.gz', 'wb') as handle:
            pickle.dump(hparam_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return hparam_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--knownp", help="Use ground truth transition model",
                        action="store_true")
    parser.add_argument("--extra-exp-prefix", type=str, default="run_port",
                        help="Append to experiment outputs")
    parser.add_argument("--dataset", type=str, default="hard",
                        help="Dataset quality", choices=["hard", "easy", "rand"])
    parser.add_argument("--run-mode", type=str, default="all",
                        help="Which experiments to run", choices=["all", "vi", "qling", "qlingknownp"])
    parser.add_argument("--beta", help="Use beta distribution for random MDPs",
                        action="store_true")
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of seed values")
    parser.add_argument("--test-mode", help="Simple test mode",
                        action="store_true")
    args = parser.parse_args()
    run_exp(num_seeds=args.num_seeds, extra_exp_prefix=args.extra_exp_prefix, known_P=args.knownp, dataset=args.dataset,
            run_mode=args.run_mode, beta_mode=args.beta, test_mode=args.test_mode)
