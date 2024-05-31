import gzip  # for saving the results

from d3rlpy.envs.env import MDPToolboxEnv, randDense, deepSea
from d3rlpy.envs.mdp_environments import make_mdp_env_and_get_PR
from d3rlpy.lcb.estimators import OnlineEstimatePR
from d3rlpy.lcb.run import run_all_for_env
from d3rlpy.lcb.util import *


def create_mdp(S_dim=4, A_dim=4, abs_rw=False, scale_rw=1, beta_mode=False, env="random", **env_kwargs):
    if beta_mode:
        print("Using beta distribution for random MDPs")
    if env == "random":
        spec_id = f"Random_{S_dim}x{A_dim}"
        P, R = randDense(S_dim, A_dim, beta_mode=beta_mode)
    elif env == "deepsea":
        spec_id = f"DeepSea_{S_dim}"
        P, R = deepSea(S_dim)
    elif env == "widenarrow":
        (P, R), info = make_mdp_env_and_get_PR('widenarrow')
        spec_id = f"WideNarrow_N{info['N']}_W{info['W']}"
    elif env == "prior":
        (P, R), info = make_mdp_env_and_get_PR('prior')
        spec_id = f"PriorMDP_{info['Ns']}x{info['Na']}"
    if abs_rw:
        # Change from -1, 1 to 0, 1 rewards
        R = np.abs(R)
        spec_id = f"{spec_id}_Abs"
    if scale_rw != 1:
        R = R * scale_rw
        spec_id = f"{spec_id}_X{scale_rw}"

    # For positivity constraint
    P_clip = P.clip(1e-6)  # clip to avoid 0 probabilities, lower the number, more DSD unbounded errors
    P_norm = P_clip / np.expand_dims(P_clip.sum(axis=-1), 2)  # normalize

    # Check that each row sums to 1
    consistency_test = (P_norm.sum(axis=-1) + 1e-5).astype(int) == 1
    assert (consistency_test).all(), "P_norm should sum to 1"

    return MDPToolboxEnv(P_norm, R, spec_id=spec_id, **env_kwargs)


def run_exp(fair_eval_mode=True, num_seeds=10, extra_exp_prefix="run_random", known_P=False, dataset="hard",
            run_mode="all", beta_mode=False, test_mode=False, s_dim=64, a_dim=64, env_name="random"):
    hparam_results = {}
    num_episodes_range = np.power(10, np.linspace(1, 4, 7)).astype(int)[2:]
    lc_range = np.power(10, np.linspace(-3, 3, 13))
    alpha_range = np.power(10, np.linspace(-1, 1, 3))
    beta_range = np.hstack((np.power(10, np.linspace(-1, 0, 2)), [0], -1 * np.power(10, np.linspace(-1, 0, 2))))
    gamma_range = np.hstack((np.power(10, np.linspace(-1, 1, 3)), [None]))
    eta_range = np.hstack((np.power(10, np.linspace(-1, 0, 3)), [0.5]))
    cb_range = np.power(10, np.linspace(-2, 4, 7))
    seed_range = list(range(num_seeds))
    v_max_range = [1, 50, 100]
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

    max_episode_steps = s_dim + 1 if env_name == "deepsea" else 100
    env_kwargs = {'S_dim': s_dim, 'A_dim': a_dim, 'beta_mode': beta_mode, 'env': env_name,
                  'max_episode_steps': max_episode_steps}
    qling_kwargs = {'T': 2 * (10 ** 5), 'samples_per_iteration': 1}
    dsd_kwargs = {'alpha_schedule': ('exp', 1.1, 1e-3)}
    alpha_search = True  # Search over alpha_range for all penalties
    sample_env = create_mdp(**env_kwargs)
    if test_mode:
        # For quick testing
        num_episodes_range = [10]
        env_kwargs['max_episode_steps'] = 10

    for num_episodes in num_episodes_range:
        print(f"Running {num_episodes} episodes")
        final_exp_prefix = f"{'test_' if test_mode else ''}{sample_env.spec.name}_{run_mode}_{dataset}_{extra_exp_prefix}"
        hparam_results[num_episodes] = run_all_for_env(num_episodes, lc_range=lc_range, alpha_range=alpha_range,
                                                       beta_range=beta_range, gamma_range=gamma_range,
                                                       known_p=known_P, separate_seed_plots=True,
                                                       extra_exp_prefix=final_exp_prefix,
                                                       vi_kwargs=vi_kwargs,
                                                       env_creator=create_mdp, fair_eval_mode=fair_eval_mode,
                                                       dataset_kwargs=dataset_kwargs, env_kwargs=env_kwargs,
                                                       seed_range=seed_range, v_max_range=v_max_range, include_cql=True,
                                                       include_qling=True, run_subset=run_subset,
                                                       estimator=OnlineEstimatePR, qling_kwargs=qling_kwargs,
                                                       eta_range=eta_range, cb_range=cb_range,
                                                       dsd_kwargs=dsd_kwargs, alpha_search=alpha_search,
                                                       test_mode=test_mode)
        print(f"Saving results for {num_episodes} episodes")
        # Save compressed result file
        with gzip.open(f'data/hparam_results_{final_exp_prefix}_{num_episodes}.pkl.gz',
                       'wb') as handle:
            pickle.dump(hparam_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return hparam_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="deepsea",
                        help="Name of MDP environment", choices=["random", "deepsea", "widenarrow", "prior"])
    parser.add_argument("--knownp", help="Use ground truth transition model",
                        action="store_true")
    parser.add_argument("--extra_exp_prefix", type=str, default="run_mdp_loweps",
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
    parser.add_argument("--num-states", type=int, default=32,
                        help="Number of states")
    parser.add_argument("--num-actions", type=int, default=64,
                        help="Number of actions")
    args = parser.parse_args()
    run_exp(num_seeds=args.num_seeds, extra_exp_prefix=args.extra_exp_prefix, known_P=args.knownp, dataset=args.dataset,
            run_mode=args.run_mode, beta_mode=args.beta, test_mode=args.test_mode, env_name=args.env,
            s_dim=args.num_states, a_dim=args.num_actions)
