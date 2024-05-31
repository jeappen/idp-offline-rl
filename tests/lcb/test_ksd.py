import gymnasium as gym
import pytest

from d3rlpy.lcb.dsd_algos import OfflineVIwDSD
from d3rlpy.lcb.ksd.kernels import exp_hamming_kernel
from d3rlpy.lcb.ksd.ksd import KSD
from d3rlpy.lcb.ksd.model import TabularModel
from d3rlpy.lcb.offvi import OfflineVI
from d3rlpy.lcb.online import ValueIteration
from d3rlpy.lcb.plot import combine_multiple_outputs, plot_all_pickled_outputs, plot_all_pickled_outputs_postfix
from d3rlpy.lcb.util import *


def get_ds_impossible(ds, P_mat):
    """Search for impossible transitions in dataset"""
    prev = None
    for ep in ds.episodes:
        for _ep_t, transition in enumerate(ep):
            o, a, rw, o_next, done = transition.observation, transition.action, transition.reward, \
                transition.next_observation, transition.terminal

            o = int(o[0])
            o_next = int(o_next[0])

            if P_mat[(o, a, o_next)] == 0:
                return _ep_t, o, a, o_next, done, prev
            prev = o, a, o_next, done
    return None


from d3rlpy.lcb.estimators import EstimatePR, OnlineEstimatePR


@pytest.mark.parametrize("num_episodes", [10])
@pytest.mark.parametrize("fixed_model", [True, False])
def test_dataset(num_episodes, fixed_model):
    """Test dataset creation"""
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    policy_fn, vi_info = ValueIteration(env).run()

    creator = CreateOfflineDataset(env, 100, num_episodes, 0)
    ds = creator.generate_full_dataset(policy_fn, 0, gymnasium=True)

    off_vi = OfflineVI(env, ds, estimator=OnlineEstimatePR)
    off_policy, off_V_s, off_info = off_vi.offline_value_iteration(N=ds.dataset_size, Lc=1e-3, V_max=1)

    assert get_ds_impossible(ds, off_vi.P_mat) is None, "Dataset should not contain impossible transitions"

    # Test estimate PR
    est_pr = EstimatePR(env.observation_space.n, env.action_space.n, ds, fixed_model=fixed_model)
    est_pr.estimate()
    assert est_pr.empirical_p_t is not None, "P_mat should be estimated"

    on_est_pr = OnlineEstimatePR(env.observation_space.n, env.action_space.n, ds, fixed_model=fixed_model)
    on_est_pr.estimate()
    assert on_est_pr.current_pt is not None, "P_mat should be estimated"

    if fixed_model:
        assert (on_est_pr.get_batch_mt(0) == est_pr.get_batch_mt(0)).all(), "Mt should be the same at time 0"
        assert (on_est_pr.current_pt == est_pr.get_batch_pt(1)).all(), "Pt should be the same at time 0"
    else:
        assert (on_est_pr.get_batch_mt(0) == est_pr.get_batch_mt(0)).all(), "Mt should be the same at time 0"
        assert (on_est_pr.current_pt == est_pr.get_batch_pt(0, cumulative=True)).all(), \
            "Pt should be the same at time 0"
        on_est_pr.estimate_online(1)  # Update current_t
        assert (on_est_pr.get_batch_mt(1) == est_pr.get_batch_mt(1)).all(), "Mt should be the same at time 1"
        assert (on_est_pr.get_batch_pt(1, cumulative=True) == est_pr.get_batch_pt(1, cumulative=True)).all(), \
            "Pt should be the same cumulative at time 1"
        assert (on_est_pr.get_batch_pt(1) == est_pr.get_batch_pt(1)).all(), "Pt should be the same at time 1"


# @pytest.mark.parametrize("n_epochs", [1])
def test_ksd():
    """Train on STL dataset using DiscreteCQL"""
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    model = TabularModel(env=env)

    samples_q = np.array([[0], [1]])
    s_array = np.array([[0], [1], [2]])
    sa_array = np.array([[0, 0], [1, 1], [2, 2]])
    # model = RBM(m, k, W, bvec, cvec)  # Null model
    ksd = KSD(neg_fun=model.neg, score_fun=model.score,
              kernel_fun=exp_hamming_kernel)  # Use null model
    kappa_vals = ksd.compute_kappa(samples=s_array, sa=sa_array)

    assert ksd is not None, "KSD should be initialized"


@pytest.mark.parametrize("known_p", [True, False])
@pytest.mark.parametrize("fixed_model", [True, False])
def test_dsd_penalty(known_p, fixed_model):
    """Train on FrozenLake dataset using Offline VI with DSD penalty"""
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    policy_fn, vi_info = ValueIteration(env).run()
    creator = CreateOfflineDataset(env, 200, 30, 0)
    ds = creator.generate_full_dataset(policy_fn, 0, gymnasium=True)
    off_vi_dsd = OfflineVIwDSD(env, ds, estimator=OnlineEstimatePR)

    estimator_kwargs = {'fixed_model': fixed_model}

    off_policy_dsd, off_V_s_dsd, off_info_dsd = off_vi_dsd.offline_value_iteration(N=ds.dataset_size,
                                                                                   known_P=known_p,
                                                                                   estimator_kwargs=estimator_kwargs)

    def off_policy_dsd_fn(obs):
        return off_policy_dsd[obs]

    off_vi_dsd_stat = evaluate_policy_on_env(off_policy_dsd_fn, env, gymnasium=True, num_episodes=1000)

    print(off_vi_dsd_stat)


@pytest.mark.parametrize("run_subset, optimal_proportion, fixed_model",
                         [(None, 1.0, True), (['qling'], 0.1, True), (['dsd'], 1, False),
                          (['cql', 'lcb-paper', 'lcb-tuned'], 2, False)])
@pytest.mark.parametrize("eps_optimal_proportion", [0.1])
@pytest.mark.parametrize("estimator", [OnlineEstimatePR])
def test_ksd_hparam_search(run_subset, optimal_proportion, eps_optimal_proportion, estimator, fixed_model,
                           test_plotting=False):
    print(
        f"Testing with run_subset={run_subset}, optimal_proportion={optimal_proportion}, eps_optimal_proportion={eps_optimal_proportion}")
    from d3rlpy.lcb.run import run_all_for_env
    hparam_results = {}
    num_episodes_range = [10]  # np.power(10, np.linspace(1, 4, 7)).astype(int)
    lc_range = [1]  # np.power(10, np.linspace(-3, 3, 13))
    alpha_range = [1]  # np.power(10, np.linspace(-1, 1, 3))
    beta_range = [1]  # np.hstack((np.power(10, np.linspace(-1, 0, 2)), [0]))
    gamma_range = [-1, None]  # np.hstack((np.power(10, np.linspace(-1, 1, 3)), [-1, None]))
    seed_range = list(range(1))
    v_max_range = [1]  # , 50, 100]
    known_P = False
    num_cql_epochs = 2
    dataset_kwargs = {'epsilon': 0.3, 'proportion_optimal': optimal_proportion,
                      "proportion_eps_optimal": eps_optimal_proportion}
    vi_kwargs = {'n_iter': 30}
    env_kwargs = {'map_name': "4x4"}
    qling_kwargs = {'T': 100, 'samples_per_iteration': 2, 'update_q_with_v': True, 'estimator_T': 10}
    estimator_kwargs = {'fixed_model': fixed_model}
    dsd_kwargs = {'alpha_schedule': ('exp', 1.1, 1e-3)}
    alpha_search = True

    for num_episodes in num_episodes_range:
        hparam_results[num_episodes] = run_all_for_env(num_episodes, lc_range=lc_range, alpha_range=alpha_range,
                                                       beta_range=beta_range, gamma_range=gamma_range,
                                                       known_p=known_P, separate_seed_plots=True,
                                                       extra_exp_prefix="test_dsd",
                                                       dataset_kwargs=dataset_kwargs, vi_kwargs=vi_kwargs,
                                                       env_kwargs=env_kwargs,
                                                       seed_range=seed_range, v_max_range=v_max_range,
                                                       include_cql=True, num_cql_epochs=num_cql_epochs,
                                                       estimator=estimator,
                                                       include_qling=True, qling_kwargs=qling_kwargs,
                                                       run_subset=run_subset, estimator_kwargs=estimator_kwargs,
                                                       dsd_kwargs=dsd_kwargs, alpha_search=alpha_search,
                                                       skip_plotting=not test_plotting)

    assert hparam_results is not None, "Should have results"
    print(f"Saving results for {num_episodes} episodes")
    pickle_filename = f'hparam_results_testksd_{num_episodes}.pkl'
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(hparam_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if test_plotting:
        combine_multiple_outputs([hparam_results[num_episodes]])
        if run_subset is None:
            test_subset = [cql_extension, value_iter_extension, vanilla_extension, dsd_extension]
            # Plot subset of all
            combine_multiple_outputs([hparam_results[num_episodes]], extra_exp_prefix="test_plot_subset",
                                     run_subset=test_subset)
            test_subset2 = test_subset + ['lcb-paper-estp', 'dsd-estp']
            plot_all_pickled_outputs_postfix([pickle_filename, pickle_filename],
                                             postfix_groups=[None, ('estp', 'Estimated P')],
                                             extra_exp_prefix="test_plot_postfix_subset", run_subset=test_subset2)
        plot_all_pickled_outputs([pickle_filename, pickle_filename], extra_exp_prefix="test_plot_all")
        plot_all_pickled_outputs_postfix([pickle_filename, pickle_filename],
                                         postfix_groups=[None, ('estp', 'Estimated P')],
                                         extra_exp_prefix="test_plot_postfix")
