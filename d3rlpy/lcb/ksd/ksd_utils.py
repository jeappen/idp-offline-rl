from .util import *


def ksd_bootstrap(kappa_vals, n_boot=5000):
    """
    Implements the multinomial bootstrap method for independent samples in
        Liu et al. (2016).

    Args:
        kappa_vals: array(n, n), pre-computed KSD kernel matrix.
        n_boot: int, number of bootstrap samples.

    Returns:
        boot_samples: array(n_boot), bootstrap samples.
    """
    n = kappa_vals.shape[0]  # Sample size
    kappa_vals = kappa_vals - np.diag(np.diag(kappa_vals))  # Remove diagonal

    # Bootstrap samples for KSD estimates
    boot_samples = np.zeros(n_boot)

    for j in range(n_boot):
        wvec = (rand.multinomial(n=n, pvals=np.ones(n) / n) - 1.) / n
        boot_samples[j] = wvec.dot(kappa_vals).dot(wvec)

    return boot_samples


def ksd_est(kappa_vals_list):
    """
    Given a list of pre-computed kappa values, compute the U- and V-statistics
        estimates for KSD.

    Args:
        n: int, sample size.
        kappa_vals_list: list of array((n, n)), list of pre-computed kappa's.

    Returns: (all lists have same length as kappa_vals_list)
        ustats: list, U-statistics KSD estimate.
        vstats: list, V-statistics KSD estimate.
    """
    n = kappa_vals_list[0].shape[0]  # Sample size
    assert all(kappa_vals.shape == (n, n) for kappa_vals in kappa_vals_list)

    ustats = np.zeros(len(kappa_vals_list))  # U-stat
    vstats = np.zeros(len(kappa_vals_list))  # U-stat

    for i, kappa_vals in enumerate(kappa_vals_list):
        diag_vals = np.diag(np.diag(kappa_vals))  # (n, n) diagonal matrix
        ustats[i] = np.sum(kappa_vals - diag_vals) / (n * (n - 1))  # U-stat
        vstats[i] = np.sum(kappa_vals) / (n ** 2)  # V-stat

    return ustats, vstats


def ksd_est_biased(kappa_vals_list):
    """
    Given a list of pre-computed kappa values, compute the (biased) statistic being
    the mean of all kappa values.

    Args:
        n: int, sample size.
        kappa_vals_list: list of array((n, n)), list of pre-computed kappa's.

    Returns: (all lists have same length as kappa_vals_list)
        stats: list, mean-statistics KSD estimate.
    """
    n = kappa_vals_list[0].shape[0]  # Sample size
    assert all(kappa_vals.shape == (n, n) for kappa_vals in kappa_vals_list)

    ustats = np.zeros(len(kappa_vals_list))  # U-stat

    for i, kappa_vals in enumerate(kappa_vals_list):
        ustats[i] = np.mean(kappa_vals)  # U-stat

    return ustats


def ksd_est_block(kappa, block_size_list):
    """
    Given a list of pre-computed kappa values, compute the U- and V-statistics
        estimates for KSD on different block sizes.

    Args:
        kappa: pre-computed kappa
        block_size_list: list of block sizes


    Returns: (all lists have same length as kappa_vals_list)
        ustats: list, U-statistics KSD estimate.
    """
    n = kappa.shape[0]  # Sample size

    ustats = np.zeros(len(block_size_list))  # U-stat

    # Compute of each block size
    curr_idx = 0
    for i, block_size in enumerate(block_size_list):
        kappa_block = kappa[curr_idx:curr_idx + block_size, curr_idx:curr_idx + block_size]
        diag_vals = np.diag(np.diag(kappa_block))  # (block_size, block_size) diagonal matrix
        ustats[i] = np.sum(kappa_block - diag_vals) / (block_size * (block_size - 1))
        curr_idx += block_size

    return ustats


def ksd_boot(kappa_vals_list, quantile=.95, n_boot=1000):
    """
    Given a list of pre-computed kappa values, compute
        the bootstrap sampling distribution for KSD; and
        the critical threshold of the KSD test obtained by taking
            some quantile of the bootstrap sampling distribution.

    Args:
        n: int, sample size.
        kappa_vals_list: list of array((n, n)), list of pre-computed kappa's.

    Returns: (all lists have same length as kappa_vals_list)
        ksd_boot_list: list of lists, samples from the bootstrap distribution.
        boot_thres: list, critical threshold for KSD test.
    """
    # Bootstrap estimates
    boot_list = [ksd_bootstrap(kappa_vals, n_boot=n_boot)
                 for kappa_vals in kappa_vals_list]

    # Compute quantile of bootstrap sampling distribution
    boot_thres = [np.percentile(boot, 100. * quantile) for boot in boot_list]

    return boot_list, boot_thres


def ksd_pvalue(boot_list, ustats):
    """
    Computes the p-value of the KSD test.

    Args:
        boot_list: list, list of bootstrap statistics.
        ustats: float, value of computed test statistic.
    """
    assert len(boot_list) == len(ustats)

    pvals = [np.mean(boot >= ustats)
             for boot, ustats in zip(boot_list, ustats)]

    return pvals
