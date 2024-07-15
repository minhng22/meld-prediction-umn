import numpy as np
from scipy.stats import t
from statsmodels.tsa.stattools import acf


def diebold_mariano_util(e1, e2, h):
    """
        Diebold-Mariano test for predictive accuracy.

        Parameters:
        e1 (np.array): Forecast errors from model 1.
        e2 (np.array): Forecast errors from model 2.
        h (int): Forecast horizon (default is 1).

        Returns:
        dm_stat (float): Diebold-Mariano test statistic.
        p_value (float): p-value of the test.
    """
    e1 = e1.ravel()
    e2 = e2.ravel()

    d = np.abs(e1) ** 2 - np.abs(e2) ** 2  # two-sided
    d_mean = np.mean(d)
    n = len(d)

    # Compute auto_covariance of d up to lag h-1
    acov_d = acf(d, fft=False, nlags=h - 1) * np.var(d) * n / (n - 1)

    # Variance of d-bar
    var_d_bar = (1 / n) * (acov_d[0] + 2 * np.sum(acov_d[1:]))

    dm_stat = d_mean / np.sqrt(var_d_bar)
    p_value = 2 * t.cdf(-np.abs(dm_stat), df=n - 1)

    return dm_stat, p_value


def diebold_mariano_two_models(e1, e2, model_1_name, model_2_name, h):
    dm_stat, p_value = diebold_mariano_util(e1, e2, h)
    alpha = 0.05
    if p_value < alpha:
        print(f"Significant diff between {model_1_name} and {model_2_name}. p-value: {p_value}")
    else:
        print(f"No significant diff between {model_1_name} and {model_2_name}. p-value: {p_value}")


# compare Xgboost with the rest of the models.
def diebold_mariano_test(eval_ress, setup_name, h):
    print(f"Evaluating diebold_mariano_test in {setup_name}. f = {h}")
    baseline_model_name, baseline_model_e = 'xgboost', None
    for eval_res in eval_ress:
        if eval_res[0] == "xgboost":
            baseline_model_e = eval_res[6]
    for eval_res in eval_ress:
        if eval_res[0] != "xgboost":
            diebold_mariano_two_models(baseline_model_e, eval_res[6], baseline_model_name, eval_res[0], h)

