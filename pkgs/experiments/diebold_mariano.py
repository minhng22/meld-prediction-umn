import numpy as np
from scipy.stats import norm


def diebold_mariano_util(e1, e2, loss_fn='two-sided'):
    """
        Diebold-Mariano test for predictive accuracy.

        Parameters:
        e1 (np.array): Forecast errors from model 1.
        e2 (np.array): Forecast errors from model 2.
        h (int): Forecast horizon (default is 1).
        alternative (str): Type of alternative hypothesis, 'two-sided', 'less', or 'greater'.

        Returns:
        DM_stat (float): Diebold-Mariano test statistic.
        p_value (float): p-value of the test.
        """
    e1 = e1.ravel()
    e2 = e2.ravel()

    d = e1 - e2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    DM_stat = mean_d / np.sqrt(var_d / len(d))

    if loss_fn == 'two-sided':
        p_value = 2 * norm.cdf(-abs(DM_stat))
    elif loss_fn == 'less':
        p_value = norm.cdf(DM_stat)
    elif loss_fn == 'greater':
        p_value = 1 - norm.cdf(DM_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return DM_stat, p_value


def diebold_mariano_two_models(e1, e2, model_1_name, model_2_name):
    dm_stat, p_value = diebold_mariano_util(e1, e2)
    alpha = 0.05
    if p_value < alpha:
        print(f"Significant diff between {model_1_name} and {model_2_name}. p-value: {p_value}")
    else:
        print(f"No significant diff between {model_1_name} and {model_2_name}. p-value: {p_value}")


# compare Xgboost with the rest of the models.
def diebold_mariano_test(eval_ress, setup_name):
    print(f"Evaluating diebold_mariano_test in {setup_name}")
    baseline_model_name, baseline_model_e = 'xgboost', None
    for eval_res in eval_ress:
        if eval_res[0] == "xgboost":
            baseline_model_e = eval_res[6]
    for eval_res in eval_ress:
        if eval_res[0] != "xgboost":
            diebold_mariano_two_models(baseline_model_e, eval_res[6], baseline_model_name, eval_res[0])

