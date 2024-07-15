import numpy as np
from scipy.stats import t


def diebold_mariano_util(e1, e2, loss_function='squared'):
    """
    Perform the Diebold-Mariano test to compare the predictive accuracy of two forecasting models.

    Parameters:
    actual: array-like
        The actual values.
    pred1: array-like
        The predictions from the first model.
    pred2: array-like
        The predictions from the second model.
    loss_function: str
        The type of loss function to use ('squared' for squared errors, 'absolute' for absolute errors).
    h: int
        The forecast horizon.

    Returns:
    dm_stat: float
        The Diebold-Mariano test statistic.
    p_value: float
        The p-value of the test.
    """

    # Compute loss differentials
    if loss_function == 'squared':
        d = e1**2 - e2**2
    elif loss_function == 'absolute':
        d = e1 - e2
    else:
        raise ValueError("Unsupported loss function. Use 'squared' or 'absolute'.")

    # Compute mean and variance of loss differentials
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    # Compute the Diebold-Mariano test statistic
    dm_stat = mean_d / np.sqrt(var_d / len(d))

    # length of forecasting horizon in this case equal length of predicted/actual MELD.
    h = len(d)

    # Compute the p-value
    p_value = 2 * (1 - t.cdf(abs(dm_stat), df=len(d) - 1))

    return dm_stat, p_value


def diebold_mariano_two_models(e1, e2, model_1_name, model_2_name):
    dm_stat, p_value = diebold_mariano_util(e1, e2)
    alpha = 0.05
    if p_value < alpha:
        if dm_stat > 0:
            print(f"Model {model_2_name} is better than Model {model_1_name}. p-value: {p_value}. dm_stat: {dm_stat}")
        else:
            print(f"Model {model_1_name} is better than Model {model_2_name}. p-value: {p_value}. dm_stat: {dm_stat}")
    else:
        print("No significant difference between the two models")


# compare Xgboost with the rest of the models.
def diebold_mariano_test(eval_ress, setup_name):
    print(f"Evaluating diebold_mariano_test in {setup_name}")
    baseline_model_name, baseline_model_e = 'xgboost', None
    for eval_res in eval_ress:
        if eval_res[0] == "xgboost":
            baseline_model_e = eval_res[5]
    for eval_res in eval_ress:
        if eval_res[0] != "xgboost":
            diebold_mariano_two_models(baseline_model_e, eval_res[5], baseline_model_name, eval_res[0])

