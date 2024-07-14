import numpy as np
from scipy.stats import t


def diebold_mariano_util(e1, e2):
    d = e1 - e2
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - t.cdf(abs(dm_stat), df=len(d) - 1))
    return dm_stat, p_value


def diebold_mariano_two_models(e1, e2, model_1_name, model_2_name):
    dm_stat, p_value = diebold_mariano_util(e1, e2)
    alpha = 0.05
    if p_value < alpha:
        if dm_stat > 0:
            print(f"Model {model_2_name} is significantly better than Model {model_1_name}")
        else:
            print(f"Model {model_1_name} is significantly better than Model {model_2_name}")
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

