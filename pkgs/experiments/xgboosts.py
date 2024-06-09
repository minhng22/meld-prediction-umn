import os
import time

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_halving_search_cv  # required by sklearn
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBRegressor

from pkgs.commons import model_save_path
from pkgs.data.commons import inverse_scale_ops
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.plot import analyze_timestep_rmse, analyze_ci_and_pi, plot_box, plot_line
from pkgs.experiments.commons import eval_and_plot_sklearn_model, sklearn_find_better_model


def run_xgboost_model(dataset: SlidingWindowDataset, num_obs, num_pred, num_feature_input,
                      num_feature_output):
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1]
    }
    model = XGBRegressor()

    grid_search = HalvingGridSearchCV(
        estimator=model,
        param_grid=param_grid,
        factor=10,
        cv=2
    )

    train_ips = np.reshape(dataset.get_train_ips(), (-1, num_obs * num_feature_input))
    train_targets = np.reshape(dataset.get_target_ips(), (-1, num_pred * num_feature_output))

    if num_obs == 1:
        train_ips = np.ravel(train_ips)
    if num_pred == 1:
        train_targets = np.ravel(train_targets)

    print(f"train shape: {train_ips.shape} {train_targets.shape}")

    grid_search.fit(train_ips, train_targets)

    return grid_search.best_estimator_


def exp_xgboost_model(dataset: SlidingWindowDataset, model_name: str, num_obs, num_pred, num_feature_input,
                      num_feature_output, compare_with_existing_best_model):
    s = time.time()

    model_path = f"{model_save_path(num_obs, num_pred)}/{model_name}.json"
    if os.path.exists(model_path):
        best_model = XGBRegressor()
        best_model.load_model(model_path)
        if compare_with_existing_best_model:
            best_model = sklearn_find_better_model(
                test_ips=dataset.get_test_ips(), test_ip_meld=dataset.get_test_ip_meld(),
                original_meld_test=dataset.get_original_meld_test(),
                scaler=dataset.meld_sc, num_obs=num_obs, num_pred=num_pred, num_feature_input=num_feature_input,
                model_a=run_xgboost_model(dataset, num_obs, num_pred, num_feature_input, num_feature_output),
                model_b=best_model, model_name=model_name)
            best_model.save_model(model_path)
    else:
        best_model = run_xgboost_model(dataset, num_obs, num_pred, num_feature_input, num_feature_output)
        best_model.save_model(model_path)

    print(f"best model: {best_model}")

    eval_and_plot_sklearn_model(
        dataset.get_test_ips(), dataset.get_test_ip_meld(), dataset.get_original_meld_test(), dataset.meld_sc,
        model_name, num_obs, num_pred, num_feature_input, best_model, "test")
    eval_and_plot_sklearn_model(
        dataset.get_generalize_ips(), dataset.get_generalize_ip_meld(), dataset.get_original_meld_generalize(),
        dataset.meld_sc,
        model_name, num_obs, num_pred, num_feature_input, best_model, "generalize"
    )

    print(f"total experiment time: {time.time() - s} seconds")
