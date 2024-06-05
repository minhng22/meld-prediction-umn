import os
import time

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_halving_search_cv # required by sklearn
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBRegressor

from pkgs.commons import model_save_path
from pkgs.data.commons import inverse_scale_ops
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.plot import analyze_timestep_rmse, analyze_ci_and_pi, plot_box, plot_line


def run_xgboost_model(dataset: SlidingWindowDataset, model_name: str, num_obs, num_pred, num_feature_input, num_feature_output):
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1]
    }
    model = XGBRegressor()

    model_path = model_save_path(num_obs, num_pred) + "/" + model_name + ".json"
    if os.path.exists(model_path):
        model.load_model(model_path)
        return model

    grid_search = HalvingGridSearchCV(model, param_grid, cv=10)

    train_ips = np.reshape(dataset.get_train_ips(), (-1, num_obs * num_feature_input))
    train_targets = np.reshape(dataset.get_target_ips(), (-1, num_pred * num_feature_output))

    if num_obs == 1:
        train_ips = np.ravel(train_ips)
    if num_pred == 1:
        train_targets = np.ravel(train_targets)

    print(f"train shape: {train_ips.shape} {train_targets.shape}")

    grid_search.fit(train_ips, train_targets)

    return grid_search.best_estimator_


def exp_xgboost_model(dataset: SlidingWindowDataset, model_name: str, num_obs, num_pred, num_feature_input, num_feature_output):
    s = time.time()

    best_model = run_xgboost_model(dataset, model_name, num_obs, num_pred, num_feature_input, num_feature_output)

    print(f"best model: {best_model}")

    # evaluate on test set
    print("evaluating on test data")
    test_ips = np.reshape(dataset.get_test_ips(), (-1, num_obs * num_feature_input))  # reshape data into 2D
    tests_ops = best_model.predict(test_ips)

    tests_ops_full = inverse_scale_ops(dataset.get_test_ip_meld(), tests_ops, dataset.meld_sc)
    tests_ops = tests_ops_full[:, num_obs:]
    tests = dataset.get_original_meld_test()[:, num_obs:]

    print(f"test shape: {tests.shape} {tests_ops.shape}")
    print(f"R-square is: {r2_score(tests, tests_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(tests, tests_ops, squared=False):.4f}")

    analyze_timestep_rmse(tests, tests_ops, "test", model_name, num_obs, num_pred)
    analyze_ci_and_pi(tests, tests_ops, "test", model_name, num_obs, num_pred)

    plot_name = f"{model_name} R-square :{round(r2_score(tests, tests_ops), 2)} RMSE {round(mean_squared_error(tests, tests_ops, squared=False), 2)}"

    plot_box(tests, tests_ops, plot_name, model_name, num_obs, num_pred, "test")
    plot_line(dataset.get_original_meld_test(),
              tests_ops_full, plot_name, model_name, num_obs, num_pred, "test")

    # evaluate on generalization set
    print("evaluating on generalize data")
    generalizes_ips = np.reshape(
        dataset.get_generalize_ips(), (-1, num_obs * num_feature_input))  # reshape data into 2D
    generalizes_ops = best_model.predict(generalizes_ips)

    generalizes_ops_full = inverse_scale_ops(
        dataset.get_generalize_ip_meld(), generalizes_ops, dataset.meld_sc)
    generalizes_ops = generalizes_ops_full[:, num_obs:]
    generalizes = dataset.get_original_meld_generalize()[:, num_obs:]

    print(f"R-square is: {r2_score(generalizes, generalizes_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(generalizes, generalizes_ops, squared=False):.4f}"
    )

    analyze_timestep_rmse(generalizes, generalizes_ops, "generalize", model_name)
    analyze_ci_and_pi(generalizes, generalizes_ops, "generalize", model_name)

    plot_name = f"{model_name} R-square :{round(r2_score(generalizes, generalizes_ops), 2)} " \
                f"RMSE {round(mean_squared_error(generalizes, generalizes_ops, squared=False), 2)}"
    plot_box(generalizes, generalizes_ops, plot_name, model_name, "generalize")
    plot_line(dataset.get_original_meld_generalize(),
              generalizes_ops_full, plot_name, model_name, "generalize")

    model_path = f"{model_save_path(num_obs, num_pred)}/{model_name}.json"
    best_model.save_model(model_path)
    print(f"total experiment time: {time.time() - s} seconds")