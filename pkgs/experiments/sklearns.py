import os
import time

import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.experimental import enable_halving_search_cv # required by sklearn
from sklearn.model_selection import HalvingGridSearchCV

from pkgs.data.commons import inverse_scale_ops
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.plot import plot_box, plot_line, analyze_timestep_rmse, analyze_ci_and_pi
from pkgs.commons import model_save_path
from pkgs.experiments.commons import sklearn_find_better_model
from pkgs.models.commons import get_sklearn_model


def exp_sklearn_model(dataset: SlidingWindowDataset, model_name: str, num_obs, num_pred, num_feature_input, num_feature_output, compare_with_existing_best_model):
    s = time.time()

    model_path = model_save_path(num_obs, num_pred) + "/" + model_name + ".pkl"

    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
        if compare_with_existing_best_model:
            best_model = sklearn_find_better_model(
                test_ips=dataset.get_test_ips(), test_ip_meld=dataset.get_test_ip_meld(),
                original_meld_test=dataset.get_original_meld_test(),
                scaler=dataset.meld_sc, num_obs=num_obs, num_pred=num_pred, num_feature_input=num_feature_input,
                model_a=run_sklearn_model(dataset, model_name, num_obs, num_pred, num_feature_input, num_feature_output),
                model_b=best_model, model_name=model_name)
    else:
        best_model = run_sklearn_model(dataset, model_name, num_obs, num_pred, num_feature_input, num_feature_output)


    joblib.dump(best_model, model_path)

    print(f"best model: {best_model}")

    # evaluate on test set
    print("evaluating on test data")
    test_ips = np.reshape(dataset.get_test_ips(), (-1, num_obs * num_feature_input))  # reshape data into 2D
    tests_ops = best_model.predict(test_ips)

    tests_ops_full = inverse_scale_ops(dataset.get_test_ip_meld(), tests_ops, dataset.meld_sc, num_obs, num_pred)
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
        dataset.get_generalize_ip_meld(), generalizes_ops, dataset.meld_sc, num_obs, num_pred)
    generalizes_ops = generalizes_ops_full[:, num_obs:]
    generalizes = dataset.get_original_meld_generalize()[:, num_obs:]

    print(f"R-square is: {r2_score(generalizes, generalizes_ops):.4f}")
    print(
        f"RMSE is: {mean_squared_error(generalizes, generalizes_ops, squared=False):.4f}"
    )

    analyze_timestep_rmse(generalizes, generalizes_ops, "generalize", model_name, num_obs, num_pred)
    analyze_ci_and_pi(generalizes, generalizes_ops, "generalize", model_name, num_obs, num_pred)

    plot_name = f"{model_name} R-square :{round(r2_score(generalizes, generalizes_ops), 2)} " \
                f"RMSE {round(mean_squared_error(generalizes, generalizes_ops, squared=False), 2)}"
    plot_box(generalizes, generalizes_ops, plot_name, model_name, num_obs, num_pred, "generalize")
    plot_line(dataset.get_original_meld_generalize(), generalizes_ops_full, plot_name, model_name, num_obs, num_pred, "generalize")

    print(f"total experiment time: {time.time() - s} seconds")


def run_sklearn_model(dataset: SlidingWindowDataset, model_name: str, num_obs, num_pred, num_feature_input, num_feature_output):
    params = get_sklearn_model_params(model_name)
    model = get_sklearn_model(model_name)

    grid_search = HalvingGridSearchCV(model, params)

    train_ips = np.reshape(dataset.get_train_ips(),
                           (-1, num_obs * num_feature_input))
    train_targets = np.reshape(
        dataset.get_target_ips(), (-1, num_pred * num_feature_output))

    if num_obs == 1:
        train_ips = np.ravel(train_ips)
    if num_pred == 1:
        train_targets = np.ravel(train_targets)

    print(f"train shape: {train_ips.shape} {train_targets.shape}")

    grid_search.fit(train_ips, train_targets)

    return grid_search.best_estimator_


def get_sklearn_model_params(model_name: str):
    if model_name == "evr":
        return {
            'n_estimators': [100, 200, 300],
            'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
            'max_depth': [None, 5, 10],
            'max_features': ['sqrt', 'log2'],
        }
    if model_name == "rfr":
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }