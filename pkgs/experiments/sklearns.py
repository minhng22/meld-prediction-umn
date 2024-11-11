import os
import time

import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # required by sklearn
from sklearn.model_selection import HalvingGridSearchCV

from pkgs.commons import sklearn_model_path
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.experiments.commons import sklearn_find_better_model
from pkgs.experiments.evaluate import sklearn_model_eval_and_plot
from pkgs.models.commons import get_sklearn_model


def exp_sklearn_model(dataset: SlidingWindowDataset, model_name: str, num_obs, num_pred, num_feature_input, num_feature_output, compare_with_existing_best_model):
    s = time.time()

    model_path = sklearn_model_path(num_obs, num_pred, model_name)

    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
        if compare_with_existing_best_model:
            best_model = sklearn_find_better_model(
                test_ips=dataset.get_test_ips(),
                original_meld_test=dataset.get_original_meld_test(),
                scaler=dataset.meld_sc, num_obs=num_obs, num_pred=num_pred, num_feature_input=num_feature_input,
                model_a=run_sklearn_model(dataset, model_name, num_obs, num_pred, num_feature_input, num_feature_output),
                model_b=best_model, model_name=model_name)
            joblib.dump(best_model, model_path)
    else:
        best_model = run_sklearn_model(dataset, model_name, num_obs, num_pred, num_feature_input, num_feature_output)
        joblib.dump(best_model, model_path)

    print(f"best model: {best_model}")

    sklearn_model_eval_and_plot(
        dataset.get_test_ips(), dataset.get_original_meld_test(), dataset.meld_sc,
        model_name, num_obs, num_pred, num_feature_input, best_model, "test")
    sklearn_model_eval_and_plot(
        dataset.get_generalize_ips(), dataset.get_original_meld_generalize(),
        dataset.meld_sc,
        model_name, num_obs, num_pred, num_feature_input, best_model, "generalize"
    )

    print(f"total experiment time: {time.time() - s} seconds")


def run_sklearn_model(dataset: SlidingWindowDataset, model_name: str, num_obs, num_pred, num_feature_input, num_feature_output):
    params = get_sklearn_model_params(model_name)
    model = get_sklearn_model(model_name)

    grid_search = HalvingGridSearchCV(
        estimator=model,
        param_grid=params,
        factor=10,
        cv=2,
        verbose=3,
        n_jobs=10,
    )
    # try RandomizedSearchCV() if computing resource is limited

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