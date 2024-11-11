import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pkgs.commons import input_path, model_save_path, models_to_run, preprocessed_train_set_data_path, \
    preprocessed_test_set_data_path, preprocessed_generalize_set_data_path, torch_model_path
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.harvest import harvest_data_with_interpolate
from pkgs.experiments.commons import rnn_find_better_model
from pkgs.experiments.evaluate import rnn_model_eval_and_plot
from pkgs.experiments.linears import exp_linear_model
from pkgs.experiments.optunas import ex_optuna
from pkgs.experiments.sklearns import exp_sklearn_model
from pkgs.experiments.xgboosts import exp_xgboost_model


# if compare_with_existing_best_model is True, we will run the model again no matter if a trained model exists.
# If a trained model exists, we will compare the new model with the existing model and save the better one.
def run_exp(num_obs, num_pred, real_data_ratio, generalize_ratio, interpolate_amount, to_run_models,
            compare_with_existing_best_model):
    batch_size = 256
    device = torch.device("cpu")
    n_trials = 1
    num_feature_input, num_feature_output = 2, 1 # MELD and timestamp

    print(f"pre-processing data, experimenting on obs {num_obs} pred {num_pred}")
    s = time.time()
    df = pd.read_csv(input_path)

    if not os.path.exists(preprocessed_train_set_data_path(num_obs, num_pred)):
        print("getting new data")
        exp_trains, exp_tests, exp_generalizes = harvest_data_with_interpolate(
            df, num_obs + num_pred, real_data_ratio, generalize_ratio, interpolate_amount)
        np.save(preprocessed_train_set_data_path(num_obs, num_pred), exp_trains)
        np.save(preprocessed_test_set_data_path(num_obs, num_pred), exp_tests)
        np.save(preprocessed_generalize_set_data_path(num_obs, num_pred), exp_generalizes)
    else:
        exp_trains = np.load(preprocessed_train_set_data_path(num_obs, num_pred))
        exp_tests = np.load(preprocessed_test_set_data_path(num_obs, num_pred))
        exp_generalizes = np.load(preprocessed_generalize_set_data_path(num_obs, num_pred))

    print(f"preprocessing data takes {time.time() - s} seconds")

    for model_name in to_run_models:
        print("=====================================")
        print(f"exp on model {model_name}")

        dataset = SlidingWindowDataset()
        dataset.setup_full(exp_trains, exp_tests, exp_generalizes, num_obs, num_pred)

        print("finding best model")
        s = time.time()

        if model_name in ["evr", "rfr"]:
            exp_sklearn_model(dataset, model_name, num_obs, num_pred, num_feature_input, num_feature_output, compare_with_existing_best_model)
            continue

        if model_name == "xgboost":
            exp_xgboost_model(dataset, model_name, num_obs, num_pred, num_feature_input, num_feature_output, compare_with_existing_best_model)
            continue

        if model_name == "linear":
            exp_linear_model(df, num_obs, num_pred)
            continue

        model_path = torch_model_path(num_obs, num_pred, model_name)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        test_ips = torch.from_numpy(dataset.get_test_ips()).float()

        if os.path.exists(model_path):
            print('best model exists')
            best_model = torch.load(model_path)
            if compare_with_existing_best_model:
                best_model = rnn_find_better_model(
                    best_model,
                    ex_optuna(
                        train_dataloader=dl, model_name=model_name, num_obs=num_obs, num_pred=num_pred,
                        n_trials=n_trials, device=device, num_feature_output=num_feature_output, num_feature_input=num_feature_input
                    ), test_ips, dataset.get_original_meld_test()[:, num_obs:], dataset.meld_sc, device, num_obs, num_pred, model_name)
                torch.save(best_model, model_path)
        else:
            best_model = ex_optuna(
                train_dataloader=dl, model_name=model_name, num_obs=num_obs, num_pred=num_pred, n_trials=n_trials,
                device=device, num_feature_input=num_feature_input, num_feature_output=num_feature_output)
            torch.save(best_model, model_path)

        print(f"evaluate best model of {model_name}")
        print(f"model params: {best_model}")

        rnn_model_eval_and_plot(
            best_model, test_ips, dataset.get_original_meld_test(),
            dataset.meld_sc, device, num_obs, num_pred, model_name, "test")
        rnn_model_eval_and_plot(
            best_model, torch.from_numpy(dataset.get_generalize_ips()).float(),
            dataset.get_original_meld_generalize(),
            dataset.meld_sc, device, num_obs, num_pred, model_name, "generalize")

        print(
            f"Model experiment takes {time.time() - s} seconds or {int((time.time() - s) / 60)} minutes")
        print(f"Finished experiment on model {model_name}")
        print("=====================================================")


if __name__ == "__main__":
    run_exp(5, 3, 0.9, 0.2, "d", models_to_run, False)
