import os
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pkgs.commons import input_path, model_save_path
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.harvest import harvest_data_with_interpolate
from pkgs.experiments.commons import find_better_model, model_eval_and_plot
from pkgs.experiments.optunas import ex_optuna
from pkgs.experiments.sklearns import exp_sklearn_model


def run_exp(num_obs, num_pred, real_data_ratio, generalize_ratio, interpolate_amount, to_run_models,
            use_existing_best_model):
    batch_size = 256
    device = torch.device("cpu")
    n_trials = 1
    num_feature_input = 2

    print(f"pre-processing data, experimenting on obs {num_obs} pred {num_pred}")
    s = time.time()
    df = pd.read_csv(input_path)

    exp_trains, exp_tests, exp_generalizes = harvest_data_with_interpolate(
        df, num_obs + num_pred, real_data_ratio, generalize_ratio, interpolate_amount)

    print(f"preprocessing data takes {time.time() - s} seconds")

    for model_name in to_run_models:
        print("=====================================")
        print(f"exp on model {model_name}")

        dataset = SlidingWindowDataset(exp_trains, exp_tests, exp_generalizes, num_obs, num_pred)

        print("finding best model")
        s = time.time()

        if model_name in ["evr", "rfr"]:
            exp_sklearn_model(dataset, model_name, num_obs, num_pred)
            continue

        model_path = model_save_path(num_obs, num_pred) + "/" + model_name + ".pt"
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        test_ips = torch.from_numpy(dataset.get_test_ips()).float()
        original_meld_test_set = dataset.get_original_meld_test()[:, num_obs:]

        if os.path.exists(model_path):
            print('best model exists')
            if not use_existing_best_model:
                best_model = find_better_model(
                    torch.load(model_path), ex_optuna(
                        train_dataloader=dl, model_name=model_name, num_obs=num_obs, num_pred=num_pred,
                        n_trials=n_trials, device=device
                    ), test_ips, original_meld_test_set, dataset.meld_sc, device, num_obs, num_pred)
            else:
                best_model = torch.load(model_path)
        else:
            best_model = ex_optuna(
                train_dataloader=dl, model_name=model_name, num_obs=num_obs, num_pred=num_pred, n_trials=n_trials,
                device=device)

        print(f"evaluate best model of {model_name}")
        print(f"model params: {best_model}")

        model_eval_and_plot(
            best_model, test_ips, original_meld_test_set, dataset.get_original_meld_test(),
            dataset.meld_sc, device, num_obs, num_pred, model_name, "test")
        model_eval_and_plot(
            best_model, torch.from_numpy(dataset.get_generalize_ips()).float(),
            dataset.get_original_meld_generalize()[:, num_obs:], dataset.get_original_meld_generalize(),
            dataset.meld_sc, device, num_obs, num_pred, model_name, "generalize")

        torch.save(best_model, model_path)

        print(
            f"Model experiment takes {time.time() - s} seconds or {int((time.time() - s) / 60)} minutes")
        print(f"Finished experiment on model {model_name}")
        print("=====================================================")


if __name__ == "__main__":
    run_exp(5, 2, 0.9, 0.25, "d",
            ["rfr", "evr"], True)
