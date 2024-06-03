import time, os
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader

from pkgs.commons import input_path, model_save_path
from pkgs.data.harvest import harvest_data_with_interpolate
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.experiments.optunas import ex_optuna
from pkgs.data.commons import inverse_scale_ops
from pkgs.data.plot import plot_box, plot_line, analyze_ci_and_pi, analyze_timestep_rmse


def model_eval(model, ips, expected_ops, scaler, device, num_obs, num_preds):
    pred = model(ips.to(device)).to("cpu").detach().numpy()
    print(
        f"Model evaluation: {model.__class__.__name__}. Input shape: {ips.shape}\n"
        f"Expected ops shape: {expected_ops.shape}\n"
        f"Predicted ops shape: {pred.shape}\n"
    )
    pred_full = inverse_scale_ops(ips[:, :, 0], pred, scaler, num_obs, num_preds)
    pred = pred_full[:, num_obs:]
    print(f"check pred shape {pred.shape}")
    best_rmse = mean_squared_error(expected_ops, pred, squared=False)
    best_rsquare = r2_score(expected_ops, pred)

    return best_rmse, best_rsquare


def model_eval_and_plot(model, ips, expected_ops, output_full, scaler, device, num_obs, num_pred, model_name,
                        subset_exp_name):
    pred = model(ips.to(device))
    print(
        f"Model evaluation: {model.__class__.__name__} on subset {subset_exp_name}. Input shape: {ips.shape}\n"
        f"Expected ops shape: {expected_ops.shape}\n"
        f"Predicted ops shape: {pred.shape}\n"
    )

    pred_ful = inverse_scale_ops(ips[:, :, 0], pred.to("cpu").detach().numpy(), scaler, num_obs, num_pred)
    pred = pred_ful[:, num_obs:]
    print(f"check pred shape {pred.shape}")

    print(f"R-square is: {r2_score(expected_ops, pred):.4f}")
    print(f"RMSE is {mean_squared_error(expected_ops, pred, squared=False):.4f}")

    analyze_timestep_rmse(expected_ops, pred, subset_exp_name, model_name, num_obs, num_pred)
    analyze_ci_and_pi(expected_ops, pred, subset_exp_name, model_name, num_obs, num_pred)

    plot_name = f"{model_name} R-square :{round(r2_score(expected_ops, pred), 2)} " \
                f"RMSE {round(mean_squared_error(expected_ops, pred, squared=False), 2)}"

    plot_box(expected_ops, pred, plot_name, model_name, num_obs, num_pred, subset_exp_name)
    plot_line(output_full, pred_ful, plot_name, model_name, num_obs, num_pred, subset_exp_name)


def find_better_model(model_a, model_b, test_ips, original_meld_test, scaler, device, num_obs, num_pred):
    print("finding better model")
    if model_b is None:
        return model_a
    if model_a is None:
        return model_b

    curr_rmse, _ = model_eval(model_a, test_ips, original_meld_test, scaler, device, num_obs, num_pred)
    best_rmse, _ = model_eval(model_b, test_ips, original_meld_test, scaler, device, num_obs, num_pred)

    if curr_rmse < best_rmse:
        return model_a
    return model_b


def run_exp(num_obs, num_pred, real_data_ratio, generalize_ratio, interpolate_amount, to_run_models,
            use_existing_best_model):
    batch_size = 256
    device = torch.device("cpu")
    n_trials = 1

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
    run_exp(5, 2, 0.9, 0.25, "d", ["transformer"], True)
