import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from pkgs.commons import input_path, preprocessed_train_set_data_path, \
    preprocessed_test_set_data_path, preprocessed_generalize_set_data_path, sklearn_model_path, xgboost_model_path, \
    torch_model_path
from pkgs.data.commons import inverse_scale_ops
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.harvest import harvest_data_with_interpolate
from pkgs.data.plot import analyze_timestep_rmse, analyze_ci_and_pi, plot_box, plot_line
from sklearn.experimental import enable_halving_search_cv  # required by sklearn


def rnn_model_eval(model, ips, expected_ops, scaler, device, num_obs, num_preds):
    model_pred = model(ips.to(device)).to("cpu").detach().numpy()
    print(
        f"Model evaluation: {model.__class__.__name__}. Input shape: {ips.shape}\n"
        f"Expected ops shape: {expected_ops.shape}\n"
        f"Predicted ops shape: {model_pred.shape}\n"
    )
    pred_full = inverse_scale_ops(ips[:, :, 0], model_pred, scaler, num_obs, num_preds)
    pred_future = pred_full[:, num_obs:]

    print(f"check pred shape {pred_future.shape}")
    best_rmse = mean_squared_error(expected_ops, pred_future, squared=False)
    best_rsquare = r2_score(expected_ops, pred_future)

    return best_rmse, best_rsquare, pred_future, pred_full


def rnn_model_eval_and_plot(model, ips, output_full, scaler, device, num_obs, num_pred, model_name,
                            subset_exp_name):
    expected_ops = output_full[:, num_obs:]
    r2, rmse, pred_future, pred_full = rnn_model_eval(model, ips, expected_ops, scaler, device, num_obs, num_pred)

    print(f"R-square is: {r2:.4f}")
    print(f"RMSE is {rmse:.4f}")

    analyze_timestep_rmse(expected_ops, pred_future, subset_exp_name, model_name, num_obs, num_pred)
    analyze_ci_and_pi(expected_ops, pred_future, subset_exp_name, model_name, num_obs, num_pred)

    plot_name = f"{model_name} R-square :{round(r2_score(expected_ops, pred_future), 2)} " \
                f"RMSE {round(mean_squared_error(expected_ops, pred_future, squared=False), 2)}"

    plot_box(expected_ops, pred_future, plot_name, model_name, num_obs, num_pred, subset_exp_name)
    plot_line(output_full, pred_full, plot_name, model_name, num_obs, num_pred, subset_exp_name)


def sklearn_model_eval(model, test_ips, expected_ops, scaler, num_obs, num_pred, num_feature_input):
    assert len(test_ips.shape) == 3
    assert test_ips.shape[2] == num_feature_input

    # reshape data into 2D, as sklearn models only accept 2D data
    test_ips_reshaped = np.reshape(test_ips, (-1, num_obs * num_feature_input))
    model_pred = model.predict(test_ips_reshaped)

    print(
        f"Model evaluation: {model.__class__.__name__}. Input shape: {test_ips.shape}\n"
        f"Expected ops shape: {expected_ops.shape}\n"
        f"Predicted ops shape: {model_pred.shape}\n"
    )

    pred_full = inverse_scale_ops(test_ips[:, :, 0], model_pred, scaler, num_obs, num_pred)
    pred_future = pred_full[:, num_obs:]

    print(f"check pred shape {pred_future.shape}")
    best_rmse = mean_squared_error(expected_ops, pred_future, squared=False)
    best_rsquare = r2_score(expected_ops, pred_future)

    return best_rmse, best_rsquare, pred_future, pred_full


def eval_and_plot_sklearn_model(test_ips, original_meld_test, scaler, model_name: str, num_obs, num_pred, num_feature_input, best_model, ext):
    expected_ops = original_meld_test[:, num_obs:]
    r2, rmse, pred_future, pred_full = sklearn_model_eval(best_model, test_ips, expected_ops, scaler, num_obs, num_pred, num_feature_input)

    print(f"R-square is: {r2:.4f}")
    print(f"RMSE is: {rmse:.4f}")

    analyze_timestep_rmse(expected_ops, pred_future, "test", model_name, num_obs, num_pred)
    analyze_ci_and_pi(expected_ops, pred_future, "test", model_name, num_obs, num_pred)

    plot_name = f"{model_name} R-square :{round(r2_score(expected_ops, pred_future), 2)} RMSE {round(mean_squared_error(expected_ops, pred_future, squared=False), 2)}"

    plot_box(expected_ops, pred_future, plot_name, model_name, num_obs, num_pred, ext)
    plot_line(original_meld_test, pred_full, plot_name, model_name, num_obs, num_pred, ext)


def analyze_models_performance(num_obs, num_pred, real_data_ratio, generalize_ratio, interpolate_amount, to_run_models):
    print(f"pre-processing data, experimenting on obs {num_obs} pred {num_pred}")
    s = time.time()
    df = pd.read_csv(input_path)

    if not os.path.exists(preprocessed_train_set_data_path):
        print("getting new data")
        exp_trains, exp_tests, exp_generalizes = harvest_data_with_interpolate(
            df, num_obs + num_pred, real_data_ratio, generalize_ratio, interpolate_amount)
        np.save(preprocessed_train_set_data_path, exp_trains)
        np.save(preprocessed_test_set_data_path, exp_tests)
        np.save(preprocessed_generalize_set_data_path, exp_generalizes)
    else:
        exp_trains = np.load(preprocessed_train_set_data_path)
        exp_tests = np.load(preprocessed_test_set_data_path)
        exp_generalizes = np.load(preprocessed_generalize_set_data_path)

    dataset = SlidingWindowDataset(exp_trains, exp_tests, exp_generalizes, num_obs, num_pred)

    print(f"preprocessing data takes {time.time() - s} seconds")
    trained_models = []
    for model_name in to_run_models:
        print("=====================================")
        print(f"exp on model {model_name}")

        if model_name in ["evr", "rfr"]:
            m = joblib.load(sklearn_model_path(num_obs, num_pred, model_name))
        elif model_name == "xgboost":
            m = XGBRegressor()
            m.load_model(xgboost_model_path(num_obs, num_pred, model_name))
        elif model_name in ["attention_lstm", "tcn", "tcn_lstm", "lstm", "cnn_lstm"]:
            m = torch.load(torch_model_path(num_obs, num_pred, model_name))
        else:
            raise ValueError(f"model {model_name} not supported")

        trained_models.append([model_name, m])


def evaluate_models(trained_models, dataset, num_obs, num_pred):
    for model_name, model in trained_models:
        print(f"evaluating model {model_name}")

