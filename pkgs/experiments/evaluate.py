import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from pkgs.commons import input_path, preprocessed_train_set_data_path, \
    preprocessed_test_set_data_path, preprocessed_generalize_set_data_path, sklearn_model_path, xgboost_model_path, \
    torch_model_path, models_to_run
from pkgs.data.commons import inverse_scale_ops
from pkgs.data.dataset import SlidingWindowDataset
from pkgs.data.harvest import harvest_data_with_interpolate
from pkgs.data.plot import plot_timestep_rmse, plot_box, plot_line, plot_line_models, \
    plot_box_models, plot_timestep_rmse_models
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
    best_mae = mean_absolute_error(expected_ops, pred_future)

    return best_rmse, best_rsquare, best_mae, pred_future, pred_full


def rnn_model_eval_and_plot(model, ips, output_full, scaler, device, num_obs, num_pred, model_name,
                            subset_exp_name):
    expected_ops = output_full[:, num_obs:]
    rmse, r2, mae, pred_future, pred_full = rnn_model_eval(model, ips, expected_ops, scaler, device, num_obs, num_pred)

    print(
        f"R-square is: {r2:.3f}\n"
        f"RMSE is: {rmse:.3f}\n"
        f"MAE is: {mae:.3f}\n"
    )

    plot_timestep_rmse(expected_ops, pred_future, subset_exp_name, model_name, num_obs, num_pred)
    evaluate_95_ci(expected_ops, pred_future, subset_exp_name, model_name)

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
    best_mae = mean_absolute_error(expected_ops, pred_future)

    return best_rmse, best_rsquare, best_mae, pred_future, pred_full


def sklearn_model_eval_and_plot(test_ips, original_meld_test, scaler, model_name: str, num_obs, num_pred, num_feature_input, best_model, ext):
    expected_ops = original_meld_test[:, num_obs:]
    rmse, r2, mae, pred_future, pred_full = sklearn_model_eval(best_model, test_ips, expected_ops, scaler, num_obs, num_pred, num_feature_input)

    print(
        f"R-square is: {r2:.3f}\n"
        f"RMSE is: {rmse:.3f}\n"
        f"MAE is: {mae:.3f}\n"
    )

    plot_timestep_rmse(expected_ops, pred_future, "test", model_name, num_obs, num_pred)
    evaluate_95_ci(expected_ops, pred_future, "test", model_name)

    plot_name = (f"{model_name} R-square :{round(r2_score(expected_ops, pred_future), 2)} "
                 f"RMSE {round(mean_squared_error(expected_ops, pred_future, squared=False), 2)}")

    plot_box(expected_ops, pred_future, plot_name, model_name, num_obs, num_pred, ext)
    plot_line(original_meld_test, pred_full, plot_name, model_name, num_obs, num_pred, ext)


def run(num_obs, num_pred, real_data_ratio, generalize_ratio, interpolate_amount, to_run_models):
    num_feature_input, num_feature_output = 2, 1  # MELD and timestamp
    device = torch.device("cpu")
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

    dataset = SlidingWindowDataset(exp_trains, exp_tests, exp_generalizes, num_obs, num_pred, None, None)

    print(f"preprocessing data takes {time.time() - s} seconds")
    trained_models = []
    for model_name in to_run_models:
        if model_name in ["evr", "rfr"]:
            m = joblib.load(sklearn_model_path(num_obs, num_pred, model_name))
        elif model_name == "xgboost":
            m = XGBRegressor()
            m.load_model(xgboost_model_path(num_obs, num_pred, model_name))
        elif model_name in ["attention_lstm", "tcn", "tcn_lstm", "lstm", "cnn_lstm", "time_series_linear"]:
            m = torch.load(torch_model_path(num_obs, num_pred, model_name))
        else:
            raise ValueError(f"model {model_name} not supported")

        trained_models.append([model_name, m])

    eval_res_test, eval_res_gen = evaluate_models(trained_models, dataset, num_obs, num_pred, num_feature_input, device)

    plot_line_models(
        y_target=dataset.get_original_meld_test()[:, num_obs:],
        ys=[res[4] for res in eval_res_test],
        model_names=[res[0] for res in eval_res_test],
        num_obs=num_obs,
        num_pred=num_pred,
        ext="test"
    )
    plot_box_models(
        y_target=dataset.get_original_meld_test()[:, num_obs:],
        ys=[res[4] for res in eval_res_test],
        model_names=[res[0] for res in eval_res_test],
        num_obs=num_obs,
        num_pred=num_pred,
        ext="test"
    )
    plot_timestep_rmse_models(
        y_target=dataset.get_original_meld_test()[:, num_obs:],
        ys=[res[4] for res in eval_res_test],
        model_names=[res[0] for res in eval_res_test],
        num_obs=num_obs,
        num_pred=num_pred,
        exp_name="test"
    )

    plot_line_models(
        y_target=dataset.get_original_meld_generalize()[:, num_obs:],
        ys=[res[4] for res in eval_res_gen],
        model_names=[res[0] for res in eval_res_gen],
        num_obs=num_obs,
        num_pred=num_pred,
        ext="generalize"
    )
    plot_box_models(
        y_target=dataset.get_original_meld_generalize()[:, num_obs:],
        ys=[res[4] for res in eval_res_gen],
        model_names=[res[0] for res in eval_res_gen],
        num_obs=num_obs,
        num_pred=num_pred,
        ext="generalize"
    )
    plot_timestep_rmse_models(
        y_target=dataset.get_original_meld_generalize()[:, num_obs:],
        ys=[res[4] for res in eval_res_gen],
        model_names=[res[0] for res in eval_res_gen],
        num_obs=num_obs,
        num_pred=num_pred,
        exp_name="generalize"
    )


def evaluate_models(trained_models, dataset, num_obs, num_pred, num_feature_input, device):
    res_test, res_generalize = [], []
    for model_name, model in trained_models:
        print(f"evaluating model {model_name} on test set")
        if model_name in ["evr", "rfr", "xgboost"]:
            test_rmse, test_r2, test_mae, test_pred_future, test_pred_full = sklearn_model_eval(
                model=model,
                test_ips=dataset.get_test_ips(),
                expected_ops=dataset.get_original_meld_test()[:, num_obs:],
                scaler=dataset.meld_sc,
                num_obs=num_obs,
                num_pred=num_pred,
                num_feature_input=num_feature_input
            )
        elif model_name in ["attention_lstm", "tcn", "tcn_lstm", "lstm", "cnn_lstm", "time_series_linear"]:
            test_rmse, test_r2, test_mae, test_pred_future, test_pred_full = rnn_model_eval(
                model = model,
                ips = torch.from_numpy(dataset.get_test_ips()).float(),
                expected_ops=dataset.get_original_meld_test()[:, num_obs:],
                scaler=dataset.meld_sc,
                device=device,
                num_obs=num_obs,
                num_preds=num_pred
            )
        else:
            raise ValueError(f"model {model_name} not supported")
        print(
            f"R-square is: {test_r2:.3f}\n"
            f"RMSE is: {test_rmse:.3f}\n"
            f"MAE is: {test_mae:.3f}\n"
        )

        evaluate_95_ci(dataset.get_original_meld_test()[:, num_obs:], test_pred_future, "test", model_name)

        print(f"evaluating model {model_name} on generalize set")
        if model_name in ["evr", "rfr", "xgboost"]:
            gen_rmse, gen_r2, gen_mae, gen_pred_future, gen_pred_full = sklearn_model_eval(
                model=model,
                test_ips=dataset.get_generalize_ips(),
                expected_ops=dataset.get_original_meld_generalize()[:, num_obs:],
                scaler=dataset.meld_sc,
                num_obs=num_obs,
                num_pred=num_pred,
                num_feature_input=num_feature_input
            )
        elif model_name in ["attention_lstm", "tcn", "tcn_lstm", "lstm", "cnn_lstm", "time_series_linear"]:
            gen_rmse, gen_r2, gen_mae, gen_pred_future, gen_pred_full = rnn_model_eval(
                model = model,
                ips = torch.from_numpy(dataset.get_generalize_ips()).float(),
                expected_ops=dataset.get_original_meld_generalize()[:, num_obs:],
                scaler=dataset.meld_sc,
                device=device,
                num_obs=num_obs,
                num_preds=num_pred)
        else:
            raise ValueError(f"model {model_name} not supported")
        print(
            f"R-square is: {gen_r2:.3f}\n"
            f"RMSE is: {gen_rmse:.3f}\n"
            f"MAE is: {gen_mae:.3f}\n"
        )
        evaluate_95_ci(dataset.get_original_meld_generalize()[:, num_obs:], gen_pred_future, "generalize", model_name)

        res_test.append([model_name, test_rmse, test_r2, test_mae, test_pred_future, test_pred_full])
        res_generalize.append([model_name, gen_rmse, gen_r2, gen_mae, gen_pred_future, gen_pred_full])

    return res_test, res_generalize


def evaluate_95_ci(target, prediction, exp_name, model_name):
    print(f"Calculating prediction interval for {exp_name} {model_name}")
    # Calculate residuals
    residuals = target - prediction

    # Calculate the standard error of the residuals
    se_residuals = np.std(residuals, ddof=2)

    # Calculate the standard error of the prediction
    se_predictions = se_residuals / np.sqrt(prediction.shape[1])

    # Critical value for 95% confidence
    critical_value = 1.96

    # Calculate the confidence intervals per time step
    lower_bounds = []
    upper_bounds = []
    for i in range(prediction.shape[1]):
        margin_of_error = critical_value * se_predictions
        lower_bound = prediction[i] - margin_of_error
        upper_bound = prediction[i] + margin_of_error
        lower_bounds.append(np.round(lower_bound, 3))
        upper_bounds.append(np.round(upper_bound, 3))

    print(
        f"95% CI of each time step for {exp_name} {model_name}\n"
        f"lower_bound {lower_bound} upper_bound {upper_bound}\n")


if __name__ == "__main__":
    run(num_obs=5, num_pred=3, real_data_ratio=0.9, generalize_ratio=0.2, interpolate_amount="d",
        to_run_models=models_to_run)
