from sklearn.metrics import mean_squared_error, r2_score

from pkgs.data.commons import inverse_scale_ops
from pkgs.data.plot import analyze_timestep_rmse, analyze_ci_and_pi, plot_box, plot_line


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
