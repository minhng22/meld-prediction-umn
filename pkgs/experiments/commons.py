import numpy as np
from sklearn.metrics import mean_squared_error

from pkgs.data.commons import inverse_scale_ops
from pkgs.experiments.evaluate import rnn_model_eval


def rnn_find_better_model(model_a, model_b, test_ips, original_meld_test, scaler, device, num_obs, num_pred, model_name):
    print(f"finding better model for {model_name}")
    if model_b is None:
        return model_a
    if model_a is None:
        return model_b

    curr_rmse, _, _, _ = rnn_model_eval(model_a, test_ips, original_meld_test, scaler, device, num_obs, num_pred)
    best_rmse, _, _, _ = rnn_model_eval(model_b, test_ips, original_meld_test, scaler, device, num_obs, num_pred)

    if curr_rmse < best_rmse:
        return model_a
    return model_b


def sklearn_find_better_model(test_ips, original_meld_test, scaler, num_obs, num_pred, num_feature_input, model_a, model_b, model_name):
    if model_b is None:
        return model_a
    if model_a is None:
        return model_b

    print(f"finding better model for model {model_name}")
    test_ips = np.reshape(test_ips, (-1, num_obs * num_feature_input))  # reshape data into 2D
    tests = original_meld_test[:, num_obs:]

    tests_ops_a = model_a.predict(test_ips)
    tests_ops_full_a = inverse_scale_ops(test_ips[:, :, 0], tests_ops_a, scaler, num_obs, num_pred)
    tests_ops_a = tests_ops_full_a[:, num_obs:]

    tests_ops_b = model_b.predict(test_ips)
    tests_ops_full_b = inverse_scale_ops(test_ips[:, :, 0], tests_ops_b, scaler, num_obs, num_pred)
    tests_ops_b = tests_ops_full_b[:, num_obs:]

    if mean_squared_error(tests, tests_ops_a, squared=False) > mean_squared_error(tests, tests_ops_b, squared=False):
        return model_b
    return model_a
