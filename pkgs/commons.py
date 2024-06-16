from pathlib import Path

patient_id_key_literal = "patient_id"
timestamp_key_literal = "timestamp"
meld_score_key_literal = "score"
is_original_key_literal = "is_original"

models = ["attention_lstm", "evr", "rfr", "xgboost", "linear", "tcn", "tcn_lstm", "lstm", "cnn_lstm"]
models_to_run = ["attention_lstm", "evr", "rfr", "tcn", "tcn_lstm", "lstm", "cnn_lstm"]

real_data_ratio = 0.9
generalize_ratio = 0.25
num_feature_input = 2 # MELD and timestamp


def project_dir():
    current_script_path = Path(__file__)
    return str(current_script_path.parent.parent)
input_path = project_dir() + "/data/patient_scores.csv"

preprocessed_train_set_data_path = project_dir() + "/data/preprocessed_train.npy"
preprocessed_test_set_data_path = project_dir() + "/data/preprocessed_test.npy"
preprocessed_generalize_set_data_path = project_dir() + "/data/preprocessed_gen.npy"


def exp_path(num_obs, num_pred):
    return project_dir() + f"/experiments_generated_data/obs_{num_obs}_pred_{num_pred}"


def figs_path(num_obs, num_pred):
    return exp_path(num_obs, num_pred) + "/figs"


def time_series_sequence_path(num_obs, num_pred):
    return figs_path(num_obs, num_pred) + "/time_series_sequence"


def model_performance_path(num_obs, num_pred):
    return figs_path(num_obs, num_pred) + "/model_performance"
def box_plot_path(num_obs, num_pred):
    return model_performance_path(num_obs, num_pred) + "/box_plot"
def line_plot_path(num_obs, num_pred):
    return model_performance_path(num_obs, num_pred) + "/line_plot"
def linear_plot_path():
    return project_dir() + f"/experiments_generated_data/linear_plot"
def rmse_by_day_path(num_obs, num_pred):
    return model_performance_path(num_obs, num_pred) + "/rmse_by_day"


# ---- Paths for the trained models ----
def model_save_path(num_obs, num_pred):
    return exp_path(num_obs, num_pred) + "/models"
def sklearn_model_path(num_obs, num_pred, model_name):
    return f"{model_save_path(num_obs, num_pred)}/{model_name}.pkl"
def xgboost_model_path(num_obs, num_pred, model_name):
    return f"{model_save_path(num_obs, num_pred)}/{model_name}.json"
def torch_model_path(num_obs, num_pred, model_name):
    return f"{model_save_path(num_obs, num_pred)}/{model_name}.pt"


def models_performance_path(num_obs, num_pred):
    return figs_path(num_obs, num_pred) + "/models_performance"
def line_plot_models_performance_path(num_obs, num_pred):
    return models_performance_path(num_obs, num_pred) + "/line_plot"
def rmse_by_day_models_performance_path(num_obs, num_pred):
    return models_performance_path(num_obs, num_pred) + "/rmse_by_day"
def box_plot_models_performance_path(num_obs, num_pred):
    return models_performance_path(num_obs, num_pred) + "/box_plot"
def rmse_by_day_models_performance_path(num_obs, num_pred):
    return models_performance_path(num_obs, num_pred) + "/rmse_by_day"