from pathlib import Path

patient_id_key_literal = "patient_id"
timestamp_key_literal = "timestamp"
meld_score_key_literal = "score"
is_original_key_literal = "is_original"

real_data_ratio = 0.9
generalize_ratio = 0.25


def project_dir():
    current_script_path = Path(__file__)
    return str(current_script_path.parent.parent)
input_path = project_dir() + "/data/patient_scores.csv"


def exp_path(num_obs, num_pred):
    return project_dir() + f"/experiments/obs_{num_obs}_pred_{num_pred}"
def figs_path(num_obs, num_pred):
    return exp_path(num_obs, num_pred) + "/figs"
def model_save_path(num_obs, num_pred):
    return exp_path(num_obs, num_pred) + "/models"