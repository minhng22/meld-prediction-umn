from pathlib import Path

patient_id_key_literal = "patient_id"
timestamp_key_literal = "timestamp"
meld_score_key_literal = "score"
is_original_key_literal = "is_original"

real_data_ratio = 0.9
generalize_ratio = 0.25

def get_input_path():
    current_script_path = Path(__file__)
    project_dir = current_script_path.parent.parent
    return str(project_dir) + "/data/patient_scores.csv"