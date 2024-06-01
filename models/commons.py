from pathlib import Path

INPUT_PATH = "./data/patient_scores.csv"
PATIENT_ID_KEY_LITERAL = "patient_id"
TIMESTAMP_KEY_LITERAL = "timestamp"
MELD_SCORE_KEY_LITERAL = "score"
IS_ORIGINAL_KEY_LITERAL = "is_original"

def get_input_path():
    current_script_path = Path(__file__)
    project_dir = current_script_path.parent.parent
    return str(project_dir) + "/data/patient_scores.csv"