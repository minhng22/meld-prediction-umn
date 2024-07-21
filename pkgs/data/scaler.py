import joblib


def get_fitted_scaler(path):
    return joblib.load(path)

def save_fitted_scaler(scaler, path):
    joblib.dump(scaler, path)