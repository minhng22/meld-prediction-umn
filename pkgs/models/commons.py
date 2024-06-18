from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from pkgs.models.cnns import CNNLSTMModel
from pkgs.models.linears import TimeSeriesLinearModel
from pkgs.models.lstms import AttentionAutoencoderLSTMModel, LSTMModel
from pkgs.models.tcns import TCNLSTMModel, TCNModel


def get_model(model_name, s_s, device, num_obs, num_pred, num_feature_input, num_feature_output):
    if model_name == "attention_lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = AttentionAutoencoderLSTMModel(
            num_layers=s_s["num_layers"],
            num_heads=s_s["num_heads"],
            hidden_size=s_s["hidden_size"],
            dropout_lstm=s_s["dropout_lstm"],
            dropout_attn=s_s["dropout_attn"],
            num_pred=num_pred,
            device=device,
            num_feature_ip=num_feature_input,
        )
    elif model_name == "cnn_lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = CNNLSTMModel(
            num_layers=s_s["num_layers"],
            hidden_size=s_s["hidden_size"],
            dropout_lstm=s_s["dropout_lstm"],
            num_obs=num_obs,
            num_pred=num_pred,
            num_feature_input=num_feature_input,
            num_feature_output=num_feature_output,
        )
    elif model_name == "tcn_lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = TCNLSTMModel(
            num_layers=s_s["num_layers"],
            hidden_size=s_s["hidden_size"],
            dropout_lstm=s_s["dropout_lstm"],
            cnn_dropout=s_s["cnn_dropout"],
            tcn_num_layers=s_s["tcn_num_layers"],
            num_obs=num_obs,
            num_pred=num_pred,
            num_feature_output=num_feature_output,
            num_feature_input=num_feature_input,
        )
    elif model_name == "tcn":
        m = TCNModel(
            cnn_dropout=s_s["cnn_dropout"],
            tcn_num_layers=s_s["tcn_num_layers"],
            num_obs=num_obs,
            num_pred=num_pred,
            num_feature_input=num_feature_input,
            num_feature_output=num_feature_output,
        )
    elif model_name == "lstm":
        if s_s["num_layers"] == 1:
            s_s["dropout_lstm"] = 0
        m = LSTMModel(
            num_layers=s_s["num_layers"],
            hidden_size=s_s["hidden_size"],
            drop_out=s_s["dropout_lstm"],
            num_pred=num_pred,
            num_obs=num_obs,
            num_feature_ip=num_feature_input,
            num_feature_op=num_feature_output,
        )
    elif model_name == "time_series_linear":
        m = TimeSeriesLinearModel(
            num_obs=num_obs,
            num_pred=num_pred,
            num_feature_ip=num_feature_input,
            num_feature_op=num_feature_output,
        )
    else:
        raise ValueError(f"Model {model_name} not found")

    m.to(device)
    return m


def get_sklearn_model(model_name: str):
    if model_name == "evr":
        return ExtraTreesRegressor()
    if model_name == "rfr":
        return RandomForestRegressor()