import time

import optuna
from torch import optim
from torch.nn import HuberLoss
from torch.utils.data import DataLoader

from pkgs.models.commons import get_model


def ex_optuna(train_dataloader: DataLoader, model_name, num_obs, num_pred, n_trials, device, num_feature_input, num_feature_output):
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: optuna_objective(
            trial=trial, train_dataloader=train_dataloader, model_name=model_name,
            device=device, num_obs=num_obs, num_pred=num_pred,
            num_feature_input=num_feature_input, num_feature_output=num_feature_output
        ),
        n_trials=n_trials,
        gc_after_trial=True,

    )
    # Print best hyperparameters and loss
    best_trial = study.best_trial

    print(f"Best hyperparameters: {study.best_params}")
    best_model = best_trial.user_attrs["model"]

    return best_model


def get_optuna_params(trial: optuna.trial.Trial, model_name, filler=False):
    if model_name == "attention_lstm":
        num_layers = "num_layers" if not filler else "num_layers_filler"
        num_heads = "num_heads" if not filler else "num_heads_filler"
        hidden_size = "hidden_size" if not filler else "hidden_size_filler"
        dropout_lstm = "dropout_lstm" if not filler else "dropout_lstm_filler"
        dropout_attn = "dropout_attn" if not filler else "dropout_attn_filler"

        return {
            "num_layers": trial.suggest_int(num_layers, 1, 6),
            "num_heads": trial.suggest_categorical(num_heads, [2, 3, 4, 5, 6, 8]),
            "hidden_size": trial.suggest_int(hidden_size, 120, 840, step = 30),
            "dropout_lstm": trial.suggest_float(dropout_lstm, 0, 0.5),
            "dropout_attn": trial.suggest_float(dropout_attn, 0, 0.5),
        }
    if model_name == "cnn_lstm":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 120, 720, step = 60),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
        }
    if model_name == "tcn_lstm":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 120, 480, step = 120),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
            "cnn_dropout": trial.suggest_float("cnn_dropout", 0, 0.5),
            "cnn_kernel_size": trial.suggest_int("cnn_kernel_size", 2, 7),
            "tcn_num_layers": trial.suggest_int("tcn_num_layers", 1, 10),
        }
    if model_name == "tcn":
        return {
            "cnn_dropout": trial.suggest_float("cnn_dropout", 0, 0.5),
            "tcn_num_layers": trial.suggest_int("tcn_num_layers", 1, 10),
        }
    if model_name == "lstm_seq_2_seq":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 128, 640, step = 128),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
        }
    if model_name == "lstm":
        return {
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "hidden_size": trial.suggest_int("hidden_size", 120, 840, step = 30),
            "dropout_lstm": trial.suggest_float("dropout_lstm", 0, 0.5),
        }


def optuna_objective(
        trial: optuna.trial.Trial, train_dataloader: DataLoader, model_name,
        device, num_obs, num_pred, num_feature_input, num_feature_output
):
    learning_rate = 0.1
    gamma_rate = 0.9
    num_epoch = 1
    patience = 15

    params = get_optuna_params(trial, model_name)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    model = get_model(
        model_name=model_name, s_s=params, device=device, num_obs=num_obs,
        num_pred=num_pred, num_feature_input=num_feature_input, num_feature_output=num_feature_output)

    # Define loss function and optimizer
    trial_loss = float("inf")
    criterion = HuberLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=gamma_rate)

    # Early stopping
    best_val_loss = float("inf")
    patient_cnt = 0
    should_stop = False

    # Train the model
    s = time.time()
    print("Training model")
    for epoch in range(num_epoch):
        epoc_s = time.time()
        total_loss = 0.0
        for train_ip, train_target in train_dataloader:
            train_ip, train_target = train_ip.float(), train_target.float()

            # Forward pass
            train_op = model(train_ip.to(device))

            # Compute loss
            loss = criterion(train_op.to(device), train_target.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= len(train_dataloader)

        if total_loss < best_val_loss:
            best_val_loss = total_loss
            patient_cnt = 0
            trial.set_user_attr(key="model", value=model)
        else:
            patient_cnt += 1
            if patient_cnt >= patience:
                should_stop = True

        trial_loss = min(trial_loss, total_loss)
        lr_scheduler.step()

        print(
            f"Epoch {epoch + 1}/{num_epoch}, Val Loss: {total_loss:.4f}"
            + f". Finished in {time.time() - epoc_s} seconds, or {int((time.time() - epoc_s) / 60)} minutes"
        )

        if should_stop:
            print("Early stopping!")
            break

    print(
        f"Time taken to train model this trial {time.time() - s} seconds or {round((time.time() - s) / 60)} minutes"
    )

    return trial_loss