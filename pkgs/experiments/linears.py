import os
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from torch import nn, optim

from pkgs.commons import model_save_path, linear_plot_path
from pkgs.models.linears import LinearModel


def exp_linear_model(df, num_obs, num_pred):
    s = time.time()

    df = df.dropna(subset=['timestamp', 'score'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract date, time, and year into numpy arrays
    day_arr = df['timestamp'].dt.day.values
    month_arr = df['timestamp'].dt.month.values
    year_arr = df['timestamp'].dt.year.values
    meld = df['score'].values
    print(f"ip shape {meld.shape}, {day_arr.shape}, {month_arr.shape}, {year_arr.shape}")

    T = np.concatenate((
        np.reshape(day_arr, (day_arr.shape[0], 1)),
        np.reshape(month_arr, (month_arr.shape[0], 1)),
        np.reshape(year_arr, (year_arr.shape[0], 1)),
    ), axis=1)
    meld = np.reshape(meld, (meld.shape[0], 1))
    print(f"meld.shape T.shape {meld.shape} {T.shape}")
    print(f"meld_train nan {np.any(np.isnan(meld))} {np.any(np.isnan(T))}")

    break_i = int(meld.shape[0] * 0.8)
    meld_train, meld_test = meld[:break_i, :], meld[:break_i, :]
    time_train, time_test = T[:break_i, :], T[:break_i, :]

    model_name = "linear"
    model_path = model_save_path(num_obs, num_pred) + "/" + model_name + ".pt"

    if os.path.exists(model_path):
        print('best model exists')
        model = torch.load(model_path)
    else:
        model = run_linear_model(time_train, meld_train)
        torch.save(model, model_path)

    def plot(pred, real, plot_name, ext):
        plt.scatter(real, pred)

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        plt.title(plot_name)
        plt.xlabel("real MELD")
        plt.ylabel("predict MELD")

        plt.savefig(f"{linear_plot_path(num_obs, num_pred)}/{ext}_{model_name}", bbox_inches="tight")
        plt.clf()

    train_meld_pred = model(torch.Tensor(time_train)).detach().numpy()
    print(f"train_meld_pred {train_meld_pred.shape} {meld_train.shape}")
    print(f"meld_train nan {np.any(np.isnan(train_meld_pred))} {np.any(np.isnan(meld_train))}")
    plot_name_train = f"{model_name} R-square :{round(r2_score(train_meld_pred, meld_train), 3)} RMSE {round(mean_squared_error(train_meld_pred, meld_train, squared=False), 3)}"

    plot(train_meld_pred, meld_train, plot_name_train, "train")

    test_meld_pred = model(torch.Tensor(time_test)).detach().numpy()
    print(f"test_meld_pred {test_meld_pred.shape} {meld_test.shape}")
    p_name = f"{model_name} R-square :{round(r2_score(test_meld_pred, meld_test), 3)} RMSE {round(mean_squared_error(test_meld_pred, meld_test, squared=False), 3)}"

    plot(test_meld_pred, meld_test, p_name, "test")

    print(f"Model experiment takes {time.time() - s} seconds or {int((time.time() - s) / 60)} minutes")


def run_linear_model(ip, op):
    print(f"run_linear_model shape {ip.shape} {op.shape}")

    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop for 5 trials
    num_trials = 5
    num_epochs = 100
    batch_size = 10
    best_loss = float('inf')
    best_model = None

    for trial in range(num_trials):
        model = LinearModel()

        for epoch in range(num_epochs):
            for i in range(0, len(op), batch_size):
                X_batch = torch.Tensor(ip[i:i + batch_size])
                y_batch = torch.Tensor(op[i:i + batch_size])

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Trial {trial + 1}, Epoch {epoch}, Loss: {loss.item()}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model

    return best_model