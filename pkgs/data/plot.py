import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pkgs.commons import box_plot_path, line_plot_path, rmse_by_day_path, \
    line_plot_models_performance_path, time_series_sequence_path, box_plot_models_performance_path, \
    rmse_by_day_models_performance_path
from pkgs.data.commons import calculate_rmse_of_time_step

target_color = "steelblue"
predict_color = "crimson"


def plot_data(train, test, generalize, num_observed, num_predicted, label_post_text=""):
    print("Plotting data")
    print(
        f"train: {train.shape}, test: {test.shape}, generalize: {generalize.shape}"
    )

    def plot(data, label):
        # Calculate the mean and standard deviation for each day
        mean_scores = data.mean(axis=0)
        std_scores = data.std(axis=0)

        # Plot the mean scores with a shaded area for the standard deviation
        plt.figure(figsize=(12, 6))
        plt.plot(mean_scores, label='Mean Score')
        plt.fill_between(range(data.shape[1]), mean_scores - std_scores, mean_scores + std_scores, alpha=0.2,
                         label='Standard Deviation')

        plt.title('Mean And Standard Deviation of MELD Score Over Consecutive Days')
        plt.xlabel('Day')
        plt.ylabel('MELD Score')
        plt.legend()

        figPath = f'{time_series_sequence_path(num_observed, num_predicted)}/analyze_{label}_{label_post_text}.png'
        print(f"Saving figure to {figPath}")

        plt.savefig(figPath, bbox_inches="tight")
        plt.clf()

    plot(train, "train")
    plot(test, "test")
    plot(generalize, "generalize")


def plot_box(y_target, y, plot_name, model_name, num_obs, num_pred, ext=""):
    print("plot_box")

    def create_df(ip, ip_target):
        df = pd.DataFrame(columns=['score', 'data', 'day'])

        for i in range(ip_target.shape[1]):
            for score in ip_target[:, i]:
                df = pd.concat([df, pd.DataFrame([{'score': score, 'data': 'target', 'day': i + 1}])],
                               ignore_index=True)
            for score in ip[:, i]:
                df = pd.concat([df, pd.DataFrame([{'score': score, 'data': 'prediction', 'day': i + 1}])],
                               ignore_index=True)
        return df

    sns.boxplot(data=create_df(y, y_target), x='day', y='score',
                hue='data', palette={'target': target_color, 'prediction': predict_color})

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    plt.savefig(box_plot_path(num_obs, num_pred) + "/" + ext + "_" + model_name, bbox_inches="tight")
    plt.clf()


def plot_line(y_target, y, plot_name, model_name, num_obs, num_pred, ext=""):
    print("plot_line")
    y_avg = np.average(y, axis=0)
    y_target_avg = np.average(y_target, axis=0)

    tsf_y = np.arange(1, y_avg.shape[0] + 1)

    plt.plot(tsf_y, y_avg, color=predict_color, label="prediction")
    plt.plot(tsf_y, y_target_avg, color=target_color, label="target")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    stk = int(y_avg.shape[0] / 10) if int(y_avg.shape[0] / 10) > 0 else 1
    plt.xticks(np.arange(1, y_avg.shape[0] + 1, stk))

    plt.savefig(line_plot_path(num_obs, num_pred) + "/" + ext + "_" + model_name, bbox_inches="tight")
    plt.clf()


def plot_line_models(y_target, ys, model_names, num_obs, num_pred, ext=""):
    print("plot_line")
    y_target_avg = np.average(y_target, axis=0)

    tsf_y = np.arange(1, y_target_avg.shape[0] + 1)

    plt.plot(tsf_y, y_target_avg, label="target")

    for i, y in enumerate(ys):
        y_avg = np.average(y, axis=0)
        plt.plot(tsf_y, y_avg, label=model_names[i])

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    stk = int(y_target_avg.shape[0] / 10) if int(y_target_avg.shape[0] / 10) > 0 else 1
    plt.xticks(np.arange(1, y_target_avg.shape[0] + 1, stk))

    plt.savefig(line_plot_models_performance_path(num_obs, num_pred) + "/" + ext, bbox_inches="tight")
    plt.clf()


def plot_box_models(y_target, ys, model_names, num_obs, num_pred, ext=""):
    print("plot_box_models")

    def create_df(ip_list, ip_target):
        D = pd.DataFrame(columns=['score', 'data', 'day'])

        for i in range(ip_target.shape[1]):
            for score in ip_target[:, i]:
                D = pd.concat([D, pd.DataFrame([{'score': score, 'data': 'target', 'day': i + 1}])], ignore_index=True)
            for j, ip in enumerate(ip_list):
                for score in ip[:, i]:
                    D = pd.concat([D, pd.DataFrame([{'score': score, 'data': model_names[j], 'day': i + 1}])], ignore_index=True)
        return D

    df = create_df(ys, y_target)
    predict_colors = sns.color_palette("husl", len(model_names))

    # Create a color palette dictionary for seaborn
    palette = {'target': target_color}
    for model_idx, model_name in enumerate(model_names):
        palette[model_name] = predict_colors[model_idx]

    sns.boxplot(data=df, x='day', y='score', hue='data', palette=palette)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    plt.savefig(box_plot_models_performance_path(num_obs, num_pred) + "/" + ext, bbox_inches="tight")
    plt.clf()


def plot_timestep_rmse(ip, op, exp_name, model_name, num_obs, num_pred):
    print(f"Calculating rmse for {exp_name}")
    if ip.shape[1] != op.shape[1]:
        raise ValueError("ip and op must have same shape")
    rmses = calculate_rmse_of_time_step(ip, op)

    print(f"rmses of model {model_name}: {rmses}")

    plt.plot(np.arange(1, num_pred + 1), rmses)
    plt.xticks(ticks=np.arange(1, num_pred + 1))  # Set x-ticks to be integers

    plt.xlabel('Day')
    plt.ylabel('RMSE')

    plt.savefig(rmse_by_day_path(num_obs, num_pred) + "/" + exp_name + "_" + model_name, bbox_inches="tight")
    plt.clf()


def plot_timestep_rmse_models(y_target, ys, exp_name, model_names, num_obs, num_pred):
    print(f"Calculating rmses for {exp_name}")

    for i, y in enumerate(ys):
        if y_target.shape[1] != y.shape[1]:
            raise ValueError("ip and y must have same shape")
        rmse = calculate_rmse_of_time_step(y_target, y)

        print(f"rmse per time step of model {model_names[i]}: {rmse}")

        plt.plot(np.arange(1, num_pred + 1), rmse, label=f"{model_names[i]}")

    plt.xticks(ticks=np.arange(1, num_pred + 1))  # Set x-ticks to be integers

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    plt.xlabel('Day')
    plt.ylabel('RMSE')

    plt.savefig(rmse_by_day_models_performance_path(num_obs, num_pred) + "/" + exp_name, bbox_inches="tight")
    plt.clf()