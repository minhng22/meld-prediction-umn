import numpy as np
import matplotlib.pyplot as plt
from pkgs.commons import figs_path, box_plot_path, line_plot_path, pi_ci_path, rmse_by_day_path
from pkgs.data.commons import generate_timestep_for_plot, calculate_rmse_of_time_step
from scipy.stats import sem, t, norm
import pandas as pd
import seaborn as sns

target_color = "steelblue"
predict_color = "crimson"


def plot_data(train, test, generalize, num_observed, num_predicted, label_post_text=""):
    print("Plotting data")
    print(
        f"train: {train.shape}, test: {test.shape}, generalize: {generalize.shape}"
    )

    def plot(data, label):
        avg = np.average(data, axis=0)
        x_avg = np.arange(1, avg.shape[0] + 1)

        x_scatter = generate_timestep_for_plot(data.shape[0], data.shape[1])

        print(
            f"Plotting data:\n"
            f"avg: {avg}\n"
            f"x_avg: {x_avg}\n"
            f"data.shape: {data.shape} x_scatter.shape: {x_scatter.shape}")

        plt.scatter(x_scatter, data, color="#ADD8E6", alpha=0.03)
        plt.plot(x_avg, avg, color="blue", label=label)

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        stk = int(avg.shape[0] / 10) if int(avg.shape[0] / 10) > 0 else 1
        plt.xticks(np.arange(1, avg.shape[0] + 1, stk))
        plt.title(f'Dataset analysis obs {num_observed} pred {num_predicted}')

        figPath = (figs_path(num_observed, num_predicted) +
                   f'/obs_{num_observed}_pred_{num_predicted}/analyze_{label}_{label_post_text}.png')
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

    plt.title(plot_name)
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
    plt.title(plot_name)

    plt.savefig(line_plot_path(num_obs, num_pred) + "/" + ext + "_" + model_name, bbox_inches="tight")
    plt.clf()


def analyze_timestep_rmse(ip, op, exp_name, model_name):
    print(f"Calculating rmse for {exp_name}")
    if ip.shape[1] != op.shape[1]:
        raise ValueError("ip and op must have same shape")
    rmses = calculate_rmse_of_time_step(ip, op)

    print(f"rmses: {rmses}")

    plt.plot(list(range(len(rmses))), rmses)

    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.title('RMSE by day')

    plt.savefig(RMSE_DAYS_FIG_PATH + "/" + exp_name + "_" + model_name, bbox_inches="tight")
    plt.clf()


def analyze_ci_and_pi(target, prediction, exp_name, model_name, num_obs, num_pred):
    CONFIDENCE_LEVEL = 0.95  # common value for confidence level

    def analyze_ci():
        data = calculate_rmse_of_time_step(target, prediction)

        n = len(data)
        mean = np.mean(data)
        standard_error = sem(data)
        t_value = t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1)
        margin_of_error = t_value * standard_error

        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        print(
            f"Confidence interval {CONFIDENCE_LEVEL * 100}% for RMSE for {exp_name} {model_name}: {lower_bound:.2f} {upper_bound:.2f}")

    def analyze_pi():
        residuals = target - prediction

        residual_std = np.std(residuals)

        z_score = norm.ppf((100 - CONFIDENCE_LEVEL) / 200)

        margin_of_error = z_score * residual_std

        lower_bound = np.min(prediction - margin_of_error, axis=0)
        upper_bound = np.max(prediction + margin_of_error, axis=0)

        print(f"Prediction interval {CONFIDENCE_LEVEL * 100}% for RMSE for {exp_name} {model_name}")

        tsf_y = np.arange(1, lower_bound.shape[0] + 1)

        plt.plot(tsf_y, lower_bound, color=predict_color, label="lower bound")
        plt.plot(tsf_y, upper_bound, color=target_color, label="upper bound")

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        stk = int(lower_bound.shape[0] / 10) if int(lower_bound.shape[0] / 10) > 0 else 1
        plt.xticks(np.arange(1, lower_bound.shape[0] + 1, stk))
        plt.title(model_name)

        plt.savefig(pi_ci_path(num_obs, num_pred) + "/" + exp_name +
                    "_" + model_name, bbox_inches="tight")
        plt.clf()

    analyze_ci()
    analyze_pi()


def analyze_timestep_rmse(ip, op, exp_name, model_name, num_obs, num_pred):
    print(f"Calculating rmse for {exp_name}")
    if ip.shape[1] != op.shape[1]:
        raise ValueError("ip and op must have same shape")
    rmses = calculate_rmse_of_time_step(ip, op)

    print(f"rmses: {rmses}")

    plt.plot(list(range(len(rmses))), rmses)

    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.title('RMSE by day')

    plt.savefig(rmse_by_day_path(num_obs, num_pred) + "/" + exp_name + "_" + model_name, bbox_inches="tight")
    plt.clf()