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

        figPath = (figs_path(num_observed, num_predicted) +
                   f'/analyze_{label}_{label_post_text}.png')
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


def analyze_ci_and_pi(target, prediction, exp_name, model_name, num_obs, num_pred):
    CONFIDENCE_LEVEL = 0.95  # common value for confidence level

    def analyze_ci():
        print(f"Calculating confidence interval for {exp_name} {model_name}")
        data = calculate_rmse_of_time_step(target, prediction)
        print(f"RMSE of timestep data: {data}")

        n = len(data)
        mean = np.mean(data)
        standard_error = sem(data)
        t_value = t.ppf((1 + CONFIDENCE_LEVEL) / 2, n - 1)
        margin_of_error = t_value * standard_error

        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error

        print(
            f"Confidence interval {CONFIDENCE_LEVEL * 100}% for RMSE for {exp_name} {model_name}.\n" 
            f"data: {data}\n"
            f"lower_bound {lower_bound:.2f} upper_bound {upper_bound:.2f} mean {mean:.2f} standard_error {standard_error:.2f}\n")

    def analyze_pi():
        print(f"Calculating prediction interval for {exp_name} {model_name}")
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