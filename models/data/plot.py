import numpy as np
import matplotlib.pyplot as plt
import os
from models.commons import get_figs_path
from models.data.commons import generate_timestep_for_plot


def plot_data(train, test, generalize, num_observed, num_predicted, label_post_text = ""):
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

        figPath = get_figs_path() + f'/obs_{num_observed}_pred_{num_predicted}/analyze_{label}_{label_post_text}.png'
        print(f"Saving figure to {figPath}")

        plt.savefig(figPath, bbox_inches="tight")
        plt.clf()

    plot(train, "train")
    plot(test, "test")
    plot(generalize, "generalize")
