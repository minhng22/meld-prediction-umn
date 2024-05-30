from sklearn.cluster import KMeans
from store import SEASONAL_DEC_FIG_PATH, INPUT_PATH
from preprocess import interpolate_with_sliding_window, interpolate
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from utils import NUM_OBS, NUM_PRED
from datetime import timedelta

import matplotlib.pyplot as plt
import os

def perturb_data(data, noise_level):
    noise = np.random.normal(scale=noise_level, size=data.shape)
    perturbed_data = data + noise
    return perturbed_data


def calculate_similarity(cluster_labels1, cluster_labels2):
    agreement = np.sum(cluster_labels1 == cluster_labels2)
    return agreement / len(cluster_labels1)


def cluster_analysis(data, dataset_name):
    num_clusters = 2  # Number of clusters to evaluate
    max_clusters = 10  # Maximum number of clusters to evaluate
    best_silhouette_score = -1
    best_num_clusters = 0

    print(f"Evaluate cluster stability of dataset: {dataset_name}")

    k_values = range(num_clusters, max_clusters + 1)

    for _, n_cluster in enumerate(k_values):
        kmeans = KMeans(n_clusters=n_cluster, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        print(f"Number of clusters: {n_cluster}. Silhouette score: {silhouette_avg:.4f}")

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = n_cluster

    print(
        f"Best number of clusters: {best_num_clusters}. Best silhouette score: {best_silhouette_score:.4f}")

    if not os.path.exists(SEASONAL_DEC_FIG_PATH):
        os.makedirs(SEASONAL_DEC_FIG_PATH)


def density_analysis(data, dataset_name):
    # Create a histogram
    plt.hist(data, bins=30)

    # Set plot title and labels
    plt.title(f"Data Distribution. Total records: {len(data)}")
    plt.xlabel("Data")
    plt.ylabel("Frequency")

    if not os.path.exists(SEASONAL_DEC_FIG_PATH):
        os.makedirs(SEASONAL_DEC_FIG_PATH)

    plt.savefig(SEASONAL_DEC_FIG_PATH + "/" + "density_analysis" +
                "_" + dataset_name, bbox_inches='tight')
    plt.clf()


def id_as_feature(df):
    patient_id_mapping = {}
    i = 100

    for patient_id in df['patient_id'].unique():
        patient_id_mapping[patient_id] = i
        i += 1

    df['patient_id'] = df['patient_id'].map(patient_id_mapping)
    df.to_csv("./data/interpolated_with_numeric_id.csv", index=False)


# Determine which MELD score appears in the most number of patients
# This also shows patients whose data are measured every day
def dominant_score_analysis(df):
    def round_list(l):
        return set([round(x) for x in l])

    def print_top(d):
        # Sort the dictionary by values in descending order and get the top 10 keys
        top_10_keys = sorted(d, key=lambda k: d[k], reverse=True)[:10]
        # Print the top 10 keys
        for key in top_10_keys:
            print(f"{key}: {d[key]}")

    print(f"Total number of patients: {len(df['patient_id'].unique())}")
    score_count = {}
    for score in round_list(df['score'].unique()):
        score_count[score] = 0
    for _, group in df.groupby('patient_id'):
        for score in round_list(group['score'].unique()):
            score_count[score] += 1
    print_top(score_count)


def fill_with_minus1(df):
    df.loc[df['is_original'] == False, 'score'] = -1
    df.to_csv("./data/interpolated_fill_minus_1.csv", index=False)


def plot_data(train, test, generalize):
    train_avg = np.average(train, axis=0)
    test_avg = np.average(test, axis=0)
    gen_avg = np.average(generalize, axis=0)

    tsf_y = np.arange(1, train_avg.shape[0] + 1)

    plt.plot(tsf_y, train_avg, color="blue", label="train")
    plt.plot(tsf_y, test_avg, color="red", label="test")
    plt.plot(tsf_y, gen_avg, color="black", label="generalize")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    stk = int(train_avg.shape[0] / 10) if int(train_avg.shape[0] / 10) > 0 else 1
    plt.xticks(np.arange(1, train_avg.shape[0] + 1, stk))
    plt.title(f'Dataset analysis obs {NUM_OBS} pred {NUM_PRED}')

    figPath = "./figs/analyze"
    if not os.path.exists(figPath):
        os.mkdir(figPath)

    plt.savefig(figPath + f'/analyze_{NUM_OBS}_{NUM_PRED}.png', bbox_inches="tight")
    plt.clf()

def interpolate_with_sliding_window_graph():
    df = pd.read_csv(INPUT_PATH)
    train, test, generalize = interpolate_with_sliding_window(
        df, NUM_OBS + NUM_PRED)
    # 2nd feature of data is time
    train, test, generalize = train[:, :, :1], test[:, :, :1], generalize[:, :, :1]
    plot_data(train, test, generalize)

def analyze_seq_len(df=None):
    if df is None:
        df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    for N in range (1, 61):
        # Sort the DataFrame by 'patient_id' and 'timestamp'
        df = df.sort_values(['patient_id', 'timestamp'])

        # Group by 'patient_id' and 'timestamp', then calculate the mean score for each group
        df = df.groupby(['patient_id', 'timestamp'])['score'].mean().reset_index()

        # Calculate the difference in timestamps with the previous row
        df['diff'] = df['timestamp'] - df['timestamp'].shift(1)

        # Mark the start of a new time series
        df['start_new_series'] = (df['diff'] != timedelta(days=N)) | (df['diff'].isnull())

        # Create a unique series ID for each series
        df['series_id'] = (df['start_new_series'].cumsum())

        # Group by 'id' and 'series_id'
        grouped = df.groupby(['patient_id', 'series_id'])

        # Filter out series with length less than 2
        consecutive_series = grouped.filter(lambda x: len(x) >= 2)
        print('consecutive_series =', consecutive_series)
        consecutive_series.to_csv(f'len_analysis_consecutive_series_{N}d.csv')

        series_length_counts = consecutive_series.groupby(consecutive_series.groupby('series_id').transform('count')['patient_id'])[
            'series_id'].nunique()
        series_length_counts_df = series_length_counts.reset_index()
        series_length_counts_df.columns = ['series_length', 'count']
        print('series_length_counts_df ', series_length_counts_df)
        # first columns is series length. second column is how many series with that len exists.
        series_length_counts_df.to_csv(f'len_analysis_series_length_counts_{N}d.csv')

def visualize_data():
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort the DataFrame by 'patient_id' and 'timestamp'
    df = df.sort_values(['patient_id', 'timestamp'])

    # Plot
    plt.figure(figsize=(12, 6))
    N = 20
    i = 0
    for _, group in df.groupby('patient_id'):
        i += 1
        if N is not None and i == N:
            break
        plt.plot(group['timestamp'], group['score'], label=f'Patient {i}')

    plt.xlabel('Timestamp')
    plt.ylabel('Score')
    plt.title(f'Score Trajectory Over Time for random {i} Patients')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.grid(True)

    # Save the plot
    plt.savefig(f'score_trajectory_{N}.png')
    plt.clf()


def visualize_mean():
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate the average score for each timestamp
    cohort_trajectory = df.groupby('timestamp')['score'].mean().reset_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(cohort_trajectory['timestamp'], cohort_trajectory['score'], label='Cohort Trajectory', color='black')

    plt.xlabel('Timestamp')
    plt.ylabel('Average Score')
    plt.title('Cohort Score Trajectory Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig('cohort_score_trajectory.png')
    plt.show()

def find_most_common_diff():
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by 'id' and 'timestamp' to ensure correct order for calculations
    df = df.sort_values(by=['patient_id', 'timestamp'])

    # Calculate the difference in days between consecutive timestamps for each patient
    df['time_diff_days'] = df.groupby('patient_id')['timestamp'].diff().dt.days

    # Filter out 0 and NaN values
    df_filtered = df[df['time_diff_days'].notna() & (df['time_diff_days'] != 0)]

    # Group by 'time_diff_days' and count number of rows for each
    count_by_time_diff = df_filtered.groupby('time_diff_days').size().reset_index(name='count')

    print("Count of rows for each time difference:")
    print(count_by_time_diff)

    # Find the most common time differences
    common_time_diff = df_filtered['time_diff_days'].mode()

    print("Most common time difference in days:", common_time_diff[0])

def test_interpolate():
    df = pd.read_csv(INPUT_PATH)
    df = interpolate(df, "d")
    analyze_seq_len(df)

if __name__ == "__main__":
    interpolate_with_sliding_window_graph()