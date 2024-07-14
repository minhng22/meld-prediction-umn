import pandas as pd
from pkgs.commons import model_performance_tables

h_5_f_1 = pd.read_excel(f"{model_performance_tables}/h=5,f=1.xlsx")
h_5_f_1.ffill(axis=0, inplace=True)

h_5_f_3 = pd.read_excel(f"{model_performance_tables}/h=5,f=3.xlsx")
h_5_f_3.ffill(axis=0, inplace=True)

h_5_f_5 = pd.read_excel(f"{model_performance_tables}/h=5,f=5.xlsx")
h_5_f_5.ffill(axis=0, inplace=True)

h_5_f_7 = pd.read_excel(f"{model_performance_tables}/h=5,f=7.xlsx")
h_5_f_7.ffill(axis=0, inplace=True)

models = [
    "RFR",
    "EVR",
    "XGBoost",
    "LSTM",
    "Attention LSTM",
    "TCN",
    "TCN LSTM",
    "Linear Regression"
]

for setup_name in ["test", "generalize"]:
    for metric in ["R-square", "RMSE"]:
        performance_df = [{'f': '1', 'RFR': '+0%', 'EVR': '+0%', 'XGBoost': '+0%', 'LSTM': '+0%',
                           'Attention LSTM': '+0%', 'TCN': '+0%', 'TCN LSTM': '+0%', 'Linear Regression': '+0%'}]
        for f in [3, 5, 7]:
            p_row = {'f': str(f)}
            H = h_5_f_3 if f == 3 else h_5_f_5 if f == 5 else h_5_f_7
            for model_name in models:
                base_perf = h_5_f_1[
                    (h_5_f_1['Model_name'] == model_name) & (h_5_f_1['Dataset'] == setup_name)][metric].values[0]
                perf = H[
                    (H['Model_name'] == model_name) & (H['Dataset'] == setup_name)][metric].values[0]
                print(f"model: {model_name} base_perf: {base_perf} perf: {perf}")
                diff_perc = (perf - base_perf) / base_perf * 100
                p_row[model_name] = f"+{diff_perc:.2f}%" if diff_perc > 0 else f"{diff_perc:.2f}%"
            performance_df.append(p_row)

        performance_df = pd.DataFrame(performance_df)
        print(f"performance df:\n{performance_df}")

        performance_df.to_excel(f"{model_performance_tables}/{setup_name}_{metric}.xlsx", index=False)

