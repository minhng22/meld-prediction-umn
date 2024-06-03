import time, os
import pandas as pd
import torch

from pkgs.commons import get_input_path
from pkgs.data.harvest import harvest_data_with_interpolate
from pkgs.data.dataset import SlidingWindowDataset


def run_exp(num_obs, num_pred, real_data_ratio, generalize_ratio, interpolate_amount, to_run_models):
    print(f"pre-processing data, experimenting on obs {num_obs} pred {num_pred}")
    s = time.time()
    df = pd.read_csv(get_input_path())

    exp_trains, exp_tests, exp_generalizes = harvest_data_with_interpolate(
        df, num_obs + num_pred, real_data_ratio, generalize_ratio, interpolate_amount)

    print(f"preprocessing data takes {time.time() - s} seconds")

    for model_name in to_run_models:
        print("=====================================")
        print(f"exp on model {model_name}")

        dataset = SlidingWindowDataset(exp_trains, exp_tests, exp_generalizes, num_obs, num_pred)

        print("finding best model")
        s = time.time()

        model_path = MODEL_SAVE_PATH + "/" + model_name + ".pt"
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(model_path)

        if os.path.exists(model_path):
            print('best model exists')
            best_model = torch.load(model_path)
        else:
            dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            best_model = ex_optuna(dl, model_name, USE_FILLER_MODEL)

            print(f"finding best model among trials of {model_name}")

            for i in range(N_TRIALS):
                cp_model_path = MODEL_CP_SAVE_PATH + f"/{model_name}_trial_{i}.pt"

                if not os.path.exists(cp_model_path):
                    raise Exception("no model here")

                print(f'study model from trial {i}')
                m = torch.load(cp_model_path)

                print("evaluating on test data")

                test_ips = torch.from_numpy(dataset.get_test_ips()).float()
                tests_ops = m(test_ips.to(DEVICE)).to("cpu").detach().numpy()
                # inverse meld
                tests_ops_full = inverse_scale_ops(test_ips[:, :, 0], tests_ops,
                                                   dataset.meld_sc)
                tests_ops = tests_ops_full[:, num_obs:]
                tests = dataset.get_original_meld_test()[:, num_obs:]

                print(f"test shape: {tests.shape} {tests_ops.shape}")
                curr_r2 = r2_score(tests, tests_ops)
                curr_rmse = mean_squared_error(tests, tests_ops, squared=False)

                print(f"R-square is: {curr_r2}")
                print(f"RMSE is: {curr_rmse}")

                if best_model == None:
                    best_model = m
                    continue

                best_model_ops = best_model(test_ips.to(DEVICE)).to("cpu").detach().numpy()
                best_model_ops_full = inverse_scale_ops(test_ips[:, :, 0], best_model_ops,
                                                        dataset.meld_sc)
                best_model_ops = best_model_ops_full[:, num_obs:]
                print(f"check best model op shape {best_model_ops.shape}")
                best_rmse = mean_squared_error(tests, best_model_ops, squared=False)

                if curr_rmse > best_rmse:
                    best_model = m

        print(f"evaluate best model of {model_name}")
        print(f"model params: {best_model}")

        print("evaluating on test data")
        test_ips = torch.from_numpy(dataset.get_test_ips()).float()
        tests_ops = best_model(test_ips.to(DEVICE))

        # inverse meld
        tests_ops_full = inverse_scale_ops(test_ips[:, :, 0], tests_ops.to("cpu").detach().numpy(), dataset.meld_sc)
        tests_ops = tests_ops_full[:, num_obs:]
        tests = dataset.get_original_meld_test()[:, num_obs:]

        print(f"test shape: {tests.shape} {tests_ops.shape}")
        print(f"R-square is: {r2_score(tests, tests_ops):.4f}")
        print(
            f"RMSE is: {mean_squared_error(tests, tests_ops, squared=False):.4f}")

        analyze_timestep_rmse(tests, tests_ops, "test", model_name)
        analyze_ci_and_pi(tests, tests_ops, "test", model_name)

        plot_name = f"{model_name} R-square :{round(r2_score(tests, tests_ops), 2)} RMSE {round(mean_squared_error(tests, tests_ops, squared=False), 2)}"

        plot_box(tests, tests_ops, plot_name, model_name, "test")
        plot_line(dataset.get_original_meld_test(),
                  tests_ops_full, plot_name, model_name, "test")

        print("evaluating on generalize data")
        generalizes_ips = torch.from_numpy(
            dataset.get_generalize_ips()).float()
        generalizes_ops = best_model(generalizes_ips.to(DEVICE))

        # inverse meld
        generalizes_ops_full = inverse_scale_ops(
            generalizes_ips[:, :, 0], generalizes_ops.to("cpu").detach().numpy(), dataset.meld_sc)
        generalizes_ops = generalizes_ops_full[:, num_obs:]
        generalizes = dataset.get_original_meld_generalize()[:, num_obs:]

        print(f"R-square is: {r2_score(generalizes, generalizes_ops):.4f}")
        print(
            f"RMSE is {mean_squared_error(generalizes, generalizes_ops, squared=False):.4f}"
        )

        analyze_timestep_rmse(generalizes, generalizes_ops, "generalize", model_name)
        analyze_ci_and_pi(generalizes, generalizes_ops, "generalize", model_name)

        plot_name = f"{model_name} R-square :{round(r2_score(generalizes, generalizes_ops), 2)} " \
                    f"RMSE {round(mean_squared_error(generalizes, generalizes_ops, squared=False), 2)}"

        plot_box(generalizes, generalizes_ops, plot_name, model_name, "generalize")
        plot_line(dataset.get_original_meld_generalize(),
                  generalizes_ops_full, plot_name, model_name, "generalize")

        torch.save(best_model, model_path)

        print(
            f"Model experiment takes {time.time() - s} seconds or {int((time.time() - s) / 60)} minutes")
        print(f"Finished experiment on model {model_name}")
        print("=====================================================")