# theoretically, we have all the runs for d=3,5,7
# Still some uncertainties in error rate (p_{err} or p_{err}^{one_layer})
import os
import subprocess
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch
import yaml
from evaluation.batch_evaluation import RESULT_KEY_COUNTS, RESULT_KEY_ENERGY, RESULT_KEY_EPISODE, RESULT_KEY_Q_VALUE_STATS, RESULT_KEY_RATES, batch_evaluation

from distributed.model_util import choose_model, load_model
# 69366	3	3	0.05	0.05
# 69312	5	5	0.01	0.01
# 69545	7	7	0.005	0.005
# 69308	7	7	0.01	0.01

job_ids = [
    69366,
    69312,
    69545,
    69308,
]

CLUSTER_NETWORK_PATH = "networks"
LOCAL_NETWORK_PATH = "threshold_networks"

do_copy = True
if do_copy:
    print("Copy Data from Cluster")
    
    for jid in job_ids:
        print(f"\tCopying {jid}...")
        for code_size in (3, 5, 7, 9):
            try:
                target_path = f"{LOCAL_NETWORK_PATH}/{code_size}"
                
                os.makedirs(target_path, exist_ok=True)
                command = f"scp -r alvis://cephyr/users/gunter/Alvis/surface-rl-decoder/{CLUSTER_NETWORK_PATH}/{code_size}/{jid} {target_path}"
                process = subprocess.run(
                    command.split(),
                    stdout=subprocess.PIPE
                )
                print(f"{target_path}")
            except Exception as err:
                print(err)
                continue

df_all_stats = pd.DataFrame(columns=[
    "jobid", "code_size", "stack_depth", "p_err_train", "p_err", "logical_err", "remaining_syndromes", "ground_states", 
    "q_value", "steps", "final_energy", "energy_diff",
    "median_logical_err", "median_remaining_syndromes",
    "median_steps", "median_final_energy", "median_energy_diff"
    ])
all_results_counter = 0
n_episodes = 256
model_name = "conv3d"
eval_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    LOCAL_NETWORK_PATH = "/surface-rl-decoder/networks"

run_evaluation = True
load_eval_results = False

max_num_of_steps = 40
if run_evaluation:
    print("Proceed to Evaluation")
    for code_size in (3, 5, 7):
        stack_depth = code_size
        os.environ["CONFIG_ENV_SIZE"] = str(code_size)
        os.environ["CONFIG_ENV_STACK_DEPTH"] = str(stack_depth)
        network_list = glob.glob(f"{LOCAL_NETWORK_PATH}/{code_size}/*")
        print(network_list)
        for load_path in network_list:
            print(f"{load_path}")
            jid = load_path.split("/")[-1]
            model_config_path = load_path + f"/{model_name}_{code_size}_meta.yaml"
            old_model_path = load_path + f"/{model_name}_{code_size}_{jid}.pt"
            with open(model_config_path, "r") as yaml_file:
                general_config = yaml.load(yaml_file)
                model_config = general_config["network"]
                model_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
                p_err_train = general_config["global"]["env"]["p_error"]
            # load model
            print("Load Model")
            try:
                model = choose_model(model_name, model_config)
                model, _, _ = load_model(model, old_model_path, model_device=eval_device)
            except Exception as err:
                error_traceback = traceback.format_exc()
                print("An error occurred!")
                print(error_traceback)
                
                continue
            p_error_list = np.arange(start=0.001, stop=0.016, step=0.001)
            for p_err in p_error_list:
                p_msmt = p_err
                # batch_evaluation(..., p_error, p_msmt)
                eval_results, all_q_values = batch_evaluation(
                    model,
                    "",
                    eval_device,
                    num_of_random_episodes=n_episodes,
                    num_of_user_episodes=0,
                    max_num_of_steps=max_num_of_steps,
                    discount_intermediate_reward=0.3,
                    verbosity=2,
                    p_msmt=p_msmt,
                    p_err=p_err,
                    post_run=False,
                    code_size=code_size,
                    stack_depth=stack_depth
                )

                # collect eval stats
                logical_err = eval_results[RESULT_KEY_EPISODE]["logical_errors_per_episode"]
                remaining_syndromes = eval_results[RESULT_KEY_EPISODE]["remaining_syndromes_per_episode"]
                mean_q_value = eval_results[RESULT_KEY_Q_VALUE_STATS]["mean_q_value"]
                energy_difference = eval_results[RESULT_KEY_COUNTS]["energy_difference"]
                final_energy = eval_results[RESULT_KEY_COUNTS]["energy_final"]
                ground_state = eval_results[RESULT_KEY_RATES]["ground_state_per_env"]
                number_of_steps = eval_results[RESULT_KEY_ENERGY]["number_of_steps"]
                median_logical_err = eval_results[RESULT_KEY_EPISODE]["median logical_errors_per_episode"]
                median_remaining_syndromes = eval_results[RESULT_KEY_EPISODE]["median remaining_syndromes_per_episode"]
                median_energy_difference = eval_results[RESULT_KEY_COUNTS]["median energy_difference"]
                median_final_energy = eval_results[RESULT_KEY_COUNTS]["median energy_final"]
                median_number_of_steps = eval_results[RESULT_KEY_ENERGY]["median number_of_steps"]
                median_ground_state = eval_results[RESULT_KEY_RATES]["median ground_state_per_env"]

                
                # save relevant eval stats to dataframe
                df_all_stats = df_all_stats.append(
                    {
                        "jobid": jid,
                        "code_size": code_size,
                        "stack_depth": stack_depth,
                        "p_err_train": p_err_train,
                        "p_err": p_err,
                        "logical_err": logical_err,
                        "remaining_syndromes": remaining_syndromes,
                        "ground_state": ground_state,
                        "q_value": mean_q_value,
                        "final_energy": final_energy,
                        "steps": number_of_steps,
                        "energy_diff": energy_difference,
                        "median_logical_err": median_logical_err,
                        "median_remaining_syndromes": median_remaining_syndromes,
                        "median_final_energy": median_final_energy,
                        "median_steps": median_number_of_steps,
                        "median_energy_diff": median_energy_difference,
                        "median_ground_state": median_ground_state
                    },
                    ignore_index=True
                )
                
    df_all_stats.to_csv("analysis/analysis_results.csv")

if load_eval_results:
    # migt need to fix indices and so on
    # df_all_stats = pd.read_csv("analysis/analysis_results2.csv")
    df_all_stats = pd.read_csv("threshold_networks/analysis_results2.csv", index_col=0)
    print(df_all_stats)

# split df into sensible groups

dfs = [df_all_stats.loc[df_all_stats["code_size"] == i] for i in (3, 5, 7)]
new_dfs = []
for df in dfs:
    df = df.loc[(df["median_steps"] < max_num_of_steps)]
    df = df.sort_values(by="ground_state", ascending=True)
    df = df.drop_duplicates(["code_size", "stack_depth", "p_err"], keep="first")
    df = df.drop(columns=["median_remaining_syndromes", "median_final_energy", "median_energy_diff"])
    new_dfs.append(df)

# TODO: filter by p_err_train
# TODO: need more evaluation samples

# need to filter best performing model for each d as well
for i, d in enumerate((3, 5, 7)):
    print(new_dfs[i])
    plt.scatter(new_dfs[i]["p_err"], new_dfs[i]["ground_state"], label=f"d={d}")

plt.legend()
plt.show()