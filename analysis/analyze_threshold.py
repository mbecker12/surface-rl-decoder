# theoretically, we have all the runs for d=3,5,7
# Still some uncertainties in error rate (p_{err} or p_{err}^{one_layer})
from typing import List

from torch._C import AggregationType
from analysis_util import analyze_succesful_episodes
import sys
import os
import multiprocessing as mp
import subprocess
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import torch
import yaml
from evaluation.batch_evaluation import RESULT_KEY_COUNTS, RESULT_KEY_ENERGY, RESULT_KEY_EPISODE, RESULT_KEY_Q_VALUE_STATS, RESULT_KEY_RATES, batch_evaluation

from distributed.model_util import choose_model, choose_old_model, load_model
plt.rcParams.update({'font.size': 18})
# 69366	3	3	0.05	0.05
# 69312	5	5	0.01	0.01
# 69545	7	7	0.005	0.005
# 69308	7	7	0.01	0.01

job_ids = [
    69366,
    69312,
    69545,
    70425
]

CLUSTER_NETWORK_PATH = "networks"
LOCAL_NETWORK_PATH = "threshold_networks"

do_copy = False
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
    "jobid", "code_size", "stack_depth", "p_err_train", "p_err", 
    "success_rate", "logical_err_rate", "remaining_syndrome_rate", "ground_state_rate",
    "solved_w_syndrome_rate", "n_episodes"
    ])

all_results_counter = 0
n_episodes = 1024
model_name = "conv3d"
eval_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    LOCAL_NETWORK_PATH = "/surface-rl-decoder/networks"

run_evaluation = False
load_eval_results = True
produce_plots = False
csv_file_path = "analysis/thresh_analysis_results_small_p_err.csv"

max_num_of_steps = 36
if run_evaluation:
    print("Proceed to Evaluation")
    # TODO: Need to make changes to evaluation
    # keep track of absolute numbers in df, rather than averages/fractions
    # also maybe need to keep track of disregarded episodes
    for code_size in (9, ):
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
                if int(jid) < 70000:
                    model = choose_old_model(model_name, model_config)
                else:
                    model = choose_model(model_name, model_config)

                model, _, _ = load_model(model, old_model_path, model_device=eval_device)
            except Exception as err:
                error_traceback = traceback.format_exc()
                print("An error occurred!")
                print(error_traceback)
                
                continue
            p_error_list = np.arange(start=0.0001, stop=0.0120, step=0.0005)
            for p_err in p_error_list:
                p_msmt = p_err
                # batch_evaluation(..., p_error, p_msmt)

                (
                    total_n_episodes,
                    counter_successful_episodes,
                    counter_logical_errors,
                    counter_syndrome_left,
                    counter_ground_state,
                    counter_solved_w_syndrome_left,
                    n_steps_arr
                ) = analyze_succesful_episodes(
                    model,
                    "",
                    device=eval_device,
                    total_n_episodes=n_episodes,
                    max_num_of_steps=max_num_of_steps,
                    discount_intermediate_reward=0.3,
                    verbosity=2,
                    p_msmt=p_msmt,
                    p_err=p_err,
                    code_size=code_size,
                    stack_depth=stack_depth
                )

                
                # save relevant eval stats to dataframe
                df_all_stats = df_all_stats.append(
                    {
                        "jobid": jid,
                        "code_size": code_size,
                        "stack_depth": stack_depth,
                        "p_err_train": p_err_train,
                        "p_err": p_err,
                        "success_rate": counter_successful_episodes / total_n_episodes,
                        "logical_err_rate": counter_logical_errors / total_n_episodes,
                        "remaining_syndrome_rate": counter_syndrome_left / total_n_episodes,
                        "ground_state_rate": counter_ground_state / total_n_episodes,
                        "solved_w_syndrome_rate": counter_solved_w_syndrome_left / total_n_episodes,
                        "avg_steps": n_steps_arr.mean(),
                        "n_episodes": n_episodes
                    },
                    ignore_index=True
                )
    print("Saving dataframe...")
    if os.path.exists(csv_file_path):
        df_all_stats.to_csv(csv_file_path, mode='a', header=False)
    else:
        df_all_stats.to_csv(csv_file_path)

if load_eval_results:
    # migt need to fix indices and so on
    # df_all_stats = pd.read_csv("analysis/analysis_results2.csv")
    df_all_stats = pd.read_csv(csv_file_path, index_col=0)

if not produce_plots:
    print("Not producing result plot. Exiting...")
    sys.exit()
print(df_all_stats)
# split df into sensible groups

dfs: List[pd.DataFrame] = [df_all_stats.loc[(df_all_stats["jobid"] == jid) | (df_all_stats["jobid"] == str(jid))] for jid in job_ids]
new_dfs = []
# TODO aggregate stats from different analysis runs
eval_key_list = [
            "success_rate",
            "logical_err_rate",
            "remaining_syndrome_rate"
        ]

agg_key_list = ["weighted_" + key for key in eval_key_list]

for df in dfs:
    if int(df["jobid"].iloc[0]) == 69308:
        continue
    df = df.sort_values(by="success_rate", ascending=True)
    df["expected_n_err"] = df["p_err"] * df["code_size"] * df["code_size"] * df["stack_depth"]
    df["p_err_one_layer"] = df["p_err"] * df["stack_depth"]

    for j, agg_key in enumerate(agg_key_list):
        df[agg_key] = df[eval_key_list[j]] * df["n_episodes"]
    
    aggregation_dict = {agg_key: ["sum"] for agg_key in agg_key_list}
    aggregation_dict["n_episodes"] = ["sum"]
    aggregation_dict["code_size"] = ["last"]
    aggregation_dict["stack_depth"] = ["mean"]
    aggregation_dict["p_err"] = ["mean"]
    aggregation_dict["expected_n_err"] = ["mean"]
    aggregation_dict["p_err_one_layer"] = ["mean"]
    groups = df.groupby(by="p_err")
    agg_groups = groups.agg(aggregation_dict)
    print(f"{type(agg_groups)=}")
    new_df = pd.DataFrame()
    for agg_key in agg_key_list:
        agg_groups[agg_key] /= agg_groups["n_episodes"]

    agg_groups.columns = agg_groups.columns.droplevel(1)
    
    print(agg_groups)
    
    agg_groups["fail_rate"] = 1 - agg_groups["weighted_success_rate"]
    agg_groups["scaled_fail_rate"] = agg_groups["fail_rate"] / agg_groups["stack_depth"]

    new_dfs.append(agg_groups)

df_all = pd.concat(new_dfs)

error_rates = df_all["p_err"]
key_success_rate = "weighted_success_rate"
title_succes_rate = "Success Rate"
key_scaled_fail_rate = "scaled_fail_rate"
title_scaled_fail_rate = "Scaled Fail Rate"


fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_scaled_fail_rate],
        label=f"d={code_size}, h={stack_depth}")
ax.plot(
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k',
    label="One Qubit"
)
ax.set(title=title_scaled_fail_rate)
ax.set(xlabel=r"$p_\mathrm{err}$", ylabel=title_scaled_fail_rate)


plt.legend()
plt.tight_layout()
plt.savefig("plots/threshold_scaled_fail_rate_p_err.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=1.0 / new_dfs[i][key_scaled_fail_rate],
        label=f"d={code_size}, h={stack_depth}")
ax.plot(
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    1.0 / np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k',
    label="One Qubit"
)
ax.set(title="Average Qubit Lifetime")
ax.set(xlabel=r"$p_\mathrm{err}$", ylabel="Lifetime / Cycles", ylim=(0, 1000))

plt.legend()
plt.tight_layout()
plt.savefig("plots/threshold_qubit_lifetime.pdf")
plt.show()


fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_success_rate],
        label=f"d={code_size}, h={stack_depth}")

ax.set(title=title_succes_rate)
ax.set(xlabel=r"$p_\mathrm{err}$")

plt.legend()
plt.tight_layout()
plt.savefig("plots/threshold_p_err.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["expected_n_err"],
        y=new_dfs[i][key_success_rate],
        label=f"d={code_size}, h={stack_depth}")
ax.set(title=title_succes_rate)
ax.set(xlabel=r"$\overline{n_\mathrm{err}}$")

plt.legend()
plt.tight_layout()
plt.savefig("plots/threshold_expected_n_err.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err_one_layer"],
        y=new_dfs[i][key_success_rate],
        label=f"d={code_size}, h={stack_depth}")
ax.set(title=title_succes_rate)
ax.set(xlabel=r"$p_\mathrm{err}^\mathrm{one \; layer}$")

plt.legend()
plt.tight_layout()
plt.savefig("plots/threshold_p_err_one_layer.pdf")
plt.show()