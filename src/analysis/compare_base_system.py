"""
Compare different architectures and approaches on d=5, h=5
"""
import json
import traceback
from typing import Dict, List
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import os
import sys
import yaml
import subprocess
from analysis.training_run_class import TrainingRun
from analysis.analysis_util import (
    analyze_succesful_episodes,
    load_analysis_model,
    provide_default_ppo_metadata,
)
from distributed.model_util import (
    choose_model,
    choose_old_model,
    extend_model_config,
    load_model,
)

# @dataclass
# class TrainingRun():
#     job_id: int
#     code_size: int
#     stack_depth: int
#     p_err: float
#     p_msmt: float
#     rl_type: str
#     architecture: str
#     data: pd.DataFrame = None
#     duration: float = None
#     model_name: str = None
#     model_config_file: str = "conv_agents_slim.json"
#     transfer_learning: int = 0

base_model_config_path = "src/config/model_spec/old_conv_agents.json"
base_model_path = "remote_networks/5/65280/simple_conv_5_65280.pt"

training_runs = [
    TrainingRun(69037, 5, 5, 0.0108, 0.0, "q", "3D Conv", model_name="conv3d"),
    TrainingRun(
        71852,
        5,
        5,
        0.008,
        0.008,
        "q",
        "2D Conv + GRU",
        model_name="conv2d",
        model_config_file="conv_agents_slim_gru.json",
        transfer_learning=1,
    ),
    TrainingRun(69312, 5, 5, 0.01, 0.01, "q", "3D Conv", model_name="conv3d"),
    TrainingRun(71873, 5, 5, 0.01, 0.01, "ppo", "3D Conv", model_name="conv3d"),
    TrainingRun(72409, 5, 5, 0.01, 0.01, "q", "2D Conv", model_name="conv2d"),
]

# training_runs = [
#     TrainingRun(72411, 7, 7, 0.003, 0.003, "q", "2D Conv", model_name="conv2d"),
#     TrainingRun(69545, 7, 7, 0.005, 0.005, "q", "3D Conv", model_name="conv3d"),
#     # TrainingRun(
#     #     72099,
#     #     7,
#     #     7,
#     #     0.003,
#     #     0.003,
#     #     "q",
#     #     "2D Conv + GRU",
#     #     model_name="conv2d",
#     #     model_config_file="conv_agents_slim_gru.json",
#     #     transfer_learning=1,
#     # ),
#     TrainingRun(
#         76564,
#         7,
#         7,
#         0.005,
#         0.005,
#         "q",
#         "2D Conv",
#         model_name="conv2d",
#         model_location="alvis://cephyr/NOBACKUP/groups/snic2021-23-319/falckk_networks/7/76564/",
#     ),
# ]


plt.rcParams.update({"font.size": 16})

# omit_job_ids = [65280]

CLUSTER_NETWORK_PATH = "networks"
LOCAL_NETWORK_PATH = "threshold_networks"

do_copy = False
if do_copy:
    print("Copy Data from Cluster")

    for run in training_runs:
        print(f"\tCopying {run.job_id}...")
        target_path = f"{LOCAL_NETWORK_PATH}/{run.code_size}"

        os.makedirs(target_path, exist_ok=True)
        if run.model_location is None:
            command = f"scp -r alvis://cephyr/NOBACKUP/groups/snic2021-23-319/networks/{CLUSTER_NETWORK_PATH}/{run.code_size}/{run.job_id} {target_path}"
        else:
            command = f"scp -r {run.model_location} {target_path}"
        process = subprocess.run(command.split(), stdout=subprocess.PIPE)
        print(f"{target_path}")

df_all_stats = pd.DataFrame(
    columns=["jobid", "code_size", "stack_depth", "p_err_train", "p_err"]
)

all_results_counter = 0

eval_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    LOCAL_NETWORK_PATH = "/surface-rl-decoder/networks"

run_evaluation = False
load_eval_results = True
produce_plots = True
# csv_file_path = "analysis/comparison_base_system_7.csv"

csv_file_path = "analysis/comparison_base_system_remote.csv"

n_episodes = 128
# model_name = "conv3d"
max_num_of_steps = 40
if run_evaluation:
    print("Proceed to Evaluation")
    # TODO: Need to make changes to evaluation
    # keep track of absolute numbers in df, rather than averages/fractions
    # also maybe need to keep track of disregarded episodes
    for run in training_runs:
        try:
            model = load_analysis_model(run)
        except Exception as err:
            error_traceback = traceback.format_exc()
            print("An error occurred!")
            print(error_traceback)

            continue
        p_error_list = np.arange(start=0.0001, stop=0.0120, step=0.0005)
        print(f"Job ID = {run.job_id}, Iterate over p_err...")
        for p_idx, p_err in enumerate(p_error_list):
            sys.stdout.write(f"\r{p_idx + 1:02d} / {len(p_error_list):02d}")
            p_msmt = p_err
            # batch_evaluation(..., p_error, p_msmt)

            result_dict: Dict = analyze_succesful_episodes(
                model,
                "",
                device=eval_device,
                total_n_episodes=n_episodes,
                max_num_of_steps=max_num_of_steps,
                discount_intermediate_reward=0.3,
                verbosity=2,
                p_msmt=p_msmt,
                p_err=p_err,
                code_size=run.code_size,
                stack_depth=run.stack_depth,
                rl_type=run.rl_type,
            )

            result_dict["jobid"] = run.job_id
            result_dict["code_size"] = run.code_size
            result_dict["stack_depth"] = run.stack_depth
            result_dict["p_err"] = p_err
            result_dict["avg_steps"] = result_dict["n_steps_arr"].mean()
            result_dict.pop("n_steps_arr")

            # save relevant eval stats to dataframe
            df_all_stats = df_all_stats.append(result_dict, ignore_index=True)
        print()
        print()

    print("Saving dataframe...")
    if os.path.exists(csv_file_path):
        df_all_stats.to_csv(csv_file_path, mode="a", header=False)
    else:
        df_all_stats.to_csv(csv_file_path)

if load_eval_results:
    # migt need to fix indices and so on
    # df_all_stats = pd.read_csv("analysis/analysis_results2.csv")
    print("Load Data File")
    df_all_stats = pd.read_csv(csv_file_path, index_col=0)
    # print(f"{df_all_stats=}")

if not produce_plots:
    print("Not producing result plot. Exiting...")
    sys.exit()

dfs: List[pd.DataFrame] = [
    df_all_stats.loc[
        (df_all_stats["jobid"] == run.job_id)
        | (df_all_stats["jobid"] == str(run.job_id))
    ].copy(deep=True)
    for run in training_runs
]
new_dfs = []
# TODO aggregate stats from different analysis runs
eval_key_list = [
    "total_n_episodes",
    "n_ground_states",
    "n_valid_episodes",
    "n_valid_ground_states",
    "n_valid_non_trivial_loops",
    "n_ep_w_syndromes",
    "n_ep_w_loops",
    "n_too_long",
    "n_too_long_w_loops",
    "n_too_long_w_syndromes",
    "avg_steps",
]

agg_key_list = [key for key in eval_key_list]

for df in dfs:
    # print(df)
    df = df.sort_values(by="n_ground_states", ascending=True)

    # TODO: aggregate / sum values first
    df["expected_n_err"] = (
        df["p_err"] * df["code_size"] * df["code_size"] * df["stack_depth"]
    )
    df["p_err_one_layer"] = df["p_err"] * df["stack_depth"]
    df["avg_steps"] = df["avg_steps"] * df["total_n_episodes"]

    aggregation_dict = {agg_key: ["sum"] for agg_key in agg_key_list}
    aggregation_dict["code_size"] = ["last"]
    aggregation_dict["stack_depth"] = ["last"]
    aggregation_dict["p_err"] = ["last"]
    aggregation_dict["expected_n_err"] = ["last"]
    aggregation_dict["p_err_one_layer"] = ["last"]

    groups = df.groupby(by="p_err")
    agg_groups = groups.agg(aggregation_dict)

    new_df = pd.DataFrame()

    agg_groups["weighted_avg_steps"] = (
        agg_groups["avg_steps"] / agg_groups["total_n_episodes"]
    )

    agg_groups.columns = agg_groups.columns.droplevel(1)

    print(agg_groups)

    agg_groups["logical_err_rate"] = (
        agg_groups["n_ep_w_loops"] / agg_groups["total_n_episodes"]
    )

    agg_groups["valid_success_rate"] = (
        agg_groups["n_valid_ground_states"] / agg_groups["n_valid_episodes"]
    )
    agg_groups["overall_success_rate"] = (
        agg_groups["n_ground_states"] + agg_groups["n_ep_w_syndromes"]
    ) / agg_groups["total_n_episodes"]

    agg_groups["valid_fail_rate"] = 1.0 - agg_groups["valid_success_rate"]
    agg_groups["overall_fail_rate"] = 1.0 - agg_groups["overall_success_rate"]

    agg_groups["valid_fail_rate_per_cycle"] = (
        agg_groups["valid_fail_rate"] / agg_groups["stack_depth"]
    )
    agg_groups["overall_fail_rate_per_cycle"] = (
        agg_groups["overall_fail_rate"] / agg_groups["stack_depth"]
    )
    agg_groups["logical_err_rate_per_cycle"] = (
        agg_groups["logical_err_rate"] / agg_groups["stack_depth"]
    )

    agg_groups["validity_rate"] = (
        agg_groups["n_valid_episodes"] / agg_groups["total_n_episodes"]
    )

    agg_groups["valid_avg_lifetime"] = 1.0 / agg_groups["valid_fail_rate_per_cycle"]
    agg_groups["overall_avg_lifetime"] = 1.0 / agg_groups["overall_fail_rate_per_cycle"]
    agg_groups["logical_avg_lifetime"] = 2.0 / agg_groups["logical_err_rate_per_cycle"]

    # agg_groups["fail_rate"] = 1 - agg_groups["weighted_success_rate"]
    # agg_groups["scaled_fail_rate"] = agg_groups["fail_rate"] / agg_groups["stack_depth"]

    new_dfs.append(agg_groups)

df_all = pd.concat(new_dfs)

max_x = new_dfs[0]["p_err"].max()

error_rates = df_all["p_err"]
key_success_rate = "weighted_success_rate"
title_succes_rate = "Success Rate"
key_scaled_fail_rate = "overall_fail_rate_per_cycle"
title_scaled_fail_rate = "Overall Fail Rate Per Cycle"

key_valid_fail_rate = "valid_fail_rate_per_cycle"
title_valid_fail_rate = "Fail Rate Per Cycle"

key_valid_avg_life = "valid_avg_lifetime"
title_valid_avg_life = "Average Lifetime"

key_overall_avg_life = "overall_avg_lifetime"
title_overall_avg_life = "Overall Average Lifetime"

key_logical_err_rate = "logical_err_rate_per_cycle"
title_logical_err_rate = "Logical Error Rate"

key_logical_lifetime = "logical_avg_lifetime"
title_logical_lifetime = "Average Lifetime"


plot_colors = ["#404E5C", "#F76C5E", "#E9B44C", "#7F95D1", "#CF1259"]
markers = ["o", "v", "^", "X", "d"]
ylim_lin_plot = (-1e-4, 0.008)
ylim_log_plot = (50, 1e5)


def set_text_lin_split(axis):
    axis.text(0.0053, 0.0049, "Single Qubit", rotation=27)


def set_text_log_split(axis):
    axis.text(0.0015, 100, "Single Qubit", rotation=-15)


if True:
    ################## Plot Valid Fail Rate per Cycle ##################
    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "wspace": 0, "hspace": 0.05},
    )
    ax = axes[0]
    ax1 = axes[1]

    for i, run in enumerate(training_runs):
        # print(new_dfs[i])
        y_error = np.sqrt(
            new_dfs[i][key_valid_fail_rate]
            * (1.0 - new_dfs[i][key_valid_fail_rate])
            / new_dfs[i]["n_valid_episodes"]
        )
        ax.errorbar(
            x=new_dfs[i]["p_err"]
            + np.random.normal(loc=0, scale=1.5e-5, size=len(new_dfs[i]["p_err"])),
            y=new_dfs[i][key_valid_fail_rate],
            yerr=y_error,
            fmt=".",
            linewidth=2,
            markersize=0,
            c=plot_colors[i],
            marker=markers[i],
        )

        ax.scatter(
            x=new_dfs[i]["p_err"],
            y=new_dfs[i][key_valid_fail_rate],
            label=r"$d=h=$"
            + f"{run.code_size}, {run.architecture}, {run.rl_type}, "
            + r"$p_\mathrm{err}$="
            + f"{run.p_err}, "
            + r"$p_\mathrm{msmt}$="
            + f"{run.p_msmt}",
            # s=100
            # * (new_dfs[i]["n_valid_episodes"] / new_dfs[i]["total_n_episodes"]) ** 1.2,
            c=plot_colors[i],
            marker=markers[i],
        )

        # plot disregard-fraction
        ax1.scatter(
            x=new_dfs[i]["p_err"],
            y=(1.0 - (new_dfs[i]["n_valid_episodes"] / new_dfs[i]["total_n_episodes"]))
            * 100,
            c=plot_colors[i],
            marker=markers[i],
        )
    ax.plot(
        np.linspace(new_dfs[0]["p_err"].min(), max_x, 100, endpoint=True),
        np.linspace(new_dfs[0]["p_err"].min(), max_x, 100, endpoint=True),
        "k",
    )

    # set_text_lin_split(ax)
    ax.text(0.0023, 0.0029, "Single Qubit", rotation=27)

    ax.set(
        title="Compare Strategies",
        # xlabel=r"$p_\mathrm{err}$",
        ylabel=title_valid_fail_rate,
        ylim=np.array(ylim_lin_plot) + (0, 0.001),
    )

    ax1.set(xlabel=r"$p_\mathrm{err}$", ylabel="%")

    ax1.set_xticks(np.arange(0.0, 0.013, 0.003))
    ax.set_xticks(np.arange(0.0, 0.013, 0.003))

    ax.set_yticks(np.arange(0.0, 0.0081, 0.002))
    ax1.text(0, 16, "Remaining Syndromes")

    # plt.legend()
    ax.legend()
    plt.savefig("plots/comparison_base_5.pdf", bbox_inches="tight")
    plt.show()
