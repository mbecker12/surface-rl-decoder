from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml
import json
import traceback
import os
import sys
from dataclasses import dataclass
from scipy.signal import savgol_filter as sg

@dataclass
class TrainingRun():
    job_id: int
    code_size: int
    stack_depth: int
    p_err: float
    p_msmt: float
    rl_type: str
    architecture: str
    data: pd.DataFrame = None
    duration: float = None

if __name__ == "__main__":
    training_runs = [
        TrainingRun(69312, 5, 5, 0.01, 0.01, "q", "3D Conv"),
        TrainingRun(69545, 7, 7, 0.005, 0.005, "q", "3D Conv"),
        TrainingRun(71852, 5, 5, 0.008, 0.008, "q", "2D Conv + GRU"),
        TrainingRun(71873, 5, 5, 0.01, 0.01, "ppo", "3D Conv"),
        TrainingRun(72099, 7, 7, 0.003, 0.003, "q", "2D Conv + GRU"),  
        TrainingRun(65280, 5, 5, 0.01, 0.01, "q", "2D Conv"),
    ]

    plt.rcParams.update({'font.size': 18})

    omit_job_ids = [65280]

    for i in range(len(training_runs)-1, -1, -1):
        if training_runs[i].job_id in omit_job_ids:
            training_runs.pop(i)

    all_dfs: List[pd.DataFrame] = []

    DATA_PATH = "remote_runs/manual/"


    # gather data
    for run in training_runs:
        files = glob.glob(DATA_PATH + str(run.job_id) + "/*")
        
        all_metrics_df_list = []
        for fpath in files:
            with open(fpath) as csv_file:
                metric_df: pd.DataFrame = pd.read_csv(csv_file, index_col=1)
                
            filename = fpath.split("/")[-1]
            description = filename.split(", p_error")[-1]
            metric = description.split("_")[1:]
            metric = str.join("_", metric)
            metric = metric.split(".")[0]
            metric_df["Value"] = sg(metric_df["Value"], 31, 1)
            metric_df.rename({"Value": metric}, axis="columns", inplace=True)
            duration = metric_df["Wall time"].iloc[-1] - metric_df["Wall time"].iloc[0]
            metric_df = metric_df.drop("Wall time", axis="columns")
            # print(metric_df)
            all_metrics_df_list.append(metric_df)
        
        left_df: pd.DataFrame = all_metrics_df_list[0]
        all_metrics = left_df.join(all_metrics_df_list[1:])
        all_metrics = all_metrics.reset_index(col_fill="Step")

        if run.rl_type == "ppo":
            all_metrics["Step"] = all_metrics["Step"] * 32 # hard-coded number of optimization epochs

        all_dfs.append(all_metrics)

        run.data = all_metrics
        run.duration = duration

        assert run.data is not None, f"{run.job_id=}, {all_metrics}"


    ############ Plot Ground State Rate ############

    fig, ax = plt.subplots(2, 1)

    plot_keys = {"ground_state_per_env": "Ground State Rate"}
    plot_lines = ["-", ":"]
    plot_colors = ["black", 'blue', 'orange', 'red', 'green', 'pink']


    for i, run in enumerate(training_runs):
        for j, (plot_key, plot_str) in enumerate(plot_keys.items()):

            ax[0].plot(
                    run.data["Step"],
                    run.data[plot_key],
                    label=f"d={run.code_size}, h={run.stack_depth}, {run.rl_type}, {run.architecture}",
                    color=plot_colors[i],
                    linestyle=plot_lines[j]
                )
    ax[0].set(
        title="Ground State Rate",
        ylabel=r"$p_\mathrm{Ground\; State}$",
        xlim=(0, 150_000)
    )
    ax[0].legend()

    plot_keys = {"syndromes_annihilated_per_step": "Syndrome Annihilation"}

    for i, run in enumerate(training_runs):
        for j, (plot_key, plot_str) in enumerate(plot_keys.items()):

            ax[1].plot(
                    run.data["Step"],
                    run.data[plot_key],
                    label=f"d={run.code_size}, h={run.stack_depth}, {run.rl_type}, {run.architecture}",
                    color=plot_colors[i],
                    linestyle=plot_lines[j]
                )
    ax[1].set(
        title="Syndrome Annihilation",
        ylabel="annihilation rate",
        xlabel="Steps",
        xlim=(0, 150_000)
    )
    plt.tight_layout()
    plt.savefig("plots/training_ground_state.pdf")
    plt.show()


    ############ Plot Steps ############

    fig, ax = plt.subplots(2, 1)

    plot_keys = {"number_of_steps": "Avg Steps"}
    plot_lines = ["-", ":"]
    plot_colors = ["black", 'blue', 'orange', 'red', 'green', 'pink']


    for i, run in enumerate(training_runs):
        for j, (plot_key, plot_str) in enumerate(plot_keys.items()):

            ax[0].plot(
                    run.data["Step"],
                    run.data[plot_key],
                    label=f"d={run.code_size}, h={run.stack_depth}, {run.rl_type}, {run.architecture}",
                    color=plot_colors[i],
                    linestyle=plot_lines[j]
                )
    ax[0].set(
        title="Avg Steps per Episode",
        ylabel="Steps",
        xlim=(0, 150_000)
    )
    ax[0].legend()

    plot_keys = {"median number_of_steps": "Median Steps"}

    for i, run in enumerate(training_runs):
        for j, (plot_key, plot_str) in enumerate(plot_keys.items()):

            ax[1].plot(
                    run.data["Step"],
                    run.data[plot_key],
                    label=f"d={run.code_size}, h={run.stack_depth}, {run.rl_type}, {run.architecture}",
                    color=plot_colors[i],
                    linestyle=plot_lines[j]
                )
    ax[1].set(
        title="Median Steps per Episode",
        ylabel="Step",
        xlabel="Steps",
        xlim=(0, 150_000)
    )
    plt.tight_layout()
    plt.savefig("plots/training_steps.pdf")
    plt.show()



    ############ Plot Q Values ############

    fig, ax = plt.subplots(2, 1)

    plot_keys = {"mean_q_value": "Max Q Value"}
    plot_lines = ["-", ":"]
    plot_colors = ["black", 'blue', 'orange', 'red', 'green', 'pink']


    for i, run in enumerate(training_runs):
        for j, (plot_key, plot_str) in enumerate(plot_keys.items()):

            ax[0].plot(
                    run.data["Step"],
                    run.data[plot_key],
                    label=f"d={run.code_size}, h={run.stack_depth}, {run.rl_type}, {run.architecture}",
                    color=plot_colors[i],
                    linestyle=plot_lines[j]
                )
    ax[0].set(
        title="Q values",
        ylabel="Q Value | Logit (ppo)",
        ylim=(-20, 100),
        xlim=(0, 150_000)
    )

    plot_keys = {"energy_final": "Final Energy"}

    for i, run in enumerate(training_runs):
        for j, (plot_key, plot_str) in enumerate(plot_keys.items()):

            ax[1].plot(
                    run.data["Step"],
                    run.data[plot_key],
                    label=f"d={run.code_size}, h={run.stack_depth}, {run.rl_type}, {run.architecture}",
                    color=plot_colors[i],
                    linestyle=plot_lines[j]
                )
    ax[1].set(
        title="Final Energy",
        ylabel="Syndrome Counts",
        xlabel="Steps",
        xlim=(0, 150_000)
    )
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("plots/training_q_values.pdf")
    plt.show()
