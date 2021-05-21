# theoretically, we have all the runs for d=3,5,7
# Still some uncertainties in error rate (p_{err} or p_{err}^{one_layer})
from typing import Dict
from analysis_util import analyze_succesful_episodes
import os
import sys
import multiprocessing as mp
import subprocess
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import glob
import torch
import yaml
from evaluation.batch_evaluation import RESULT_KEY_COUNTS, RESULT_KEY_ENERGY, RESULT_KEY_EPISODE, RESULT_KEY_Q_VALUE_STATS, RESULT_KEY_RATES, batch_evaluation

from distributed.model_util import choose_model, choose_old_model, load_model

plt.rcParams.update({'font.size': 18})

# 3D Conv
job_ids = [
    70282,
    70278,
    70283,
    70286,
    70287,
]

# 2D Conv
job_ids = [
    72407,
    72408,
    72409,
    72410,
]

omit_job_ids = []

# 3D Conv
stack_depths = [3, 5, 7, 9, 11]
# 2D Conv
stack_depths = [2, 3, 5, 7]

job_id_mapping = {jid: stack_depths[i] for i, jid in enumerate(job_ids)}
print(f"{job_id_mapping=}")
CLUSTER_NETWORK_PATH = "networks"
LOCAL_NETWORK_PATH = "stack_depth_networks"

do_copy = False
if do_copy:
    print("Copy Data from Cluster")
    
    for jid in job_ids:
        print(f"\tCopying {jid}...")
        for code_size in (5, ):
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
    "jobid", "code_size", "stack_depth", "p_err_train", "p_err"
    ])

all_results_counter = 0
n_episodes = 128
# model_name = "conv3d"
model_name = "conv2d"
eval_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    LOCAL_NETWORK_PATH = "/surface-rl-decoder/networks"

run_evaluation = True
load_eval_results = True
produce_plots = True
csv_file_path = "analysis/depth2_analysis_results_p_err.csv"

max_num_of_steps = 32
if run_evaluation:
    print("Proceed to Evaluation")
    for jid, stack_depth in job_id_mapping.items():
    # for code_size in (5, ):
        print(f"\n{jid=}, {stack_depth=}\n")
        # stack_depth = code_size
        code_size = 5
        os.environ["CONFIG_ENV_SIZE"] = str(code_size)
        os.environ["CONFIG_ENV_STACK_DEPTH"] = str(stack_depth)
        network_list = glob.glob(f"{LOCAL_NETWORK_PATH}/{code_size}/*")
        # print(network_list)
        for load_path in network_list:
            if str(jid) not in load_path:
                continue
            # print(f"{load_path=}")
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
                # model = choose_old_model(model_name, model_config)
                # TODO: may have to choose old model
                model = choose_model(model_name, model_config, transfer_learning=0)
                model, _, _ = load_model(model, old_model_path, model_device=eval_device)
            except Exception as err:
                error_traceback = traceback.format_exc()
                print("An error occurred!")
                print(error_traceback)
                
                continue
            p_error_list = np.arange(start=0.0001, stop=0.0120, step=0.0005)
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
                    code_size=code_size,
                    stack_depth=stack_depth
                )

                result_dict["jobid"]= jid
                result_dict["code_size"]= code_size
                result_dict["stack_depth"]= stack_depth
                result_dict["p_err_train"]= p_err_train
                result_dict["p_err"]= p_err
                result_dict["avg_steps"] = result_dict["n_steps_arr"].mean()
                result_dict.pop("n_steps_arr")
                
                # save relevant eval stats to dataframe
                df_all_stats = df_all_stats.append(result_dict, ignore_index=True)
            print()
            print()
    print("Saving dataframe...")
    if os.path.exists(csv_file_path):
        df_all_stats.to_csv(csv_file_path, mode='a', header=False)
    else:
        df_all_stats.to_csv(csv_file_path)

if load_eval_results:
    # migt need to fix indices and so on
    # df_all_stats = pd.read_csv("analysis/analysis_results2.csv")
    print("Load Data File")
    df_all_stats = pd.read_csv(csv_file_path, index_col=0)
    print(f"{df_all_stats=}")

if not produce_plots:
    print("Not producing result plot. Exiting...")
    sys.exit()

print(df_all_stats)
# split df into sensible groups

dfs = [
    df_all_stats.loc[
    (df_all_stats["jobid"] == jid) | (df_all_stats["jobid"] == str(jid))]
    for jid in job_ids
]
new_dfs = []

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
            "avg_steps"
        ]

agg_key_list = [key for key in eval_key_list]

for df in dfs:
    print(f"{df=}")
    if int(df["jobid"].iloc[0]) == 69308:
        continue
    if int(df["jobid"].iloc[0]) == 71571:
        continue

    df = df.sort_values(by="n_ground_states", ascending=True)

    #TODO: aggregate / sum values first
    df["expected_n_err"] = df["p_err"] * df["code_size"] * df["code_size"] * df["stack_depth"]
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

    agg_groups["weighted_avg_steps"] = agg_groups["avg_steps"] / agg_groups["total_n_episodes"]

    agg_groups.columns = agg_groups.columns.droplevel(1)
    
    print(agg_groups)

    agg_groups["valid_success_rate"] = agg_groups["n_valid_ground_states"] / agg_groups["n_valid_episodes"]
    agg_groups["overall_success_rate"] = (agg_groups["n_ground_states"] + agg_groups["n_ep_w_syndromes"]) / agg_groups["total_n_episodes"]

    agg_groups["valid_fail_rate"] = 1.0 - agg_groups["valid_success_rate"]
    agg_groups["overall_fail_rate"] = 1.0 - agg_groups["overall_success_rate"]

    agg_groups["valid_fail_rate_per_cycle"] = agg_groups["valid_fail_rate"] / agg_groups["stack_depth"]
    agg_groups["overall_fail_rate_per_cycle"] = agg_groups["overall_fail_rate"] / agg_groups["stack_depth"]

    agg_groups["validity_rate"] = agg_groups["n_valid_episodes"] / agg_groups["total_n_episodes"]

    agg_groups["valid_avg_lifetime"] = 1.0 / agg_groups["valid_fail_rate_per_cycle"]
    agg_groups["overall_avg_lifetime"] = 1.0 / agg_groups["overall_fail_rate_per_cycle"]

    # agg_groups["fail_rate"] = 1 - agg_groups["weighted_success_rate"]
    # agg_groups["scaled_fail_rate"] = agg_groups["fail_rate"] / agg_groups["stack_depth"]

    new_dfs.append(agg_groups)

df_all = pd.concat(new_dfs)
print(df_all)
# need to filter best performing model for each d as well
max_x = new_dfs[0]["p_err"].max()

error_rates = df_all["p_err"]
key_success_rate = "weighted_success_rate"
title_succes_rate = "Success Rate"
key_scaled_fail_rate = "overall_fail_rate_per_cycle"
title_scaled_fail_rate = "Overall Fail Rate Per Cycle"

key_valid_fail_rate = "valid_fail_rate_per_cycle"
title_valid_fail_rate = "Valid Fail Rate Per Cycle"

key_valid_avg_life = "valid_avg_lifetime"
title_valid_avg_life = "Valid Average Lifetime"

key_overall_avg_life = "overall_avg_lifetime"
title_overall_avg_life = "Overall Average Lifetime"

for o_jid in omit_job_ids:
    job_ids.remove(o_jid)

plot_colors = ["#404E5C", "#F76C5E", "#E9B44C", "#7F95D1", "#CF1259"]
markers = ["o", "v", "^", "X", "d"]
ylim_lin_plot = (-1e-4, 0.008)
ylim_log_plot = (50, 1e5)

def set_text_lin(axis):
    axis.text(
        0.0055,
        0.0052,
        "Single Qubit",
        rotation=49
    )

def set_text_lin_split(axis):
    axis.text(
        0.0053,
        0.0044,
        "Single Qubit",
        rotation=42
    )

def set_text_log(axis):
    axis.text(
        0.0015,
        125,
        "Single Qubit",
        rotation=-15
    )

def set_text_log_split(axis):
    axis.text(
        0.0015,
        100,
        "Single Qubit",
        rotation=-15
    )


################## Plot Overall Fail Rate per Cycle ##################
fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    if jid in omit_job_ids:
        continue

    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_scaled_fail_rate],
        label=f"d={code_size}, h={stack_depth}",
        c=plot_colors[i],
        marker=markers[i]
    )
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
plt.savefig("plots/depth_overall_fail_rate_p_err.pdf")
plt.show()

################## Plot Valid Fail Rate per Cycle ##################
fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_valid_fail_rate],
        label=f"d={code_size}, h={stack_depth}",
        c=plot_colors[i],
        marker=markers[i]
    )
ax.plot(
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k',
    label="One Qubit"
)
ax.set(title=title_valid_fail_rate)
ax.set(xlabel=r"$p_\mathrm{err}$", ylabel=title_valid_fail_rate)


plt.legend()
plt.tight_layout()
plt.savefig("plots/depth_valid_fail_rate_p_err.pdf")
plt.show()

################## Plot Valid Fail Rate per Cycle over Scaled Error Rate ##################
fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err_one_layer"],
        y=new_dfs[i][key_valid_fail_rate],
        label=f"h={stack_depth}",
        c=plot_colors[i],
        marker=markers[i]
    )
ax.plot(
    np.linspace(new_dfs[0]["p_err_one_layer"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    np.linspace(new_dfs[0]["p_err_one_layer"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k'
)
ax.set(title=title_valid_fail_rate)
ax.set(xlabel=r"$p_\mathrm{err}^\mathrm{one layer}$", ylabel=title_valid_fail_rate)


plt.legend()
plt.tight_layout()
plt.savefig("plots/depth_valid_fail_rate_p_err_one_layer.pdf")
plt.show()

################## Plot Valid Average Lifetime ##################
fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_valid_avg_life],
        label=f"d={code_size}, h={stack_depth}",
        c=plot_colors[i],
        marker=markers[i]
    )
ax.plot(
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    1.0 / np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k',
    label="One Qubit"
)
ax.set(title=title_valid_avg_life)
ax.set(xlabel=r"$p_\mathrm{err}$", ylabel=title_valid_avg_life, ylim=(0, 10000))


plt.legend()
plt.tight_layout()
plt.savefig("plots/depth_valid_lifetime_p_err.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_valid_avg_life],
        label=f"h={stack_depth}",
        c=plot_colors[i],
        marker=markers[i]
    )
ax.plot(
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    1.0 / np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k'
)
ax.set(title=title_valid_avg_life)
ax.set(
    xlabel=r"$p_\mathrm{err}$",
    ylabel=title_valid_avg_life,
    xlim=(1e-3, new_dfs[0]["p_err"].max()), yscale="log"
)

plt.legend()
plt.tight_layout()
plt.savefig("plots/depth_valid_lifetime_p_err_log.pdf")
plt.show()

################## Plot Overall Average Lifetime ##################
fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_overall_avg_life],
        label=f"d={code_size}, h={stack_depth}",
        c=plot_colors[i],
        marker=markers[i]
    )
ax.plot(
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    1.0 / np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k',
    label="One Qubit"
)
ax.set(title=title_overall_avg_life)
ax.set(xlabel=r"$p_\mathrm{err}$", ylabel=title_overall_avg_life, ylim=(0, 10000))


plt.legend()
plt.tight_layout()
plt.savefig("plots/depth_overall_lifetime_p_err.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, sharex=True)

for i, jid in enumerate(job_ids):
    code_size = new_dfs[i]["code_size"].iloc[0]
    stack_depth = new_dfs[i]["stack_depth"].iloc[0]
    # print(new_dfs[i])
    ax.scatter(
        x=new_dfs[i]["p_err"],
        y=new_dfs[i][key_overall_avg_life],
        label=f"h={stack_depth}",
        c=plot_colors[i],
        marker=markers[i]
    )
ax.plot(
    np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    1.0 / np.linspace(new_dfs[0]["p_err"].min(), new_dfs[0]["p_err"].max(), 100, endpoint=True),
    'k'
)
ax.set(title=title_overall_avg_life)
ax.set(
    xlabel=r"$p_\mathrm{err}$",
    ylabel=title_overall_avg_life,
    xlim=(1e-3, new_dfs[0]["p_err"].max()), yscale="log"
)

plt.legend()
plt.tight_layout()
plt.savefig("plots/depth_overall_lifetime_p_err_log.pdf")
plt.show()

sys.exit()


exit()


error_rates = df_all["p_err"]
key_success_rate = "weighted_success_rate"
title_succes_rate = "Success Rate"
key_scaled_fail_rate = "scaled_fail_rate"
title_scaled_fail_rate = "Scaled Fail Rate"
# TODO: what to plot?
# idea 1) fix p_err
fig, ax = plt.subplots(1, 1, sharex=True)

n_err_min = [0.809, 1.21]
n_err_max = [0.9, 1.27]

for i in range(len(n_err_min)):
    plot_df = df_all.loc[
        (df_all["expected_n_err"] <= n_err_max[i]) &
        (df_all["expected_n_err"] >= n_err_min[i])
        ]

    ax.scatter(
        x=plot_df["stack_depth"],
        y=plot_df[key_success_rate],
        label=f"{n_err_min[i]:.2f}" + r"$ < \overline{n_\mathrm{err}} < $" + f"{n_err_max[i]:.2f}",
        s=50
    )

ax.set(title=title_succes_rate)

ax.set(xlabel="Stack Depth", ylim=(0.8, 1.0))

plt.legend()
plt.tight_layout()
plt.savefig("plots/stack_depth_expected_n_err.pdf")
plt.show()

# idea 2) fix p_err_one_layer
# idea 3) plot all different stack depths over p_err
# idea 5) plot all different stack depths over p_err_one_layer

# fig, ax = plt.subplots(len(agg_key_list), 1, sharex=True)
fig, ax = plt.subplots(1, 1, sharex=True)
stack_depths = [3, 7, 11]
# for ax_idx, key in enumerate(agg_key_list):
ax_idx = 0
for h_idx, stack_depth in enumerate(stack_depths):
    df_plot = df_all.loc[
        df_all["stack_depth"] == stack_depth
    ]

    ax.scatter(
        x=df_plot["p_err"],
        y=df_plot[key_success_rate],
        label=f"h={str(stack_depth)}"
    )
ax.set(title=title_succes_rate)
ax.set(xlabel=r"$p_\mathrm{err}$")

plt.legend()
plt.tight_layout()
plt.savefig("plots/stack_depth_p_err.pdf")
plt.show()

# idea 4) plot different error rates over stack depth
# fig, ax = plt.subplots(len(agg_key_list), 1, sharex=True)
fig, ax = plt.subplots(1, 1, sharex=True)
tolerance = 1e-4
plot_error_rates = [0.0041, 0.0071, 0.0101]
# for ax_idx, key in enumerate(agg_key_list):
for p_idx, p_err in enumerate(plot_error_rates):
    df_plot = df_all.loc[
        (p_err - tolerance <= df_all["p_err"]) &
        (df_all["p_err"] <= p_err + tolerance)
    ]

    ax.scatter(
        x=df_plot["stack_depth"],
        y=df_plot[key_success_rate],
        label=r"$p_\mathrm{err}$=" + f"{p_err:.4f}"
    )
ax.set(title=title_succes_rate)
ax.set(xlabel="Stack Depth")

plt.legend()
plt.tight_layout()
plt.savefig("plots/stack_depth_compare_p_err.pdf")
plt.show()

# idea 5) plot all different stack depths over p_err_one_layer
fig, ax = plt.subplots(1, 1, sharex=True)
stack_depths = [3, 7, 11]

for h_idx, stack_depth in enumerate(stack_depths):
    df_plot = df_all.loc[
        df_all["stack_depth"] == stack_depth
    ]

    ax.scatter(
        x=df_plot["p_err_one_layer"],
        y=df_plot[key_success_rate],
        label=f"h={str(stack_depth)}"
    )
ax.set(title=title_succes_rate)
ax.set(xlabel=r"$p_\mathrm{err}^\mathrm{one \; layer}$")

plt.legend()
plt.tight_layout()
plt.savefig("plots/stack_depth_p_err_one_layer.pdf")
plt.show()
