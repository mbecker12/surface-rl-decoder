import os
import subprocess
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from distributed.model_util import choose_model, load_model
from distributed.learner_util import (
    safe_append_in_dict,
    transform_list_dict,
)
from evaluation.batch_evaluation import RESULT_KEY_COUNTS, RESULT_KEY_ENERGY, RESULT_KEY_EPISODE, RESULT_KEY_Q_VALUE_STATS, RESULT_KEY_RATES, batch_evaluation

code_size = 5
stack_depth = 5

# job ID, code size, stack depth, p_err_train
# job_info = [
#     (68051, 5, 3, 0.1),
#     (68052, 5, 5, 0.1),
#     (68062, 5, 7, 0.1),
#     (68054, 5, 9, 0.1),
#     (68055, 5, 11, 0.1)
# ]

job_info = [
    (68553, 7, 1, 0.1),
    (68554, 7, 3, 0.1),
    (68555, 7, 5, 0.1),
    (68556, 7, 7, 0.1),
    (68557, 7, 9, 0.1),
    (68558, 7, 11, 0.1),
    # (68559, 7, 13, 0.1),

]

CLUSTER_NETWORK_PATH = "networks"
LOCAL_NETWORK_PATH = "stack_depth_networks"
full_local_network_path = os.path.join("$HOME/Projects/surface-rl-decoder", f"{LOCAL_NETWORK_PATH}")
os.makedirs(full_local_network_path, exist_ok=True)


do_copy = False
if do_copy:
    print("Copy Data from Cluster")
    for (jid, code_size, stack_depth, p_err) in job_info:
        print(f"\tCopying {jid}...")
        target_path = f"{LOCAL_NETWORK_PATH}/{code_size}"
        print(f"{target_path=}")
        os.makedirs(target_path, exist_ok=True)
        command = f"scp -r alvis://cephyr/users/gunter/Alvis/surface-rl-decoder/{CLUSTER_NETWORK_PATH}/{code_size}/{jid} {target_path}"
        process = subprocess.run(
                command.split(),
                stdout=subprocess.PIPE
            )
# exit()
# model_name = "conv2d"
model_name = "simple_conv"

eval_device = torch.device("cpu")


# TODO: run evaluation by loading model from full_local_network_path
all_results = []
all_results_counter = 0
n_episodes = 256

df = pd.DataFrame(columns=[
    "jobid", "code_size", "stack_depth", "p_err_train", "p_err", "logical_err", "remaining_syndromes", "q_value",
    "steps", "final_energy", "energy_diff",
    "median_logical_err", "median_remaining_syndromes",
    "median_steps", "median_final_energy", "median_energy_diff"
    ])
print("Proceed to Evaluation")
for (jid, code_size, stack_depth, p_err) in job_info:
    os.environ["CONFIG_ENV_SIZE"] = str(code_size)
    os.environ["CONFIG_ENV_STACK_DEPTH"] = str(stack_depth)
    target_path = f"{LOCAL_NETWORK_PATH}/{code_size}/{jid}"
    model_config_path = target_path + f"/{model_name}_{code_size}_meta.yaml"
    old_model_path = target_path + f"/{model_name}_{code_size}_{jid}.pt"
    with open(model_config_path, "r") as yaml_file:
        general_config = yaml.load(yaml_file)
        # conf_model_name = general_config["learner"]["model_name"]
        # print(f"{conf_model_name=}")
        model_config = general_config["network"]
        model_config["device"] = "cpu"

    # print(yaml.dump(general_config, default_flow_style=False))
    # os.environ["CONFIG_LEARNER_LOAD_MODEL_PATH"] = target_path
    # os.environ["CONFIG_LEARNER_LOAD_MODEL"] = "1"
    os.environ["CONFIG_ENV_SIZE"] = str(code_size)
    os.environ["CONFIG_ENV_STACK_DEPTH"] = str(stack_depth)
    
    print("Load Model")
    model = choose_model(model_name, model_config)
    model, _, _ = load_model(model, old_model_path, model_device="cpu")

    final_result_dict = {
        RESULT_KEY_EPISODE: {},
        RESULT_KEY_Q_VALUE_STATS: {},
        RESULT_KEY_ENERGY: {},
        RESULT_KEY_COUNTS: {},
        RESULT_KEY_RATES: {},
    }
    # p_error_list = np.linspace(0.01, 0.2, num=10, endpoint=True)
    p_error_list = [0.05, 0.1, 0.2]
    results_list = [None] * len(p_error_list)
    print("Run Evaluation for Different Error Rates")
    for i_err_list, p_error in enumerate(p_error_list):
        print(f"\t{p_error=}")
        expected_n_errors = float(p_error) * int(code_size) * int(code_size)
        p_one_layer = expected_n_errors / (int(stack_depth) * int(code_size) * int(code_size))
        eval_results, all_q_values = batch_evaluation(
            model,
            "",
            eval_device,
            num_of_random_episodes=n_episodes,
            num_of_user_episodes=0,
            max_num_of_steps=25,
            discount_intermediate_reward=0.3,
            verbosity=2,
            p_msmt=0.0,
            p_err=p_error,
            post_run=False,
            code_size=code_size,
            stack_depth=stack_depth
        )

        # print(yaml.dump(eval_results, default_flow_style=False))
        logical_err = eval_results[RESULT_KEY_EPISODE]["logical_errors_per_episode"]
        remaining_syndromes = eval_results[RESULT_KEY_EPISODE]["remaining_syndromes_per_episode"]
        mean_q_value = eval_results[RESULT_KEY_Q_VALUE_STATS]["mean_q_value"]
        energy_difference = eval_results[RESULT_KEY_COUNTS]["energy_difference"]
        final_energy = eval_results[RESULT_KEY_COUNTS]["energy_final"]
        number_of_steps = eval_results[RESULT_KEY_ENERGY]["number_of_steps"]
        median_logical_err = eval_results[RESULT_KEY_EPISODE]["median logical_errors_per_episode"]
        median_remaining_syndromes = eval_results[RESULT_KEY_EPISODE]["median remaining_syndromes_per_episode"]
        median_energy_difference = eval_results[RESULT_KEY_COUNTS]["median energy_difference"]
        median_final_energy = eval_results[RESULT_KEY_COUNTS]["median energy_final"]
        median_number_of_steps = eval_results[RESULT_KEY_ENERGY]["median number_of_steps"]

        df = df.append(
            {
                "jobid": jid,
                "code_size": code_size,
                "stack_depth": stack_depth,
                "p_err_train": p_err,
                "p_err": p_error,
                "logical_err": logical_err,
                "remaining_syndromes": remaining_syndromes,
                "q_value": mean_q_value,
                "final_energy": final_energy,
                "steps": number_of_steps,
                "energy_diff": energy_difference,
                "median_logical_err": median_logical_err,
                "median_remaining_syndromes": median_remaining_syndromes,
                "median_final_energy": median_final_energy,
                "median_steps": median_number_of_steps,
                "median_energy_diff": median_energy_difference
            },
            ignore_index=True
        )
        print(df)
    # all_results[all_results_counter] = ((jid, code_size, stack_depth, p_err), results_list)
    # all_results_counter += 1

print("Conclude Evauation")
print("Plotting")

print(df)

plt.title("Logical Error Rate")
df0 = df.loc[df["p_err"] <= 0.06]
df1 = df.loc[(0.06 < df["p_err"]) & (df["p_err"] < 0.15)]
df2 = df.loc[(0.15 < df["p_err"]) & (df["p_err"] < 0.25)]

plt.errorbar(
    x=df0["stack_depth"],
    y=df0["logical_err"],
    yerr=np.sqrt(df0["logical_err"] * (1 - df0["logical_err"]) / n_episodes),
    label="logical_err; p=0.05",
    c='blue',
    mew=2)
plt.errorbar(
    x=df1["stack_depth"],
    y=df1["logical_err"],
    yerr=np.sqrt(df1["logical_err"] * (1 - df1["logical_err"]) / n_episodes),
    label="logical_err; p=0.1",
    c='blue',
    mew=2)
plt.errorbar(
    x=df2["stack_depth"],
    y=df2["logical_err"],
    yerr=np.sqrt(df2["logical_err"] * (1 - df2["logical_err"]) / n_episodes),
    label="logical_err; p=0.2",
    c='orange',
    mew=2)
plt.legend()
plt.show()


plt.scatter(df0["stack_depth"], df0["steps"], label="steps; p=0.1")
plt.scatter(df1["stack_depth"], df1["steps"], label="steps; p=0.2")
plt.scatter(df0["stack_depth"], df0["final_energy"], label="final_energy; p=0.1")
plt.scatter(df1["stack_depth"], df1["final_energy"], label="final_energy; p=0.2")
plt.scatter(df0["stack_depth"], df0["energy_diff"], label="energy_diff; p=0.1")
plt.scatter(df1["stack_depth"], df1["energy_diff"], label="energy_diff; p=0.2")
plt.legend()
plt.show()


plt.scatter(df0["stack_depth"], df0["remaining_syndromes"], label="remaining syndromes; p=0.1")
plt.scatter(df1["stack_depth"], df1["remaining_syndromes"], label="remaining syndromes; p=0.2")
# plt.plot(df["stack_depth"], df["q_value"], label="mean_q_value")
plt.legend()
plt.show()


