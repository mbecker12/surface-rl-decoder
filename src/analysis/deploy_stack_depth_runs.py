# TODO: iterate over parameters d, h, p
# and queue a new run for that parameter set
# via sbatch
import argparse
import subprocess

DEFAULT_STACK_DEPTH = ["1", "3", "5", "7", "9", "11", "13"]
DEFAULT_CODE_SIZE = ["7"]
DEFAULT_P_ERROR = ["0.1"]
DEFAULT_P_MSMT = ["0.1"]

DEFAULT_VALUES = {
    "d": DEFAULT_CODE_SIZE,
    "h": DEFAULT_STACK_DEPTH,
    "p": DEFAULT_P_ERROR,
    "P": DEFAULT_P_MSMT
}

script_name = "surface-rl-decoder/mp-script.sh"
# script_name = "mp-script.sh"
workdir = "surface-rl-decoder"
singularity_image = "qec-mp_dql-new.sif"
tensorboard_log_dir = "stack_depth_runs"
network_save_dir = "networks"
base_path_config_file = "surface-rl-decoder/stack_depth_conf"
base_description = "stack depth / gtrxl / "

parser = argparse.ArgumentParser(description='Deploy runs for stack depth analysis using different child env config files.')
parser.add_argument('-b', '--base-script', metavar='b', type=str,
                    help='give the path to the file to use as the base config file')
parser.add_argument('-p', '--parameters', metavar='p', type=str, default="dhp",
                    help="Give a list of parameters that should be changed in child files.")
parser.add_argument('-B', '--base-path', metavar='B', type=str, default="stack_depth_test",
                    help="The base path to write the new config files to.")

for h_value in DEFAULT_VALUES["h"]:
    h_config_file = base_path_config_file + f"/h{h_value}"
    h_description = base_description + f"h={h_value} / "
    for d_value in DEFAULT_VALUES["d"]:
        d_config_file = h_config_file + f"/d{d_value}"
        d_description = h_description + f"d={d_value} / "
        for p_value in DEFAULT_VALUES["p"]:
            p_val = int(float(p_value) * 100)
            p_config_file = d_config_file + f"/p{p_val:03d}.env"
            p_description = d_description + f"p={p_value}"

            command_list = [
                    "sbatch",
                    script_name,
                    "-w",
                    workdir,
                    "-i",
                    singularity_image,
                    "-C",
                    p_config_file,
                    "-t",
                    tensorboard_log_dir,
                    "-n",
                    network_save_dir,
                    "-d",
                    p_description
                ]
            sbatch_command = str(command_list)
            print(f"sbatch_command: {sbatch_command}")

            command = f"sbatch {script_name} -w {workdir} -i {singularity_image} -C {p_config_file} -t {tensorboard_log_dir} -n {network_save_dir} -d '{p_description}'"

            process = subprocess.run(
                command.split(),
                stdout=subprocess.PIPE
            )
