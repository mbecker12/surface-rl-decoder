# TODO: iterate over parameters d, p
# and queue a new run for that parameter set
# via sbatch
import subprocess

DEFAULT_CODE_SIZE = ["3", "5", "7", "9"]
DEFAULT_STACK_DEPTH = ["3", "5", "7", "9"]
DEFAULT_P_ERROR = ["0.005", "0.01", "0.02", "0.03", "0.04", "0.05"]
DEFAULT_P_MSMT = ["0.005", "0.01", "0.02", "0.03", "0.04", "0.05"]

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
tensorboard_log_dir = "threshold_runs"
network_save_dir = "networks"
base_path_config_file = "surface-rl-decoder/threshold_conf"
base_description = "threshold / gtrxl / "


for d_value in DEFAULT_VALUES["d"]:
    d_config_file = base_path_config_file + f"/d{d_value}"
    d_description = base_description + f"d={d_value} / "
    for p_value in DEFAULT_VALUES["p"]:
        p_val = int(float(p_value) * 1000)
        p_config_file = d_config_file + f"/p{p_val:04d}.env"
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
        

        command = f"sbatch {script_name} -w {workdir} -i {singularity_image} -C {p_config_file} -t {tensorboard_log_dir} -n {network_save_dir} -d '{p_description}'"
        print(f"sbatch_command: {command}")
        # TODO: find a way around splitting the -d argument
        process = subprocess.run(
            command.split(),
            stdout=subprocess.PIPE
        )
