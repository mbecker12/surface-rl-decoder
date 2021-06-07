import os
import re
import argparse

PARAMETER_MAPPING = {
    "CONFIG_ENV_SIZE": "d",
    "CONFIG_ENV_STACK_DEPTH": "h",
    "CONFIG_ENV_P_ERROR": "p",
    "CONFIG_ENV_P_MSMT": "P",
    "d": "CONFIG_ENV_SIZE",
    "h": "CONFIG_ENV_STACK_DEPTH",
    "p": "CONFIG_ENV_P_ERROR",
    "P": "CONFIG_ENV_P_MSMT",
}
DEFAULT_CODE_SIZE = ["3", "5", "7", "9"]
DEFAULT_STACK_DEPTH = ["3", "5", "7", "9"]
DEFAULT_P_ERROR = ["0.005", "0.01", "0.02", "0.03", "0.04", "0.05"]
DEFAULT_P_MSMT = ["0.005", "0.01", "0.02", "0.03", "0.04", "0.05"]

DEFAULT_VALUES = {
    "d": DEFAULT_CODE_SIZE,
    "h": DEFAULT_STACK_DEPTH,
    "p": DEFAULT_P_ERROR,
    "P": DEFAULT_P_MSMT,
}
MATCH_NUMERICAL = r"=\d\.?\d{0,}"

parser = argparse.ArgumentParser(
    description="Generate child scripts from one base env config file."
)
parser.add_argument(
    "-b",
    "--base-script",
    metavar="b",
    type=str,
    help="give the path to the file to use as the base config file",
)
parser.add_argument(
    "-p",
    "--parameters",
    metavar="p",
    type=str,
    default="dhpP",
    help="Give a list of parameters that should be changed in child files.",
)
parser.add_argument(
    "-B",
    "--base-path",
    metavar="B",
    type=str,
    default="threshold_test",
    help="The base path to write the new config files to.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    print(f"{args.base_script=}")
    print(f"{args.parameters=}")

    base_path = os.getcwd()
    base_path = os.path.join(base_path, args.base_path)

    base_file = args.base_script
    change_parameters = args.parameters

    with open(base_file, "r") as env_file:
        env_config = env_file.readlines()

    print()

    def find_lines_to_change(config_file, search_term):
        line_idx = None
        for i, line in enumerate(config_file):
            if search_term in line:
                line_idx = i
        assert line_idx is not None, f"{search_term=}"
        return line_idx

    # line_to_change_h, line_to_change_d, line_to_change_p, line_to_change_P = find_lines_to_change(env_config)

    for d_value in DEFAULT_VALUES["d"]:
        d_path_name = os.path.join(base_path, f"d{d_value}")
        h_value = d_value
        os.makedirs(d_path_name, exist_ok=True)

        if int(d_value) == 3:
            time_value = "4"
        elif int(d_value) == 5:
            time_value = "6"
        elif int(d_value) == 7:
            time_value = "9"
        elif int(d_value) == 9:
            time_value = "12"

        for p_value in DEFAULT_VALUES["p"]:
            p_val = int(float(p_value) * 1000)
            p_path_name = os.path.join(base_path, f"d{d_value}", f"p{p_val:04d}")
            # os.makedirs(os.path.join(base_path, f"h{h_value}", f"d{d_value}", f"{p_val:3d}"), exist_ok=True)

            print(f"{d_value=}, {p_value=}")
            P_value = p_value

            line_to_change_h = find_lines_to_change(env_config, PARAMETER_MAPPING["h"])
            if line_to_change_h is not None:
                old_line_h = env_config.pop(line_to_change_h)
                new_value_h = "=" + h_value
                new_line_h = re.sub(MATCH_NUMERICAL, new_value_h, old_line_h)
                env_config.append(new_line_h)

            # line_to_change_h, line_to_change_d, line_to_change_p, line_to_change_P = find_lines_to_change(env_config)
            line_to_change_p = find_lines_to_change(env_config, PARAMETER_MAPPING["p"])
            if line_to_change_p is not None:
                old_line_p = env_config.pop(line_to_change_p)
                new_value_p = "=" + p_value
                new_line_p = re.sub(MATCH_NUMERICAL, new_value_p, old_line_p)
                env_config.append(new_line_p)

                # line_to_change_h, line_to_change_d, line_to_change_p, line_to_change_P = find_lines_to_change(env_config)
            line_to_change_d = find_lines_to_change(env_config, PARAMETER_MAPPING["d"])
            if line_to_change_d is not None:
                old_line_d = env_config.pop(line_to_change_d)
                new_value_d = "=" + d_value
                new_line_d = re.sub(MATCH_NUMERICAL, new_value_d, old_line_d)
                env_config.append(new_line_d)

                # line_to_change_h, line_to_change_d, line_to_change_p, line_to_change_P = find_lines_to_change(env_config)
            line_to_change_P = find_lines_to_change(env_config, PARAMETER_MAPPING["P"])
            if line_to_change_P is not None:
                old_line_P = env_config.pop(line_to_change_P)
                new_value_P = "=" + P_value
                new_line_P = re.sub(MATCH_NUMERICAL, new_value_P, old_line_P)
                env_config.append(new_line_P)

                # line_to_change_h, line_to_change_d, line_to_change_p, line_to_change_P = find_lines_to_change(env_config)

            line_to_change_time = find_lines_to_change(
                env_config, "CONFIG_LEARNER_MAX_TIME_H"
            )
            if line_to_change_time is not None:
                old_line_time = env_config.pop(line_to_change_time)
                new_value_time = "=" + time_value
                new_line_time = re.sub(MATCH_NUMERICAL, new_value_time, old_line_time)
                env_config.append(new_line_time)

            with open(f"{p_path_name}.env", "w") as new_file:
                new_file.writelines(env_config)
