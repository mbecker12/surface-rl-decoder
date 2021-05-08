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
    "P": "CONFIG_ENV_P_MSMT"
}
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
MATCH_NUMERICAL = r"=\d\.?\d{0,}"

parser = argparse.ArgumentParser(description='Generate child scripts from one base env config file.')
parser.add_argument('-b', '--base-script', metavar='b', type=str,
                    help='give the path to the file to use as the base config file')
parser.add_argument('-p', '--parameters', metavar='p', type=str, default="dhp",
                    help="Give a list of parameters that should be changed in child files.")
parser.add_argument('-B', '--base-path', metavar='B', type=str, default="stack_depth_test",
                    help="Give a list of parameters that should be changed in child files.")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    print(f"{args.base_script}")
    print(f"{args.parameters}")
    base_path = os.getcwd()
    base_path = os.path.join(base_path, args.base_path)
    base_file = args.base_script
    change_parameters = args.parameters

    with open(base_file, "r") as env_file:
        env_config = env_file.readlines()

    print()

    line_to_change_h = []
    line_to_change_d = []
    line_to_change_p = []

    def find_lines_to_change(config_file):
        h_line = None
        d_line = None
        p_line = None
        for i, line in enumerate(config_file):
            if PARAMETER_MAPPING["h"] in line:
                h_line = i
            if PARAMETER_MAPPING["d"] in line:
                d_line = i
            if PARAMETER_MAPPING["p"] in line:
                p_line = i
        return h_line, d_line, p_line

    line_to_change_h, line_to_change_d, line_to_change_p = find_lines_to_change(env_config)
                    
    for h_value in DEFAULT_VALUES["h"]:
        h_path_name = os.path.join(base_path, f"h{h_value}")

        os.makedirs(os.path.join(base_path, f"h{h_value}"), exist_ok=True)
        for d_value in DEFAULT_VALUES["d"]:
            d_path_name = os.path.join(base_path, f"h{h_value}", f"d{d_value}")
            os.makedirs(os.path.join(base_path, f"h{h_value}", f"d{d_value}"), exist_ok=True)
            for p_value in DEFAULT_VALUES["p"]:
                expected_n_errors = float(p_value) * int(d_value) * int(d_value)
                expected_n_errors_str = f"{expected_n_errors:.1f}"

                p_one_layer = expected_n_errors / (int(h_value) * int(d_value) * int(d_value))
                p_one_layer = f"{p_one_layer:.3f}"

                # print(f"{p_value=}, {expected_n_errors_str=}, {p_one_layer=}, {d_value=}, {h_value=}")
                # continue
                p_val = int(float(p_value) * 100)
                p_path_name = os.path.join(base_path, f"h{h_value}", f"d{d_value}", f"p{p_val:03d}.env")
                # os.makedirs(os.path.join(base_path, f"h{h_value}", f"d{d_value}", f"{p_val:3d}"), exist_ok=True)
                # print(f"{h_value=}, {d_value=}, {p_value=}")
                # print(f"{p_path_name=}")

                if line_to_change_h is not None:
                    old_line_h = env_config.pop(line_to_change_h)
                    new_value_h = "=" + h_value
                    new_line_h = re.sub(MATCH_NUMERICAL, new_value_h, old_line_h)
                    env_config.append(new_line_h)

                line_to_change_h, line_to_change_d, line_to_change_p = find_lines_to_change(env_config)

                if line_to_change_p is not None:
                    old_line_p = env_config.pop(line_to_change_p)
                    # new_value_p = "=" + p_value
                    new_value_p = "=" + str(p_one_layer)
                    new_line_p = re.sub(MATCH_NUMERICAL, new_value_p, old_line_p)
                    env_config.append(new_line_p)

                    line_to_change_h, line_to_change_d, line_to_change_p = find_lines_to_change(env_config)

                if line_to_change_d is not None:
                    old_line_d = env_config.pop(line_to_change_d)
                    new_value_d = "=" + d_value
                    new_line_d = re.sub(MATCH_NUMERICAL, new_value_d, old_line_d)
                    env_config.append(new_line_d)

                    if 3 <= int(d_value)  <= 5:
                        # TODO: adjust NN architecture / number of layers
                        # something like:
                        # CONFIG_LEARNER_MODEL_CONFIG_FILE=conv_agents_gtrxl.json
                        pass
                    if int(d_value) == 7:
                        # TODO: adjust NN architecture / number of layers
                        # CONFIG_LEARNER_MODEL_CONFIG_FILE=conv_agents_gtrxl_medium.json
                        pass
                    if int(d_value) == 9:
                        # TODO: adjust NN architecture / number of layers
                        # CONFIG_LEARNER_MODEL_CONFIG_FILE=conv_agents_gtrxl_deep.json
                        pass

                    line_to_change_h, line_to_change_d, line_to_change_p = find_lines_to_change(env_config)

                with open(f"{p_path_name}", "w") as new_file:
                    new_file.writelines(env_config)
