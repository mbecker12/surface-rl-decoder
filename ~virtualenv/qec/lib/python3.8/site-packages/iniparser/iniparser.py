"""
This module provides the Config class. It's object can be used to parse your ini files and expose
it's variables to the environment, which comes in handy if using a docker runtime or k8s.
"""
import collections
import os
from os import path as pth

from typing import List, Union

from iniparser.exceptions import (
    InvalidArgumentsList,
    KeyCombinationNotExisting,
    KeyCombinationAmbigious,
)

from .helper import (
    config_dict,
    deep_merge,
    recursively_call_functions_in_dictionary,
    gen_dict_extract,
)


class Config:
    """
    Config parser Class to persist configuration values.
    """

    def __init__(self, mode="all_allowed"):
        """
        Initializes the Config Object
        :param mode: string all_allowed or all_forbidden. In all_allowed mode, all variables
        are exposed to env by default. In all_forbidden, all variables are unexposed to env
        by default.
        """
        assert mode in [
            # pylint: disable=no-self-use
            "all_allowed",
            "all_forbidden",
        ], "mode has to be either all_allowed or all_forbidden"
        self.config_files_ = []
        self._config = dict()
        self.mode = mode
        self.environment_variables = dict()

    @property
    def map_file_names_to_unique_keys(self):
        """
        Walks the files paths backwards until a unique name is constructed
        :return: dict: Mapping from filename to unique key
        """
        name_parts = {c: c.strip(".ini").split("/") for c in self.config_files_}
        # print(name_parts)
        no_duplicates = dict()
        for i in range(max([len(p) for p in name_parts.values()])):
            current_keys = {
                np[0]: "_".join(np[1][-(i + 1) :]) for np in name_parts.items()
            }
            # print(current_keys)
            counts = dict(collections.Counter(current_keys.values()).items())
            for key, val in [
                (key, val) for key, val in current_keys.items() if counts[val] <= 1
            ]:
                no_duplicates[key] = val
                del name_parts[key]
            # Break early if no more separation required
            if len(name_parts.keys()) == 0:
                break
        return no_duplicates

    @property
    def config(self):
        """
        Returns the config as dictionary, where exposed variables are lambdas and unexposed
        variables are strings.
        :return:
        """
        assert self._config != dict(), "Please make sure to scan your files first"
        mapping = self.map_file_names_to_unique_keys
        merged = dict()
        for file, uniquekey in mapping.items():
            # print(uniquekey)
            # print(self._config.keys())
            merged[uniquekey] = deep_merge(
                self._config[file]["exposed"], self._config[file]["unexposed"]
            )
        return merged

    @property
    def config_rendered(self):
        """
        Returns the always updated configuration dictionary, filled with the default values from
        the INI file(s) or the environment variables respectively.
        :return: dictionary
        """
        conf = dict(self.config)
        return recursively_call_functions_in_dictionary(conf)

    def get(self, *args, default=KeyCombinationNotExisting):
        """
        Getter, which makes accessing a unique key easier in a mess of multiple ini files
        :param args: list of string keys that are specifying the desired config entry
        :param default: either a default return Key, or if not set the
            KeyCombinationNotExisting Error
        :return: value of config file which fits to the keys
        """
        if len(args) == 0:
            raise InvalidArgumentsList("No arguments provided")
        all_strings = all([isinstance(arg, str) for arg in args])
        one_list_arg = len(args) == 1 and isinstance(args[0], list)
        if not any([all_strings, one_list_arg]):
            raise InvalidArgumentsList(
                "The passed list of arguments {} is not accepted. "
                "Please provide string keys as arguments or one list argument"
                " that contains all string keys.".format(args)
            )
        if one_list_arg:
            all_strings = all([isinstance(arg, str) for arg in args[0]])
        if not all_strings:
            raise InvalidArgumentsList(
                "All keys provdided as argument have to be of type string. "
                "However, {} was received".format(args)
            )
        current = self.config_rendered
        # print("rendered", current)
        for key in args:
            current = list(gen_dict_extract(key, current))
        #     print(key, current)
        # print("end with", current)
        if len(current) > 1:
            raise KeyCombinationAmbigious(
                "The key combination {} cannot be found".format(args)
            )
        if len(current) == 0:
            if default is KeyCombinationNotExisting:
                raise default("No combination for {} could be found".format(args))
            return default
        return current[0]

    def to_env(self, output_path=None):
        """
        Writes the exposed variables to the output_path
        :param output_path:
        :return: env.md string
        """
        mark_down = str("")
        mark_down += "# ENV\nThis file keeps track of the exposed Variables.\n"
        for key in self.environment_variables:
            mark_down += f"## {key}\n```\n"
            mark_down += "\n".join(self.environment_variables[key])
            mark_down += "\n```\n\n"
        if output_path is not None:
            with open(output_path, "w") as output_file:
                output_file.write(mark_down)
        return mark_down

    def scan(self, entry_points: Union[str, List[str]], recursive: bool = True):
        """
        Scans the root directory/ies for *.ini files.
        :param entry_points: str or list of strings with paths
        :param recursive: boolean to indicate whether to search recursively
        :return: void
        """
        entry_points = (
            [entry_points] if isinstance(entry_points, str) else list(entry_points)
        )
        inis = set()
        for root in entry_points:
            root = pth.abspath(root)
            if recursive:
                _ = [
                    inis.add(pth.join(dp, f))
                    for dp, dn, fn in os.walk(root)
                    for f in fn
                    if f.endswith(".ini")
                ]
            else:
                _ = [
                    inis.add(pth.join(root, f))
                    for f in os.listdir(root)
                    if f.endswith(".ini")
                ]
        self.config_files_ = [pth.relpath(ini) for ini in sorted(inis)]
        assert (
            len(self.config_files_) > 0
        ), "No config files were found. Please check your entrypoint"
        return self

    @property
    def config_files(self):
        """
        Returns the absolute paths of the registered config files
        :return: list
        """
        return [pth.abspath(pth.join(".", file)) for file in self.config_files_]

    # It's a very complex run, but this is the counterpart ot catching all possible combinations.
    # pylint: disable=too-many-return-statements,too-many-branches
    def _include_or_exclude(self, section_hint=None, line_hint=None):
        """
        Decides for the available hints, whether to expose a variable or not
        :param section_hint: None|include|exclude|hint
        :param line_hint: None|include|exclude|hint
        :return: bool
        """
        if section_hint is None and line_hint is None:
            return self.mode == "all_allowed"
        if section_hint is None:
            if line_hint == "include":
                return True
            if line_hint == "exclude":
                return False
            if line_hint == "hint":
                return self.mode == "all_forbidden"
        elif section_hint == "include" or (
            section_hint == "hint" and self.mode == "all_forbidden"
        ):
            if line_hint is None or line_hint == "include":
                return True
            if line_hint == "exclude":
                return False
            if line_hint == "hint":
                return self.mode == "all_forbidden"
        elif section_hint == "exclude" or (
            section_hint == "hint" and self.mode == "all_allowed"
        ):
            if line_hint is None or line_hint == "exclude":
                return False
            if line_hint == "include":
                return True
            if line_hint == "hint":
                return self.mode == "all_forbidden"
        return True

    # pylint: disable=too-many-locals
    def read(self):
        """
        Reads the INI files
        :return: void
        """
        for file, filekey in zip(self.config_files, self.config_files_):
            content = str()
            with open(file, "r") as opened_file:
                content = opened_file.readlines()

            sections = [i for i, c in enumerate(content) if c.startswith("[")]

            hints = {
                i: c.split("ceb:")[-1].strip()
                for i, c in enumerate(content)
                if c.startswith(";") and "ceb:" in c
            }
            for line_key, hint_value in hints.items():
                assert hint_value in ["include", "exclude", "hint"], (
                    f"{file} has an unrecognised hint in line {line_key}."
                    f' "{hint_value}" is not recognized.'
                )

            hinted_sections = [sec for sec in sections if sec - 1 in hints.keys()]
            sections = [sec for sec in sections if (sec - 1) not in hints.keys()]

            exposed_config = str("")
            unexposed_config = str("")

            last_section_hint = None
            for i, line in enumerate(content):
                # print('Line', i, line)
                section_line = False
                if i in sections + hinted_sections:
                    # print('is section line')
                    exposed_config += line
                    unexposed_config += line
                    section_line = True
                    last_section_hint = hints.get(i - 1, None)

                if line.strip() == "\n":
                    continue
                if not section_line and (not line.startswith(";") or "ceb" not in line):
                    # print('is key_value', hints.get(i-1, None))
                    # print(i, line)
                    if self._include_or_exclude(
                        last_section_hint, hints.get(i - 1, None)
                    ):
                        # print('which is exposed')
                        exposed_config += line
                    else:
                        # print('which is not exposed')
                        # print(i, hints.get(i-1, None))
                        unexposed_config += line

            mpping = self.map_file_names_to_unique_keys
            unexposed_config = config_dict(
                file=mpping[filekey], content=unexposed_config, exposed=False
            )
            exposed_config = config_dict(
                file=mpping[filekey], content=exposed_config, exposed=True
            )
            self._config[filekey] = dict(
                exposed=exposed_config[1], unexposed=unexposed_config[1]
            )
            self.environment_variables[filekey] = exposed_config[2]
        return self

    def __str__(self):
        """String representation of this object"""
        return f"Memory: {id(self)}\tFiles: {str(self.config_files)}"
