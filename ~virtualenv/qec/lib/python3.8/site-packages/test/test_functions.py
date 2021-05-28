"""
Unittests for config parser
"""
import unittest
import os

from parameterized import parameterized

from iniparser import (
    recursively_call_functions_in_dictionary,
    InvalidArgumentsList,
    KeyCombinationNotExisting,
    KeyCombinationAmbigious,
)
from iniparser.__init__ import Config
from iniparser.helper import gen_dict_extract

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# pylint: disable=unused-argument
def custom_name_func(testcase_func, param_num, param):
    """
    Naming function for test cases
    """
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name(
            "__".join(
                "List_{}".format("_".join([str(j) for j in x]))
                if isinstance(x, list)
                else str(x)
                for x in param.args
            )
        ),
    )


class TestBasicImportedFunctions(unittest.TestCase):
    """
    Tests some Basic functionality
    """

    def test_recursive_function_call(self):
        """
        Tests recursive dictionary function calls.
        """
        dict_ = dict(test=dict(level="second", func=lambda: "abc"))
        dict_ = recursively_call_functions_in_dictionary(dict_)
        self.assertEqual(dict_["test"]["func"], "abc")

    def test_recursive_search(self):
        """
        This unittest tests the recursive search in a dictionary with 3 simple cases
        """
        dict_ = dict(
            test=dict(level="second", func="abc", second_level={"level": "third"})
        )
        self.assertEqual(list(gen_dict_extract("func", dict_)), ["abc"])
        self.assertEqual(list(gen_dict_extract("level", dict_)), ["second", "third"])
        self.assertEqual(list(gen_dict_extract("not_found", dict_)), [])
        self.assertEqual(
            list(
                gen_dict_extract(
                    "excluded",
                    [
                        {
                            "new_test": "new_value",
                            "excluded": "not available",
                            "test_key": "test_value",
                        }
                    ],
                )
            ),
            ["not available"],
        )

    def test_duplicate_field(self):
        """
        This test case tests the sanity with duplicate keys
        """
        config_ = Config(mode="all_allowed")
        config_.scan(os.path.join(FILE_DIR, "some_folder"), True)
        config_.read()
        self.assertEqual(
            list(gen_dict_extract("test_key", config_.config_rendered)),
            ["test_value", "test_value"],
        )


class TestBasicConfigDict(unittest.TestCase):
    """
    Test class for simple function tests
    """

    @parameterized.expand(
        [
            # All allowed
            ["all_allowed", None, None, True],
            ["all_allowed", None, "include", True],
            ["all_allowed", None, "exclude", False],
            ["all_allowed", None, "hint", False],
            ["all_allowed", "include", None, True],
            ["all_allowed", "include", "include", True],
            ["all_allowed", "include", "exclude", False],
            ["all_allowed", "include", "hint", False],
            ["all_allowed", "exclude", None, False],
            ["all_allowed", "exclude", "exclude", False],
            ["all_allowed", "exclude", "include", True],
            ["all_allowed", "exclude", "hint", False],
            ["all_allowed", "hint", "include", True],
            ["all_allowed", "hint", "hint", False],
            ["all_allowed", "hint", "exclude", False],
            ["all_allowed", "hint", "exclude", False],
            # All forbidden
            ["all_forbidden", None, None, False],
            ["all_forbidden", None, "include", True],
            ["all_forbidden", None, "exclude", False],
            ["all_forbidden", None, "hint", True],
            ["all_forbidden", "include", None, True],
            [
                "all_forbidden",
                "include",
                "include",
                True,
            ],
            [
                "all_forbidden",
                "include",
                "exclude",
                False,
            ],
            ["all_forbidden", "include", "hint", True],
            ["all_forbidden", "exclude", None, False],
            [
                "all_forbidden",
                "exclude",
                "exclude",
                False,
            ],
            [
                "all_forbidden",
                "exclude",
                "include",
                True,
            ],
            ["all_forbidden", "exclude", "hint", True],
            ["all_forbidden", "hint", "include", True],
            ["all_forbidden", "hint", "hint", True],
            ["all_forbidden", "hint", "exclude", False],
            ["all_forbidden", "hint", None, True],
        ],
        name_func=custom_name_func,
    )
    def test_inclusion(self, mode, section, line, expect):
        """
        Tests in a parameterised way the inclusion/exclusion cases
        """
        config_ = Config(mode=mode)
        # pylint: disable=protected-access
        self.assertEqual(expect, config_._include_or_exclude(section, line))


class TestGetterOfConfig(unittest.TestCase):
    """
    Test cases for the getter method
    """

    @parameterized.expand(
        [
            [["test_section", "excluded"], None, "not available", None],
            [["test_section", "new_test"], None, "new_value", None],
            [["test_key"], None, None, KeyCombinationAmbigious],
            [["foo"], None, None, KeyCombinationAmbigious],
            [["some_folder_simple", "foo"], KeyCombinationNotExisting, "bar", None],
            [["some_folder_simple", "test_key"], None, None, KeyCombinationAmbigious],
            [["test"], KeyCombinationNotExisting, None, KeyCombinationNotExisting],
            [["test", 1], None, None, InvalidArgumentsList],
            [["test", ["hallo", "zwei"]], None, None, InvalidArgumentsList],
            [["test", "hallo"], None, None, None],
            [[["test", 2]], None, None, InvalidArgumentsList],
            [[1], None, None, InvalidArgumentsList],
            [[], None, None, InvalidArgumentsList],
            [[{"a": "b"}], None, None, InvalidArgumentsList],
        ],
        name_func=custom_name_func,
    )
    def test_standard_get(self, args, default, expected_result, expected_raise):
        """
        Test the getter method for Exceptions and other inputs
        :param args: arguments passed
        :param default: Default return value
        :param expected_raise: What is expected to be raised
        """
        config_ = Config(mode="all_allowed")
        config_.scan(os.path.join(FILE_DIR, "some_folder"), True)
        config_.read()
        if expected_raise is not None:
            with self.assertRaises(expected_raise):
                result = config_.get(*args, default=default)
        else:
            result = config_.get(*args, default=default)
            self.assertEqual(expected_result, result)


class TestConfigReading(unittest.TestCase):
    """
    Test the reading of test configs
    """

    @parameterized.expand(
        [
            ["all_allowed", "another", "not_default", "biz", "baz"],
            ["all_allowed", "another", "not_default", "foo", "bar"],
        ],
        name_func=custom_name_func,
    )
    def test_config_reading(self, mode, file, section, key, expected):
        """
        Test the env exposing reading in config files
        """
        config_ = Config(mode=mode)
        config_.scan(os.path.join(FILE_DIR, "some_folder"), True)
        config_.read()
        returned = config_.config[file][section][key]
        if callable(returned):
            self.assertEqual(returned(), expected)
        else:
            self.assertEqual(returned, expected)

    @parameterized.expand(
        [
            ["all_allowed", "another", "not_default", "biz", "str"],
            ["all_allowed", "another", "not_default", "foo", "func"],
            ["all_allowed", "some_folder_simple", "test_section", "test_key", "func"],
            ["all_allowed", "some_folder_simple", "test_section", "new_test", "str"],
        ],
        name_func=custom_name_func,
    )
    def test_exposed_unexposed(self, mode, file, section, key, expected):
        """
        Test the env exposing reading in config files
        """
        config_ = Config(mode=mode)
        config_.scan(os.path.join(FILE_DIR, "some_folder"), True)
        config_.read()
        returned = config_.config[file][section][key]
        if expected == "func":
            self.assertEqual(callable(returned), True)
        elif expected == "str":
            self.assertIsInstance(returned, str)
        else:
            self.fail("Unrecognized expected value")

    @parameterized.expand(
        [
            [
                "all_allowed",
                "some_folder_simple",
                "default",
                "test_key",
                "test_value",
                "test123",
                True,
            ],
            [
                "all_allowed",
                "some_folder_simple",
                "test_section",
                "test_key",
                "test_value",
                "test123",
                True,
            ],
            [
                "all_allowed",
                "some_folder_simple",
                "test_section",
                "new_test",
                "new_value",
                "test123",
                False,
            ],
            [
                "all_forbidden",
                "some_folder_simple",
                "default",
                "test_key",
                "test_value",
                "test123",
                False,
            ],
        ],
        name_func=custom_name_func,
    )
    def test_env_changes(
        self, mode, file, section, key, expected, env_change, successful_env_change
    ):
        """
        Test the effect of changing a environment variable
        """
        config_ = Config(mode=mode)
        config_.scan(os.path.join(FILE_DIR, "some_folder"), True)
        config_.read()
        envkey = f"{file}_{section}_{key}".upper()
        self.assertEqual(
            envkey
            in [
                item
                for sublist in config_.environment_variables.values()
                for item in sublist
            ],
            successful_env_change,
        )
        os.environ[envkey] = env_change
        returned = config_.config[file][section][key]
        if callable(returned):
            returned = returned()
        del os.environ[envkey]
        self.assertEqual(returned == env_change, successful_env_change)

    @parameterized.expand(
        [
            [
                "all_allowed",
                "some_folder_simple",
                "default",
                "test_key",
                True,
            ],
            [
                "all_allowed",
                "some_folder_simple",
                "test_section",
                "test_key",
                True,
            ],
            [
                "all_allowed",
                "some_folder_simple",
                "test_section",
                "new_test",
                False,
            ],
            [
                "all_forbidden",
                "some_folder_simple",
                "default",
                "test_key",
                False,
            ],
        ],
        name_func=custom_name_func,
    )
    def test_integrity_of_config_files(self, mode, file, section, key, callable_):
        """
        Test the integrity of the config file when rendering static config file
        """
        config_ = Config(mode=mode)
        config_.scan(os.path.join(FILE_DIR, "some_folder"), True)
        config_.read()
        old = dict(config_.config)
        rendered = config_.config_rendered
        new = dict(config_.config)
        returned = config_.config[file][section][key]
        rendered = rendered[file][section][key]
        self.assertEqual(old, new)
        self.assertEqual(callable(returned), callable_)
        self.assertFalse(callable(rendered))


if __name__ == "__main__":
    unittest.main()
