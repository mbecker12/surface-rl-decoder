"""
Dumps a config file of the type readable by ConfigParser
into a dictionary
"""
import configparser
from os.path import basename
from os import getenv as ge


# noinspection PyBroadException
def _env_builder(file, section, key_, prefix=None):
    """
    Generates a  ENV Name to the current config variable
    :param file: string
    :param section: string
    :param key_: string
    :return: string
    """
    return (
        f"{prefix.upper() if isinstance(prefix, str) else ''}"
        + f"{basename(file).strip('.ini').upper()}_{section.upper()}_{key_.upper()}"
    )


def log(file, section, option, alternative):
    """Logs the Created ENV Var"""
    print(
        _env_builder(file, section, option),
        "=",
        ge(_env_builder(file, section, option), alternative),
    )


def config_dict(file, content, exposed=True):
    """
    Reads a INI file and parses it into a dictionary
    :param file: string of the filename
    ;:param content: ini content as string
    :return: configparser opject, parsed dictionary
    """
    config = configparser.ConfigParser()
    config.read_string(content)

    sections_dict = dict()

    env_map = []

    # get sections and iterate over each
    sections = config.sections()

    for section in sections:
        options = config.options(section)
        temp_dict = dict()
        for option in options:
            # print(section, option, _env_builder(file, section, option))
            if exposed:
                temp_dict[option] = lambda section_=str(section), option_=str(
                    option
                ): ge(
                    _env_builder(file, section_, option_), config.get(section_, option_)
                )
                env_map.append(_env_builder(file, section, option))
            else:
                temp_dict[option] = config.get(section, option)
            # log(file, section, option, config.get(section, option))

        sections_dict[section] = dict(temp_dict)
        # print(id(temp_dict))
        del temp_dict

    return config, sections_dict, env_map


def deep_merge(source, destination):
    """
    Deep merge of two dictionaries.
    Based of / Copied from
    https://stackoverflow.com/questions/20656135/python-deep-merge-dictionary-data/20666342

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> deep_merge(b, a) == { 'first' : { 'all_rows' :
                                { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }

    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value

    return destination


def recursively_call_functions_in_dictionary(dictionary):
    """
    Walks through the dictionary and executes all callable functions.
    :param dictionary:
    :return: executed dictionary
    """
    dictionary = dict(dictionary)
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = dict(recursively_call_functions_in_dictionary(value))
        else:
            dictionary[key] = value() if callable(value) else value
    return dictionary


def gen_dict_extract(key, dict_):
    """
    Recursively search for key in the dictionary and return all found values as generator
    :param key: string key
    :param dict_: dict to search through
    """
    if hasattr(dict_, "items"):
        for key_, value_ in dict_.items():
            if key_ == key:
                yield value_
            if isinstance(value_, dict):
                for result in gen_dict_extract(key, value_):
                    yield result
            elif isinstance(value_, list):
                for dictionary_ in value_:
                    for result in gen_dict_extract(key, dictionary_):
                        yield result
    elif isinstance(dict_, list):
        for item in dict_:
            for result in gen_dict_extract(key, item):
                yield result
