"""
Init
"""

from .iniparser import Config, recursively_call_functions_in_dictionary
from .exceptions import (
    InvalidArgumentsList,
    KeyCombinationNotExisting,
    KeyCombinationAmbigious,
    ALL_EXCEPTIONS,
)
