"""Exception cases"""


class InvalidArgumentsList(Exception):
    """
    Exception for invalid args passed
    """


class KeyCombinationNotExisting(Exception):
    """
    Exception to describe when a key or a key combination is not existing in the given config files
    """


class KeyCombinationAmbigious(Exception):
    """
    Exception to describe when a key or a key combination is not existing in the given config files
    """


ALL_EXCEPTIONS = [
    InvalidArgumentsList,
    KeyCombinationNotExisting,
    KeyCombinationAmbigious,
]
