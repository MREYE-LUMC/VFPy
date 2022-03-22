import wand.exceptions
from warnings import warn  # noqa

# noinspection PyUnresolvedReferences
DelegateError = wand.exceptions.DelegateError  # noqa


class ValueWarning(Warning):
    """A Warning indicating that a supplied value might be incorrect."""

