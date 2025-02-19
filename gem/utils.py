import collections
import functools
import numbers
from functools import cached_property  # noqa: F401
from typing import Any

import numpy as np


def groupby(iterable, key=None):
    """Groups objects by their keys.

    :arg iterable: an iterable
    :arg key: key function

    :returns: list of (group key, list of group members) pairs
    """
    if key is None:
        key = lambda x: x
    groups = collections.OrderedDict()
    for elem in iterable:
        groups.setdefault(key(elem), []).append(elem)
    return groups.items()


def make_proxy_class(name, cls):
    """Constructs a proxy class for a given class.

    :arg name: name of the new proxy class
    :arg cls: the wrapee class to create a proxy for
    """
    def __init__(self, wrapee):
        self._wrapee = wrapee

    def make_proxy_property(name):
        def getter(self):
            return getattr(self._wrapee, name)
        return property(getter)

    dct = {'__init__': __init__}
    for attr in dir(cls):
        if not attr.startswith('_'):
            dct[attr] = make_proxy_property(attr)
    return type(name, (), dct)


# Implementation of dynamically scoped variables in Python.
class UnsetVariableError(LookupError):
    pass


_unset = object()


class DynamicallyScoped(object):
    """A dynamically scoped variable."""

    def __init__(self, default_value=_unset):
        if default_value is _unset:
            self._head = None
        else:
            self._head = (default_value, None)

    def let(self, value):
        return _LetBlock(self, value)

    @property
    def value(self):
        if self._head is None:
            raise UnsetVariableError("Dynamically scoped variable not set.")
        result, tail = self._head
        return result


class _LetBlock(object):
    """Context manager representing a dynamic scope."""

    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        self.state = None

    def __enter__(self):
        assert self.state is None
        value = self.value
        tail = self.variable._head
        scope = (value, tail)
        self.variable._head = scope
        self.state = scope

    def __exit__(self, exc_type, exc_value, traceback):
        variable = self.variable
        assert self.state is variable._head
        value, variable._head = variable._head
        self.state = None


@functools.singledispatch
def safe_repr(obj: Any) -> str:
    """Return a 'safe' repr for an object, accounting for floating point error.

    Parameters
    ----------
    obj :
        The object to produce a repr for.

    Returns
    -------
    str :
        A repr for the object.

    """
    raise TypeError(f"Cannot provide a safe repr for {type(obj).__name__}")


@safe_repr.register(str)
def _(text: str) -> str:
    return text


@safe_repr.register(numbers.Integral)
def _(num: numbers.Integral) -> str:
    return repr(num)


@safe_repr.register(numbers.Real)
def _(num: numbers.Real) -> str:
    # set roundoff to close-to-but-not-exactly machine epsilon
    precision = np.finfo(num).precision - 2
    return "{:.{prec}}".format(num, prec=precision)


@safe_repr.register(np.ndarray)
def _(array: np.ndarray) -> str:
    return f"{type(array).__name__}([{', '.join(map(safe_repr, array))}])"


@safe_repr.register(list)
def _(list_: list) -> str:
    return f"[{', '.join(map(safe_repr, list_))}]"


@safe_repr.register(tuple)
def _(tuple_: tuple) -> str:
    return f"({', '.join(map(safe_repr, tuple_))})"
