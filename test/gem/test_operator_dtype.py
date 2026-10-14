import operator

import numpy
import pytest

import gem
from gem.interpreter import evaluate


def add_left(node, number):
    return number + node


def multiply_left(node, number):
    return number * node


OPERATIONS = [operator.add, add_left, operator.mul, multiply_left]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_integer_operand_keeps_the_dtype_of_an_integer_node(operation):
    """An integer added to or multiplied with an integer node stays an integer."""
    node = gem.Variable("n", (), dtype=gem.uint_type)
    assert operation(node, 3).dtype == gem.uint_type


@pytest.mark.parametrize("operation", OPERATIONS)
def test_integer_operand_of_a_float_node_stays_float(operation):
    """An integer with a float node converts to float, as before."""
    node = gem.Variable("x", (), dtype=numpy.float64)
    assert operation(node, 3).dtype == numpy.float64


@pytest.mark.parametrize("operation", OPERATIONS)
def test_integer_operand_of_an_untyped_node_is_a_float_literal(operation):
    """An integer with an untyped node converts to a float literal, as before."""
    node = gem.Variable("x", ())
    result = operation(node, 3)
    assert result.dtype is None
    literal, = (child for child in result.children if isinstance(child, gem.Literal))
    assert literal.dtype == numpy.float64


@pytest.mark.parametrize("operation", OPERATIONS)
def test_float_operand_of_an_integer_node_is_float(operation):
    """A non-integer number with an integer node gives a float result."""
    node = gem.Variable("n", (), dtype=gem.uint_type)
    assert operation(node, 1.5).dtype == numpy.float64


def test_integer_arithmetic_on_an_integer_node_evaluates_exactly():
    """Index arithmetic built with "+" and "*" gives unsigned integers."""
    i = gem.Index(extent=3)
    node = gem.Indexed(gem.Literal(numpy.array([0, 1, 2]), dtype=gem.uint_type), (i,))
    result, = evaluate([2 + 3 * node])
    assert result.arr.dtype.kind == "u"
    assert numpy.array_equal(result.arr, numpy.array([2, 5, 8]))
