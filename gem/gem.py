"""GEM is the intermediate language of TSFC for describing
tensor-valued mathematical expressions and tensor operations.
It is similar to Einstein's notation.

Its design was heavily inspired by UFL, with some major differences:
 - GEM has got nothing FEM-specific.
 - In UFL free indices are just unrolled shape, thus UFL is very
   restrictive about operations on expressions with different sets of
   free indices. GEM is much more relaxed about free indices.

Similarly to UFL, all GEM nodes have 'shape' and 'free_indices'
attributes / properties. Unlike UFL, however, index extents live on
the Index objects in GEM, not on all the nodes that have those free
indices.
"""

from abc import ABCMeta
from itertools import chain
from functools import reduce
from operator import attrgetter
from numbers import Integral, Number

import numpy
from numpy import asarray

from gem.node import Node as NodeBase, traversal

from FIAT.orientation_utils import Orientation as FIATOrientation


__all__ = ['Node', 'Identity', 'Literal', 'Zero', 'Failure',
           'Variable', 'Sum', 'Product', 'Division', 'FloorDiv', 'Remainder', 'Power',
           'MathFunction', 'MinValue', 'MaxValue', 'Comparison',
           'LogicalNot', 'LogicalAnd', 'LogicalOr', 'Conditional',
           'Index', 'VariableIndex', 'Indexed', 'ComponentTensor',
           'IndexSum', 'ListTensor', 'Concatenate', 'Delta', 'OrientationVariableIndex',
           'index_sum', 'partial_indexed', 'reshape', 'view',
           'indices', 'as_gem', 'FlexiblyIndexed',
           'Inverse', 'Solve', 'extract_type', 'uint_type']


uint_type = numpy.dtype(numpy.uintc)


class NodeMeta(type):
    """Metaclass of GEM nodes.

    When a GEM node is constructed, this metaclass automatically
    collects its free indices if 'free_indices' has not been set yet.
    """

    def __call__(self, *args, **kwargs):
        # Create and initialise object
        obj = super(NodeMeta, self).__call__(*args, **kwargs)

        # Set free_indices if not set already
        if not hasattr(obj, 'free_indices'):
            obj.free_indices = unique(chain(*[c.free_indices
                                              for c in obj.children]))
        # Set dtype if not set already.
        if not hasattr(obj, 'dtype'):
            obj.dtype = obj.inherit_dtype_from_children(obj.children)

        return obj


class Node(NodeBase, metaclass=NodeMeta):
    """Abstract GEM node class."""

    __slots__ = ('free_indices', 'dtype')

    def is_equal(self, other):
        """Common subexpression eliminating equality predicate.

        When two (sub)expressions are equal, the children of one
        object are reassigned to the children of the other, so some
        duplicated subexpressions are eliminated.
        """
        result = NodeBase.is_equal(self, other)
        if result:
            self.children = other.children
        return result

    def __getitem__(self, indices):
        try:
            indices = tuple(indices)
        except TypeError:
            indices = (indices, )
        return Indexed(self, indices)

    def __add__(self, other):
        return componentwise(Sum, self, as_gem(other))

    def __radd__(self, other):
        return as_gem(other).__add__(self)

    def __sub__(self, other):
        return componentwise(
            Sum, self,
            componentwise(Product, Literal(-1), as_gem(other)))

    def __rsub__(self, other):
        return as_gem(other).__sub__(self)

    def __mul__(self, other):
        return componentwise(Product, self, as_gem(other))

    def __rmul__(self, other):
        return as_gem(other).__mul__(self)

    def __matmul__(self, other):
        other = as_gem(other)
        if not self.shape and not other.shape:
            return Product(self, other)
        elif not (self.shape and other.shape):
            raise ValueError("Both objects must have shape for matmul")
        elif self.shape[-1] != other.shape[0]:
            raise ValueError(f"Mismatching shapes {self.shape} and {other.shape} in matmul")
        *i, k = indices(len(self.shape))
        _, *j = indices(len(other.shape))
        expr = Product(Indexed(self, tuple(i) + (k, )),
                       Indexed(other, (k, ) + tuple(j)))
        return ComponentTensor(IndexSum(expr, (k, )), tuple(i) + tuple(j))

    def __rmatmul__(self, other):
        return as_gem(other).__matmul__(self)

    @property
    def T(self):
        i = indices(len(self.shape))
        return ComponentTensor(Indexed(self, i), tuple(reversed(i)))

    def __truediv__(self, other):
        other = as_gem(other)
        if other.shape:
            raise ValueError("Denominator must be scalar")
        return componentwise(Division, self, other)

    def __rtruediv__(self, other):
        return as_gem(other).__truediv__(self)

    def __floordiv__(self, other):
        other = as_gem_uint(other)
        if other.shape:
            raise ValueError("Denominator must be scalar")
        return componentwise(FloorDiv, self, other)

    def __rfloordiv__(self, other):
        return as_gem_uint(other).__floordiv__(self)

    def __mod__(self, other):
        other = as_gem_uint(other)
        if other.shape:
            raise ValueError("Denominator must be scalar")
        return componentwise(Remainder, self, other)

    def __rmod__(self, other):
        return as_gem_uint(other).__mod__(self)

    @staticmethod
    def inherit_dtype_from_children(children):
        if any(c.dtype is None for c in children):
            # Set dtype = None will let _assign_dtype()
            # assign the default dtype for this node later.
            return
        else:
            return numpy.result_type(*(c.dtype for c in children))


class Terminal(Node):
    """Abstract class for terminal GEM nodes."""

    __slots__ = ('_dtype',)

    children = ()

    is_equal = NodeBase.is_equal

    @property
    def dtype(self):
        """Data type of the node.

        We only need to set dtype (or _dtype) on terminal nodes, and
        other nodes inherit dtype from their children.

        Currently dtype is significant only for nodes under index DAGs
        (DAGs underneath `VariableIndex`s representing indices), and
        `VariableIndex` checks if the dtype of the node that it wraps is
        of uint_type. _assign_dtype() will then assign uint_type to those nodes.

        dtype can be `None` otherwise, and _assign_dtype() will assign
        the default dtype to those nodes.

        """
        if hasattr(self, '_dtype'):
            return self._dtype
        else:
            raise AttributeError(f"Must set _dtype on terminal node, {type(self)}")


class Scalar(Node):
    """Abstract class for scalar-valued GEM nodes."""

    __slots__ = ()

    shape = ()


class Failure(Terminal):
    """Abstract class for failure GEM nodes."""

    __slots__ = ('shape', 'exception')
    __front__ = ('shape', 'exception')

    def __init__(self, shape, exception):
        self.shape = shape
        self.exception = exception
        self._dtype = None


class Constant(Terminal):
    """Abstract base class for constant types.

    Convention:
     - array: numpy array of values
     - value: float or complex value (scalars only)
    """
    pass


class Zero(Constant):
    """Symbolic zero tensor"""

    __slots__ = ('shape',)
    __front__ = ('shape',)
    __back__ = ('dtype',)

    def __init__(self, shape=(), dtype=None):
        self.shape = shape
        self._dtype = dtype

    @property
    def value(self):
        assert not self.shape
        return numpy.array(0, dtype=self.dtype or float).item()


class Identity(Constant):
    """Identity matrix"""

    __slots__ = ('dim',)
    __front__ = ('dim',)
    __back__ = ('dtype',)

    def __init__(self, dim, dtype=None):
        self.dim = dim
        self._dtype = dtype

    @property
    def shape(self):
        return (self.dim, self.dim)

    @property
    def array(self):
        return numpy.eye(self.dim, dtype=self.dtype)


class Literal(Constant):
    """Tensor-valued constant"""

    __slots__ = ('array',)
    __front__ = ('array',)
    __back__ = ('dtype',)

    def __new__(cls, array, dtype=None):
        array = asarray(array)
        return super(Literal, cls).__new__(cls)

    def __init__(self, array, dtype=None):
        array = asarray(array)
        if dtype is None:
            # Assume float or complex.
            try:
                self.array = array.astype(float, casting="safe")
            except TypeError:
                self.array = array.astype(complex)
        else:
            # Can be int, etc.
            self.array = array.astype(dtype)
        self._dtype = self.array.dtype

    def is_equal(self, other):
        if type(self) is not type(other):
            return False
        if self.shape != other.shape:
            return False
        return tuple(self.array.flat) == tuple(other.array.flat)

    def get_hash(self):
        return hash((type(self), self.shape, tuple(self.array.flat)))

    @property
    def value(self):
        assert self.shape == ()
        return self.array.dtype.type(self.array)

    @property
    def shape(self):
        return self.array.shape


class Variable(Terminal):
    """Symbolic variable tensor"""

    __slots__ = ('name', 'shape')
    __front__ = ('name', 'shape')
    __back__ = ('dtype',)

    def __init__(self, name, shape, dtype=None):
        self.name = name
        self.shape = shape
        self._dtype = dtype


class Sum(Scalar):
    __slots__ = ('children',)

    def __new__(cls, *args):
        try:
            a, b = args
        except ValueError:
            # Handle more than two arguments
            return reduce(Sum, args)
        assert not a.shape
        assert not b.shape

        # Constant folding
        if isinstance(a, Zero):
            return b
        elif isinstance(b, Zero):
            return a

        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value + b.value, dtype=Node.inherit_dtype_from_children([a, b]))

        self = super(Sum, cls).__new__(cls)
        self.children = a, b
        return self


class Product(Scalar):
    __slots__ = ('children',)

    def __new__(cls, *args):
        try:
            a, b = args
        except ValueError:
            # Handle more than two arguments
            return reduce(Product, args)
        assert not a.shape
        assert not b.shape

        # Constant folding
        if isinstance(a, Zero) or isinstance(b, Zero):
            return Zero()

        if a == one:
            return b
        if b == one:
            return a

        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value * b.value, dtype=Node.inherit_dtype_from_children([a, b]))

        self = super(Product, cls).__new__(cls)
        self.children = a, b
        return self


class Division(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Constant folding
        if isinstance(b, Zero):
            raise ValueError("division by zero")
        if isinstance(a, Zero):
            return Zero()

        if b == one:
            return a

        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value / b.value, dtype=Node.inherit_dtype_from_children([a, b]))

        self = super(Division, cls).__new__(cls)
        self.children = a, b
        return self


class FloorDiv(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape
        dtype = Node.inherit_dtype_from_children([a, b])
        if dtype != uint_type:
            raise ValueError(f"dtype ({dtype}) != unit_type ({uint_type})")
        # Constant folding
        if isinstance(b, Zero):
            raise ValueError("division by zero")
        if isinstance(a, Zero):
            return Zero(dtype=dtype)
        if isinstance(b, Constant) and b.value == 1:
            return a
        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value // b.value, dtype=dtype)
        self = super(FloorDiv, cls).__new__(cls)
        self.children = a, b
        return self


class Remainder(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape
        dtype = Node.inherit_dtype_from_children([a, b])
        if dtype != uint_type:
            raise ValueError(f"dtype ({dtype}) != uint_type ({uint_type})")
        # Constant folding
        if isinstance(b, Zero):
            raise ValueError("division by zero")
        if isinstance(a, Zero):
            return Zero(dtype=dtype)
        if isinstance(b, Constant) and b.value == 1:
            return Zero(dtype=dtype)
        if isinstance(a, Constant) and isinstance(b, Constant):
            return Literal(a.value % b.value, dtype=dtype)
        self = super(Remainder, cls).__new__(cls)
        self.children = a, b
        return self


class Power(Scalar):
    __slots__ = ('children',)

    def __new__(cls, base, exponent):
        assert not base.shape
        assert not exponent.shape
        dtype = Node.inherit_dtype_from_children([base, exponent])

        # Constant folding
        if isinstance(base, Zero):
            if isinstance(exponent, Zero):
                raise ValueError("cannot solve 0^0")
            return Zero(dtype=dtype)
        elif isinstance(exponent, Zero):
            return Literal(1, dtype=dtype)
        elif isinstance(base, Constant) and isinstance(exponent, Constant):
            return Literal(base.value ** exponent.value, dtype=dtype)

        self = super(Power, cls).__new__(cls)
        self.children = base, exponent
        return self


class MathFunction(Scalar):
    __slots__ = ('name', 'children')
    __front__ = ('name',)

    def __new__(cls, name, *args):
        assert isinstance(name, str)
        assert all(arg.shape == () for arg in args)

        if name in {'conj', 'real', 'imag'}:
            arg, = args
            if isinstance(arg, Zero):
                return arg

        self = super(MathFunction, cls).__new__(cls)
        self.name = name
        self.children = args
        return self


class MinValue(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class MaxValue(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class Comparison(Scalar):
    __slots__ = ('operator', 'children')
    __front__ = ('operator',)

    def __init__(self, op, a, b):
        assert not a.shape
        assert not b.shape

        if op not in [">", ">=", "==", "!=", "<", "<="]:
            raise ValueError("invalid operator")

        self.operator = op
        self.children = a, b
        self.dtype = None  # Do not inherit dtype from children.


class LogicalNot(Scalar):
    __slots__ = ('children',)

    def __init__(self, expression):
        assert not expression.shape

        self.children = expression,


class LogicalAnd(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class LogicalOr(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class Conditional(Node):
    __slots__ = ('children', 'shape')

    def __new__(cls, condition, then, else_):
        assert not condition.shape
        assert then.shape == else_.shape == ()

        # If both branches are the same, just return one of them.  In
        # particular, this will help constant-fold zeros.
        if then == else_:
            return then

        self = super(Conditional, cls).__new__(cls)
        self.children = condition, then, else_
        self.shape = then.shape
        self.dtype = Node.inherit_dtype_from_children([then, else_])
        return self


class IndexBase(metaclass=ABCMeta):
    """Abstract base class for indices."""
    pass


IndexBase.register(int)


class Index(IndexBase):
    """Free index"""

    # Not true object count, just for naming purposes
    _count = 0

    __slots__ = ('name', 'extent', 'count')

    def __init__(self, name=None, extent=None):
        self.name = name
        Index._count += 1
        self.count = Index._count
        self.extent = extent

    def set_extent(self, value):
        # Set extent, check for consistency
        if self.extent is None:
            self.extent = value
        elif self.extent != value:
            raise ValueError("Inconsistent index extents!")

    def __str__(self):
        if self.name is None:
            return "i_%d" % self.count
        return self.name

    def __repr__(self):
        if self.name is None:
            return "Index(%r)" % self.count
        return "Index(%r)" % self.name

    def __lt__(self, other):
        # Allow sorting of free indices in Python 3
        return id(self) < id(other)

    def __getstate__(self):
        return self.name, self.extent, self.count

    def __setstate__(self, state):
        self.name, self.extent, self.count = state


class VariableIndex(IndexBase):
    """An index that is constant during a single execution of the
    kernel, but whose value is not known at compile time."""

    __slots__ = ('expression',)

    def __init__(self, expression):
        assert isinstance(expression, Node)
        assert not expression.shape
        if expression.dtype != uint_type:
            raise ValueError(f"expression.dtype ({expression.dtype}) != uint_type ({uint_type})")
        self.expression = expression

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return self.expression == other.expression

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self.expression))

    def __str__(self):
        return str(self.expression)

    def __repr__(self):
        return "%r(%r)" % (type(self), self.expression,)

    def __reduce__(self):
        return type(self), (self.expression,)


class Indexed(Scalar):
    __slots__ = ('children', 'multiindex', 'indirect_children')
    __back__ = ('multiindex',)

    def __new__(cls, aggregate, multiindex):
        # Accept numpy or any integer, but cast to int.
        multiindex = tuple(int(i) if isinstance(i, Integral) else i
                           for i in multiindex)

        # Set index extents from shape
        assert len(aggregate.shape) == len(multiindex)
        for index, extent in zip(multiindex, aggregate.shape):
            assert isinstance(index, IndexBase)
            if isinstance(index, Index):
                index.set_extent(extent)
            elif isinstance(index, int) and not (0 <= index < extent):
                raise IndexError("Invalid literal index")

        # Empty multiindex
        if not multiindex:
            return aggregate

        # Zero folding
        if isinstance(aggregate, Zero):
            return Zero(dtype=aggregate.dtype)

        # All indices fixed
        if all(isinstance(i, int) for i in multiindex):
            if isinstance(aggregate, Constant):
                return Literal(aggregate.array[multiindex], dtype=aggregate.dtype)
            elif isinstance(aggregate, ListTensor):
                return aggregate.array[multiindex]

        self = super(Indexed, cls).__new__(cls)
        self.children = (aggregate,)
        self.multiindex = multiindex
        self.indirect_children = tuple(i.expression for i in self.multiindex if isinstance(i, VariableIndex))

        new_indices = []
        for i in multiindex:
            if isinstance(i, Index):
                new_indices.append(i)
            elif isinstance(i, VariableIndex):
                new_indices.extend(i.expression.free_indices)
        self.free_indices = unique(aggregate.free_indices + tuple(new_indices))

        return self

    def index_ordering(self):
        """Running indices in the order of indexing in this node."""
        free_indices = []
        for i in self.multiindex:
            if isinstance(i, Index):
                free_indices.append(i)
            elif isinstance(i, VariableIndex):
                free_indices.extend(i.expression.free_indices)
        return tuple(free_indices)


class FlexiblyIndexed(Scalar):
    """Flexible indexing of :py:class:`Variable`s to implement views and
    reshapes (splitting dimensions only)."""

    __slots__ = ('children', 'dim2idxs', 'indirect_children')
    __back__ = ('dim2idxs',)

    def __init__(self, variable, dim2idxs):
        """Construct a flexibly indexed node.

        Parameters
        ----------
        variable : Node
            `Node` that has a shape.
        dim2idxs : tuple
            Tuple of (offset, ((index, stride), (...), ...)) mapping indices,
            where offset is {Node, int}, index is {Index, VariableIndex, int}, and
            stride is {Node, int}.

        For example, if ``variable`` is rank two, and ``dim2idxs`` is

            ((1, ((i, 12), (j, 4), (k, 1))), (0, ()))

        then this corresponds to the indexing:

            variable[1 + i*12 + j*4 + k][0]

        """
        assert variable.shape
        assert len(variable.shape) == len(dim2idxs)
        dim2idxs_ = []
        free_indices = []
        for dim, (offset, idxs) in zip(variable.shape, dim2idxs):
            offset_ = offset
            idxs_ = []
            last = 0
            if isinstance(offset, Node):
                free_indices.extend(offset.free_indices)
            for index, stride in idxs:
                if isinstance(index, Index):
                    assert index.extent is not None
                    free_indices.append(index)
                    idxs_.append((index, stride))
                    last += (index.extent - 1) * stride
                elif isinstance(index, VariableIndex):
                    base_indices = index.expression.free_indices
                    assert all(base_index.extent is not None for base_index in base_indices)
                    free_indices.extend(base_indices)
                    idxs_.append((index, stride))
                    # last += (unknown_extent - 1) * stride
                elif isinstance(index, int):
                    # TODO: Attach dtype to each Node.
                    # Here, we should simply be able to do:
                    # >>> offset_ += index * stride
                    # but "+" and "*" are not currently correctly overloaded
                    # for indices (integers); they assume floats.
                    if not isinstance(offset, Integral):
                        raise NotImplementedError(f"Found non-Integral offset : {offset}")
                    if isinstance(stride, Constant):
                        offset_ += index * stride.value
                    else:
                        offset_ += index * stride
                else:
                    raise ValueError("Unexpected index type for flexible indexing")
                if isinstance(stride, Node):
                    free_indices.extend(stride.free_indices)
            if dim is not None and isinstance(offset_ + last, Integral) and offset_ + last >= dim:
                raise ValueError("Offset {0} and indices {1} exceed dimension {2}".format(offset, idxs, dim))
            dim2idxs_.append((offset_, tuple(idxs_)))
        self.children = (variable,)
        self.dim2idxs = tuple(dim2idxs_)
        self.free_indices = unique(free_indices)
        indirect_children = []
        for offset, idxs in self.dim2idxs:
            if isinstance(offset, Node):
                indirect_children.append(offset)
            for idx, stride in idxs:
                if isinstance(idx, VariableIndex):
                    indirect_children.append(idx.expression)
                if isinstance(stride, Node):
                    indirect_children.append(stride)
        self.indirect_children = tuple(indirect_children)

    def index_ordering(self):
        """Running indices in the order of indexing in this node."""
        free_indices = []
        for offset, idxs in self.dim2idxs:
            if isinstance(offset, Node):
                free_indices.extend(offset.free_indices)
            for index, stride in idxs:
                if isinstance(index, Index):
                    free_indices.append(index)
                elif isinstance(index, VariableIndex):
                    free_indices.extend(index.expression.free_indices)
                if isinstance(stride, Node):
                    free_indices.extend(stride.free_indices)
        return tuple(free_indices)


class ComponentTensor(Node):
    __slots__ = ('children', 'multiindex', 'shape')
    __back__ = ('multiindex',)

    def __new__(cls, expression, multiindex):
        assert not expression.shape

        # Empty multiindex
        if not multiindex:
            return expression

        # Collect shape
        shape = tuple(index.extent for index in multiindex)
        assert all(s >= 0 for s in shape)

        # Zero folding
        if isinstance(expression, Zero):
            return Zero(shape, dtype=expression.dtype)

        self = super(ComponentTensor, cls).__new__(cls)
        self.children = (expression,)
        self.multiindex = multiindex
        self.shape = shape

        # Collect free indices
        assert set(multiindex) <= set(expression.free_indices)
        self.free_indices = unique(set(expression.free_indices) - set(multiindex))

        return self


class IndexSum(Scalar):
    __slots__ = ('children', 'multiindex')
    __back__ = ('multiindex',)

    def __new__(cls, summand, multiindex):
        # Sum zeros
        assert not summand.shape
        if isinstance(summand, Zero):
            return summand

        # Unroll singleton sums
        unroll = tuple(index for index in multiindex if index.extent <= 1)
        if unroll:
            assert numpy.prod([index.extent for index in unroll]) == 1
            summand = Indexed(ComponentTensor(summand, unroll),
                              (0,) * len(unroll))
            multiindex = tuple(index for index in multiindex
                               if index not in unroll)

        # No indices case
        multiindex = tuple(multiindex)
        if not multiindex:
            return summand

        self = super(IndexSum, cls).__new__(cls)
        self.children = (summand,)
        self.multiindex = multiindex

        # Collect shape and free indices
        assert set(multiindex) <= set(summand.free_indices)
        self.free_indices = unique(set(summand.free_indices) - set(multiindex))

        return self


class ListTensor(Node):
    __slots__ = ('array',)

    def __new__(cls, array):
        array = asarray(array)
        assert numpy.prod(array.shape)
        dtype = Node.inherit_dtype_from_children(tuple(array.flat))

        # Handle children with shape
        child_shape = array.flat[0].shape
        assert all(elem.shape == child_shape for elem in array.flat)

        if child_shape:
            # Destroy structure
            direct_array = numpy.empty(array.shape + child_shape, dtype=object)
            for alpha in numpy.ndindex(array.shape):
                for beta in numpy.ndindex(child_shape):
                    direct_array[alpha + beta] = Indexed(array[alpha], beta)
            array = direct_array

        # Constant folding
        if all(isinstance(elem, Constant) for elem in array.flat):
            return Literal(numpy.vectorize(attrgetter('value'))(array), dtype=dtype)

        self = super(ListTensor, cls).__new__(cls)
        self.array = array
        return self

    @property
    def children(self):
        return tuple(self.array.flat)

    @property
    def shape(self):
        return self.array.shape

    def __reduce__(self):
        return type(self), (self.array,)

    def reconstruct(self, *args):
        return ListTensor(asarray(args).reshape(self.array.shape))

    def __repr__(self):
        return "ListTensor(%r)" % self.array.tolist()

    def is_equal(self, other):
        """Common subexpression eliminating equality predicate."""
        if type(self) is not type(other):
            return False
        if (self.array == other.array).all():
            self.array = other.array
            return True
        return False

    def get_hash(self):
        return hash((type(self), self.shape, self.children))


class Concatenate(Node):
    """Flattens and concatenates GEM expressions by shape.

    Similar to what UFL MixedElement does to value shape.  For
    example, if children have shapes (2, 2), (), and (3,) then the
    concatenated expression has shape (8,).
    """
    __slots__ = ('children',)

    def __new__(cls, *children):
        dtype = Node.inherit_dtype_from_children(children)
        if all(isinstance(child, Zero) for child in children):
            size = int(sum(numpy.prod(child.shape, dtype=int) for child in children))
            return Zero((size,), dtype=dtype)

        self = super(Concatenate, cls).__new__(cls)
        self.children = children
        return self

    @property
    def shape(self):
        return (int(sum(numpy.prod(child.shape, dtype=int) for child in self.children)),)


class Delta(Scalar, Terminal):
    __slots__ = ('i', 'j')
    __front__ = ('i', 'j')
    __back__ = ('dtype',)

    def __new__(cls, i, j, dtype=None):
        if isinstance(i, tuple) and isinstance(j, tuple):
            # Handle multiindices
            return Product(*map(Delta, i, j))
        assert isinstance(i, IndexBase)
        assert isinstance(j, IndexBase)

        # \delta_{i,i} = 1
        if i == j:
            return one

        # Fixed indices
        if isinstance(i, Integral) and isinstance(j, Integral):
            return one if i == j else Zero()

        if isinstance(i, Integral):
            return Indexed(Literal(numpy.eye(j.extent)[i]), (j,))

        if isinstance(j, Integral):
            return Indexed(Literal(numpy.eye(i.extent)[j]), (i,))

        self = super(Delta, cls).__new__(cls)
        self.i = i
        self.j = j
        # Set up free indices
        free_indices = []
        for index in (i, j):
            if isinstance(index, Index):
                free_indices.append(index)
            elif isinstance(index, VariableIndex):
                raise NotImplementedError("Can not make Delta with VariableIndex")
        self.free_indices = tuple(unique(free_indices))
        self._dtype = dtype
        return self


class Inverse(Node):
    """The inverse of a square matrix."""
    __slots__ = ('children', 'shape')

    def __new__(cls, tensor):
        assert len(tensor.shape) == 2
        assert tensor.shape[0] == tensor.shape[1]

        # Invert 1x1 matrix
        if tensor.shape == (1, 1):
            multiindex = (Index(), Index())
            return ComponentTensor(Division(one, Indexed(tensor, multiindex)), multiindex)

        self = super(Inverse, cls).__new__(cls)
        self.children = (tensor,)
        self.shape = tensor.shape

        return self


class Solve(Node):
    """Solution of a square matrix equation with (potentially) multiple right hand sides.

    Represents the X obtained by solving AX = B.
    """
    __slots__ = ('children', 'shape')

    def __init__(self, A, B):
        # Shape requirements
        assert B.shape
        assert len(A.shape) == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]

        self.children = (A, B)
        self.shape = A.shape[1:] + B.shape[1:]


class OrientationVariableIndex(VariableIndex, FIATOrientation):
    """VariableIndex representing a fiat orientation.

    Notes
    -----
    In the current implementation, we need to extract
    `VariableIndex.expression` as index arithmetic
    is not supported (indices are not `Node`).

    """

    def __floordiv__(self, other):
        other = other.expression if isinstance(other, VariableIndex) else as_gem_uint(other)
        return type(self)(FloorDiv(self.expression, other))

    def __rfloordiv__(self, other):
        other = other.expression if isinstance(other, VariableIndex) else as_gem_uint(other)
        return type(self)(FloorDiv(other, self.expression))

    def __mod__(self, other):
        other = other.expression if isinstance(other, VariableIndex) else as_gem_uint(other)
        return type(self)(Remainder(self.expression, other))

    def __rmod__(self, other):
        other = other.expression if isinstance(other, VariableIndex) else as_gem_uint(other)
        return type(self)(Remainder(other, self.expression))


def unique(indices):
    """Sorts free indices and eliminates duplicates.

    :arg indices: iterable of indices
    :returns: sorted tuple of unique free indices
    """
    return tuple(sorted(set(indices), key=id))


def index_sum(expression, indices):
    """Eliminates indices from the free indices of an expression by
    summing over them.  Skips any index that is not a free index of
    the expression."""
    multiindex = tuple(index for index in indices
                       if index in expression.free_indices)
    return IndexSum(expression, multiindex)


def partial_indexed(tensor, indices):
    """Generalised indexing into a tensor by eating shape off the front.
    The number of indices may be less than or equal to the rank of the tensor,
    so the result may have a non-empty shape.

    :arg tensor: tensor-valued GEM expression
    :arg indices: indices, at most as many as the rank of the tensor
    :returns: a potentially tensor-valued expression
    """
    if len(indices) == 0:
        return tensor
    elif len(indices) < len(tensor.shape):
        rank = len(tensor.shape) - len(indices)
        shape_indices = tuple(Index() for i in range(rank))
        return ComponentTensor(
            Indexed(tensor, indices + shape_indices),
            shape_indices)
    elif len(indices) == len(tensor.shape):
        return Indexed(tensor, indices)
    else:
        raise ValueError("More indices than rank!")


def strides_of(shape):
    """Calculate cumulative strides from per-dimension capacities.

    For example:

        [2, 3, 4] ==> [12, 4, 1]

    """
    temp = numpy.flipud(numpy.cumprod(numpy.flipud(list(shape)[1:])))
    return list(temp) + [1]


def decompose_variable_view(expression):
    """Extract information from a shaped node.
       Decompose ComponentTensor + FlexiblyIndexed."""
    if (isinstance(expression, (Variable, Inverse, Solve))):
        variable = expression
        indexes = tuple(Index(extent=extent) for extent in expression.shape)
        dim2idxs = tuple((0, ((index, 1),)) for index in indexes)
    elif (isinstance(expression, ComponentTensor) and
          not isinstance(expression.children[0], FlexiblyIndexed)):
        variable = expression
        indexes = expression.multiindex
        dim2idxs = tuple((0, ((index, 1),)) for index in indexes)
    elif isinstance(expression, ComponentTensor) and isinstance(expression.children[0], FlexiblyIndexed):
        variable = expression.children[0].children[0]
        indexes = expression.multiindex
        dim2idxs = expression.children[0].dim2idxs
    else:
        raise ValueError("Cannot handle {} objects.".format(type(expression).__name__))

    return variable, dim2idxs, indexes


def reshape(expression, *shapes):
    """Reshape a variable (splitting indices only).

    :arg expression: view of a :py:class:`Variable`
    :arg shapes: one shape tuple for each dimension of the variable.
    """
    variable, dim2idxs, indexes = decompose_variable_view(expression)
    assert len(indexes) == len(shapes)
    shape_of = dict(zip(indexes, shapes))

    dim2idxs_ = []
    indices = [[] for _ in range(len(indexes))]
    for offset, idxs in dim2idxs:
        idxs_ = []
        for idx in idxs:
            index, stride = idx
            assert isinstance(index, Index)
            dim = index.extent
            shape = shape_of[index]
            if dim is not None and numpy.prod(shape) != dim:
                raise ValueError("Shape {} does not match extent {}.".format(shape, dim))
            strides = strides_of(shape)
            for extent, stride_ in zip(shape, strides):
                index_ = Index(extent=extent)
                idxs_.append((index_, stride_ * stride))
                indices[indexes.index(index)].append(index_)
        dim2idxs_.append((offset, tuple(idxs_)))

    expr = FlexiblyIndexed(variable, tuple(dim2idxs_))
    return ComponentTensor(expr, tuple(chain.from_iterable(indices)))


def view(expression, *slices):
    """View a part of a shaped object.

    :arg expression: a node that has a shape
    :arg slices: one slice object for each dimension of the expression.
    """
    variable, dim2idxs, indexes = decompose_variable_view(expression)
    assert len(indexes) == len(slices)
    slice_of = dict(zip(indexes, slices))

    dim2idxs_ = []
    indices = [None] * len(slices)
    for offset, idxs in dim2idxs:
        offset_ = offset
        idxs_ = []
        for idx in idxs:
            index, stride = idx
            assert isinstance(index, Index)
            dim = index.extent
            s = slice_of[index]
            start = s.start or 0
            stop = s.stop or dim
            if stop is None:
                raise ValueError("Unknown extent!")
            if dim is not None and stop > dim:
                raise ValueError("Slice exceeds dimension extent!")
            step = s.step or 1
            offset_ += start * stride
            extent = 1 + (stop - start - 1) // step
            index_ = Index(extent=extent)
            indices[indexes.index(index)] = index_
            idxs_.append((index_, step * stride))
        dim2idxs_.append((offset_, tuple(idxs_)))

    expr = FlexiblyIndexed(variable, tuple(dim2idxs_))
    return ComponentTensor(expr, tuple(indices))


# Static one object for quicker constant folding
one = Literal(1)


# Syntax sugar
def indices(n):
    """Make some :class:`Index` objects.

    :arg n: The number of indices to make.
    :returns: A tuple of `n` :class:`Index` objects.
    """
    return tuple(Index() for _ in range(n))


def componentwise(op, *exprs):
    """Apply gem op to exprs component-wise and wrap up in a ComponentTensor.

    :arg op: function that returns a gem Node.
    :arg exprs: expressions to apply op to.
    :raises ValueError: if the expressions have mismatching shapes.
    :returns: New gem Node constructed from op.

    Each expression must either have the same shape, or else be
    scalar. Shaped expressions are indexed, the op is applied to the
    scalar expressions and the result is wrapped up in a ComponentTensor.

    """
    shapes = set(e.shape for e in exprs)
    if len(shapes - {()}) > 1:
        raise ValueError("expressions must have matching shape (or else be scalar)")
    shape = max(shapes)
    i = indices(len(shape))
    exprs = tuple(Indexed(e, i) if e.shape else e for e in exprs)
    return ComponentTensor(op(*exprs), i)


def as_gem(expr):
    """Attempt to convert an expression into GEM of scalar type.

    Parameters
    ----------
    expr : Node or Number
        The expression.

    Returns
    -------
    Node
        A GEM representation of the expression.

    Raises
    ------
    ValueError
        If conversion was not possible.

    """
    if isinstance(expr, Node):
        return expr
    elif isinstance(expr, Number):
        return Literal(expr)
    else:
        raise ValueError("Do not know how to convert %r to GEM" % expr)


def as_gem_uint(expr):
    """Attempt to convert an expression into GEM of uint type.

    Parameters
    ----------
    expr : Node or Integral
        The expression.

    Returns
    -------
    Node
        A GEM representation of the expression.

    Raises
    ------
    ValueError
        If conversion was not possible.

    """
    if isinstance(expr, Node):
        return expr
    elif isinstance(expr, Integral):
        return Literal(expr, dtype=uint_type)
    else:
        raise ValueError("Do not know how to convert %r to GEM" % expr)


def extract_type(expressions, klass):
    """Collects objects of type klass in expressions."""
    return tuple(node for node in traversal(expressions) if isinstance(node, klass))
