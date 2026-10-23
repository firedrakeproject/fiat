"""Schedules operations to evaluate a multi-root expression DAG,
forming an ordered list of Impero terminals."""

import collections
import functools
import itertools

from gem import gem, impero
from gem.node import collect_refcount


class OrderedDefaultDict(collections.OrderedDict):
    """A dictionary that provides a default value and ordered iteration.

    :arg factory: The callable used to create the default value.

    See :class:`collections.OrderedDict` for description of the
    remaining arguments.
    """
    def __init__(self, factory, *args, **kwargs):
        self.factory = factory
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        val = self[key] = self.factory()
        return val


class ReferenceStager:
    """Provides staging for nodes in reference counted expression
    DAGs.  A callback function is called once the reference count is
    exhausted."""

    def __init__(self, reference_count, callback):
        """Initialises a ReferenceStager.

        :arg reference_count: initial reference counts for all
                              expected nodes
        :arg callback: function to call on each node when
                       reference count is exhausted
        """
        self.waiting = reference_count.copy()
        self.callback = callback

    def decref(self, o):
        """Decreases the reference count of a node, and possibly
        triggering a callback (when the reference count drops to
        zero)."""
        assert 1 <= self.waiting[o]

        self.waiting[o] -= 1
        if self.waiting[o] == 0:
            self.callback(o)

    def empty(self):
        """All reference counts exhausted?"""
        return not any(self.waiting.values())


class Queue:
    """Special queue for operation scheduling.  GEM / Impero nodes are
    inserted when they are ready to be scheduled, i.e. any operation
    which depends on the operation to be inserted must have been
    scheduled already.  This class implements a heuristic for ordering
    operations within the constraints in a way which aims to achieve
    maximum loop fusion to minimise the size of temporaries which need
    to be introduced.
    """
    def __init__(self, callback):
        """Initialises a Queue.

        :arg callback: function called on each element "popped" from the queue
        """
        # Must have deterministic iteration over the queue
        self.queue = OrderedDefaultDict(list)
        self.callback = callback

    def insert(self, indices, elem):
        """Insert element into queue.

        :arg indices: loop indices used by the scheduling heuristic
        :arg elem: element to be scheduled
        """
        self.queue[indices].append(elem)

    def process(self):
        """Pops elements from the queue and calls the callback
        function on them until the queue is empty.  The callback
        function can insert further elements into the queue.
        """
        indices = ()
        while self.queue:
            # Find innermost non-empty outer loop
            while indices not in (i[:len(indices)] for i in self.queue.keys()):
                indices = indices[:-1]

            # Pick a loop
            for i in self.queue.keys():
                if i[:len(indices)] == indices:
                    indices = i
                    break

            while self.queue[indices]:
                self.callback(self.queue[indices].pop())
            del self.queue[indices]


def handle(ops, depends_on, push, decref, node):
    """Helper function for scheduling.

    The idea is:

    ``ops`` is the sequence of scheduled operations (in reverse order). When we
    append to it we are therefore concretely scheduling something.

    When we consume something we say "we have now consumed this attribute" and can
    decref its sub-attributes so that they may then get processed. When a reference
    count drops to zero then the object has no outstanding dependencies and can be scheduled.

    """
    if isinstance(node, gem.Variable):
        # Declared in the kernel header
        pass
    elif isinstance(node, gem.Constant):
        # Constant literals inlined, unless tensor-valued
        if node.shape:
            ops.append(impero.Evaluate(node))
    elif isinstance(node, gem.Zero):  # should rarely happen
        assert not node.shape
    elif isinstance(node, (gem.Indexed, gem.FlexiblyIndexed)):
        for child in itertools.chain(node.children, node.indirect_children):
            decref(child)
    elif isinstance(node, gem.IndexSum):
        ops.append(impero.Noop(node))
        push(impero.Accumulate(node))
    elif isinstance(node, gem.Node):
        ops.append(impero.Evaluate(node))  # might want to materialise this as a temporary
        for child in node.children:
            decref(child)
    elif isinstance(node, impero.Initialise):
        ops.append(node)
    elif isinstance(node, impero.Accumulate):
        ops.append(node)
        push(impero.Initialise(node.indexsum))
        decref(node.indexsum.children[0])
    elif isinstance(node, impero.Return):
        ops.append(node)
        decref(node.expression)
    elif isinstance(node, impero.ReturnAccumulate):
        ops.append(node)
        decref(node.indexsum.children[0])
    elif isinstance(node, impero.Assignment):
        ops.append(node)
        decref(node.expression)
        # maybe we can now emit instructions that this instruction depends on
        for op in depends_on[node]:
            decref(op)
    else:
        raise AssertionError("no handler for node type %s" % type(node))


def emit_operations(assignments, get_indices, emit_return_accumulate=True):
    """Makes an ordering of operations to evaluate a multi-root
    expression DAG.

    :arg assignments: Iterable of (variable, expression) pairs.
                      The value of expression is written into variable
                      upon execution.
    :arg get_indices: mapping from GEM nodes to an ordering of free
                      indices
    :arg emit_return_accumulate: emit ReturnAccumulate nodes? Set to
                      False if the output variables are not guaranteed
                      zero on entry to the kernel.
    :returns: list of Impero terminals correctly ordered to evaluate
              the assignments
    """
    # Prepare reference counts
    refcount = collect_refcount([e for v, e in assignments])

    # Stage return operations
    staging = []
    for variable, expression in assignments:
        if emit_return_accumulate and \
                refcount[expression] == 1 and isinstance(expression, gem.IndexSum) \
                and set(variable.free_indices) == set(expression.free_indices):
            staging.append(impero.ReturnAccumulate(variable, expression))
            refcount[expression] -= 1
        else:
            staging.append(impero.Return(variable, expression))

    # Prepare data structures
    def push_node(node):
        queue.insert(get_indices(node), node)

    def push_op(op):
        queue.insert(op.loop_shape(get_indices), op)

    ops = []

    stager = ReferenceStager(refcount, push_node)
    queue = Queue(functools.partial(handle, ops, "unused", push_op, stager.decref))

    # Enqueue return operations
    for op in staging:
        push_op(op)

    # Schedule operations
    queue.process()

    # Assert that nothing left unprocessed
    assert stager.empty()

    # Return
    ops.reverse()
    return ops


def emit_operations_new(assignments, get_indices, emit_return_accumulate=True):
    """Makes an ordering of operations to evaluate a multi-root
    expression DAG.

    :arg assignments: Iterable of (variable, expression) pairs.
                      The value of expression is written into variable
                      upon execution.
    :arg get_indices: mapping from GEM nodes to an ordering of free
                      indices
    :arg emit_return_accumulate: emit ReturnAccumulate nodes? Set to
                      False if the output variables are not guaranteed
                      zero on entry to the kernel.
    :returns: list of Impero terminals correctly ordered to evaluate
              the assignments
    """
    # Prepare reference counts
    # refcount = collect_refcount([a.assignee for a in assignments]+[a.expression for a in assignments])
    refcount = collect_refcount([a.expression for a in assignments])

    # some assignments depend on each other, for now assume all do
    #
    # consider
    #
    #   y <- x
    #   a <- b
    #   z <- y
    #
    # We can emit instructions for a and z immediately as they have no downstream
    # consumers. y must be assigned to before z though, so it must have an extra reference.
    # NOTE: What about RW access? Produces a cyclic reference
    #
    #   t <- dat
    #   f(t)
    #   dat <- t
    #
    # FIXME: It's a hack but do things in reverse order for now because of ordering issues
    # with the output temporary (e.g. don't zero it after computing it)
    import pyop3.collections
    ready_to_schedule = pyop3.collections.OrderedSet(list(reversed(assignments)))
    depends_on = collections.defaultdict(list)
    for insn, maybe_dep in itertools.permutations(reversed(assignments), 2):
        # Determine if maybe_dep is a dependency of insn by looking to see if its
        # LHS appears in insn's RHS
        insn_refcount = collect_refcount([insn.expression])
        if maybe_dep.assignee.aggregate in insn_refcount:
            # maybe_dep LHS appears in the RHS expression, don't emit an
            # instruction until that is dealt with first.
            refcount[maybe_dep] += 1
            ready_to_schedule.discard(maybe_dep)
            depends_on[insn].append(maybe_dep)

    assert ready_to_schedule, "Must have something to start with"

    # TODO: not handling emit_return_accumulate any more!
    # Stage return operations
    # staging = []
    # for a in assignments:  # do in reverse order so final order is closer to what we expect, avoid arbitrary movement
    #     variable = a.assignee
    #     expression = a.expression
    #     if emit_return_accumulate and \
    #             refcount[expression] == 1 and isinstance(expression, gem.IndexSum) \
    #             and set(variable.free_indices) == set(expression.free_indices):
    #         staging.append(impero.Assignment(variable, expression, "inc"))
    #         refcount[expression] -= 1
    #     else:
    #         staging.append(impero.Assignment(variable, expression, "write"))
    # staging = list(assignments)

    # Callbacks executed when a reference count hits zero, push the node (either
    # GEM or Impero) onto the stack to be executed.
    def push(node):
        if isinstance(node, gem.Node):
            queue.insert(get_indices(node), node)
        else:
            assert isinstance(node, impero.Node)
            queue.insert(node.loop_shape(get_indices), node)

    ops = []

    stager = ReferenceStager(refcount, push)
    queue = Queue(lambda node: handle(ops, depends_on, push, stager.decref, node))

    # Enqueue return operations
    for op in ready_to_schedule:
        push(op)

    # Schedule operations
    queue.process()

    # Assert that nothing left unprocessed
    assert stager.empty()

    # Return
    ops.reverse()
    return ops
