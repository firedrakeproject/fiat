from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import mis
from FIAT import functional
import symfem
from sympy import lambdify,diff,Expr
from sympy.abc import x,y,z
import numpy

def SymFEM_initialize_entity_ids(topology,ref_map):
    entity_ids = {}
    for (i, entity) in list(topology.items()):
        entity_ids[i] = {}
        for j in entity:
            entity_ids[i][j] = ref_map(i,j)
    return entity_ids

def SymFEMRefToUFLRef(P):
    return tuple(map(lambda i: 2*i-1,P))

class SymFEMDualSet(DualSet):
    """The dual basis for Bernstein elements."""

    def __init__(self, ref_el, degree,sym_el):
        topology = ref_el.get_topology()
        entity_ids = SymFEM_initialize_entity_ids(topology,sym_el.entity_dofs)
        nodes =  [None for i in range(len(sym_el.dofs))]
        i=0
        for dof in sym_el.dofs:
            if dof.__class__ == symfem.functionals.PointEvaluation:
                #Wrap DoF of Point Evaluation type
                P = tuple(map(numpy.float64,dof.point));
                nodes[i] = functional.PointEvaluation(ref_el,P)#SymFEMRefToUFLRef(P)) 

            else:
                raise RuntimeError('This type of DoF has not yet been wrapped from SymFEM.') 
            i=i+1
        super(SymFEMDualSet, self).__init__(nodes, ref_el, entity_ids)

class SymFEM(FiniteElement):
    """A finite element generated from symfem."""

    def __init__(self, ref_el,entity_ids,dual,degree,k,mapping="affine"):
        self.order = degree
        self.formdegree = k
        self.ref_el = ref_el
        self.dual = dual
        self.entity_ids = entity_ids
        self._mapping = mapping
    def generateBasis(self):
        self.sym_basis = self.sym_el.get_basis_functions()
        self.numberBases = len(self.sym_basis)
        self.B = {}
        for i in range(self.numberBases):
            self.B[i] = {};
            self.B[i][(0,0)] = lambdify([(x,y)],self.sym_basis[i]._sympy_())
            self.B[i][(1,0)] = lambdify([(x,y)],diff(self.sym_basis[i]._sympy_(),x))
            self.B[i][(0,1)] = lambdify([(x,y)],diff(self.sym_basis[i]._sympy_(),y))

    def degree(self):
        """The degree of the polynomial space."""
        return self.get_order()

    def value_shape(self):
        """The value shape of the finite element functions."""
        return ()

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.entity_ids

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        
        if order > 1:
            raise RuntimeError("SymFEM are only implemented for first derivative operator.");
        # Transform points to reference cell coordinates
        ref_el = self.get_reference_element()
        if entity is None:
            entity = (ref_el.get_spatial_dimension(), 0)

        entity_dim, entity_id = entity
        entity_transform = ref_el.get_entity_transform(entity_dim, entity_id)
        cell_points = list(map(entity_transform, points))

        # Evaluate everything
        deg = self.degree()
        dim = ref_el.get_spatial_dimension()

        # Rearrange result
        space_dim = self.space_dimension()
        dtype = numpy.float64#numpy.array(list(raw_result.values())).dtype
        result = {alpha: numpy.zeros((space_dim, len(cell_points)), dtype=dtype)
                  for o in range(order + 1)
                  for alpha in mis(dim, o)}
        for i in range(self.numberBases):
            for o in range(order+1):
                for alpha, vec in self.basisFunction(cell_points, i, o).items():
                    result[alpha][i, :] = vec
        return result
    
    def basisFunction(self,points,i, order):
        points = numpy.asarray(points);
        N, d_1 = points.shape
        result = {}
        for alpha in mis(d_1, order):
            values = self.B[i][alpha](points.T);
            result[alpha] = values
        return result
