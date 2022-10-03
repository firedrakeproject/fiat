from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import mis
from FIAT.symfem import *
from FIAT import functional,quadrature
import symfem
import numpy

class CrouzeixFalk(SymFEM):
    """An implementation fo the Crouzeix Falk finite element."""

    def __init__(self, ref_el, degree):
        k = 0  # 0-formi
        topology = ref_el.get_topology()
        print("Degree {}".format(degree))
        self.sym_el = symfem.create_element("triangle", "CF", degree);

        entity_ids = SymFEM_initialize_entity_ids(topology,self.sym_el.entity_dofs)
        dual = SymFEMDualSet(ref_el, degree,self.sym_el)
        self.generateBasis();
        super(CrouzeixFalk, self).__init__(ref_el, entity_ids, dual, degree, k)
