from finat.physically_mapped import PhysicallyMappedElement, identity
from finat.tensor_product import TensorProductElement
from finat.hermite import Hermite
from functools import reduce
from gem import Product
import numpy


class PhysicallyMappedTensorProductElement(PhysicallyMappedElement, TensorProductElement):

    def __init__(self, factors):
        TensorProductElement.__init__(self, factors)

    def basis_transformation(self, coordinate_mapping=None):
        M = []
        for f in self.factors:
            if isinstance(f, PhysicallyMappedElement):
                M.append(f.basis_transformation(coordinate_mapping))
            else:
                M.append(identity(f.space_dimension()))
        return Product(*M)


class BognerFoxSchmit(PhysicallyMappedTensorProductElement):

    def __init__(self, cell, degree=3):
        factors = [Hermite(sub_cell, degree) for sub_cell in cell.product.cells]
        super().__init__(factors)
