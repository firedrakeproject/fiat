from FIAT.restricted import RestrictedElement
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.expansions import polynomial_dimension

def BrezziDouglasFortinMarini(ref_el, degree, variant=None):
    """The BDFM element"""
    BDM = BrezziDouglasMarini(ref_el, degree, variant=variant)

    entity_ids = BDM.dual.get_entity_ids()
    sd = ref_el.get_spatial_dimension()
    indices = []
    for dim in sorted(entity_ids):
        if dim == sd-1:
            s = slice(polynomial_dimension(ref_el.construct_subelement(dim), degree-1))
        else:
            s = slice(None)
        for entity in sorted(entity_ids[dim]):
            indices.extend(entity_ids[dim][entity][s])
    return RestrictedElement(BDM, indices)
