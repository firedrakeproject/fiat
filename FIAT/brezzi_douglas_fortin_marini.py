from FIAT.expansions import polynomial_dimension
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.nodal_enriched import NodalEnrichedElement
from FIAT.restricted import RestrictedElement


def BrezziDouglasFortinMarini(ref_el, degree, variant=None):
    """The BDFM element"""
    if variant == "point":
        BDM_I = RestrictedElement(BrezziDouglasMarini(ref_el, degree, variant=variant), restriction_domain="interior")
        BDM_F = RestrictedElement(BrezziDouglasMarini(ref_el, degree-1, variant=variant), restriction_domain="facet")
        return NodalEnrichedElement(BDM_I, BDM_F)
    else:
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
