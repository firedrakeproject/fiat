import re

from FIAT.macro import IsoSplit, AlfeldSplit, WorseyFarinSplit, PowellSabinSplit, PowellSabin12Split

# dicts mapping Lagrange variant names to recursivenodes family names
supported_cg_variants = {
    "spectral": "gll",
    "chebyshev": "lgc",
    "equispaced": "equispaced",
    "gll": "gll"}

supported_dg_variants = {
    "spectral": "gl",
    "chebyshev": "gc",
    "equispaced": "equispaced",
    "equispaced_interior": "equispaced_interior",
    "gll": "gll",
    "gl": "gl"}

supported_splits = {
    "iso": IsoSplit,
    "alfeld": AlfeldSplit,
    "worsey-farin": WorseyFarinSplit,
    "powell-sabin": PowellSabinSplit,
    "powell-sabin(12)": PowellSabin12Split,
}


def check_format_variant(variant, degree):
    splitting, variant = parse_lagrange_variant(variant, integral=True)

    if variant is None:
        variant = "integral"

    match = re.match(r"^integral(?:\((\d+)\))?$", variant)
    if match:
        variant = "integral"
        extra_degree, = match.groups()
        extra_degree = int(extra_degree) if extra_degree is not None else 0
        interpolant_degree = degree + extra_degree
        if interpolant_degree < degree:
            raise ValueError("Warning, quadrature degree should be at least %s" % degree)
    elif variant == "point":
        interpolant_degree = None
    else:
        raise ValueError('Choose either variant="point" or variant="integral"'
                         'or variant="integral(q)"')

    return splitting, variant, interpolant_degree


def parse_lagrange_variant(variant, discontinuous=False, integral=False):
    """Parses variant options for Lagrange elements.

    variant may be a single option or comma-separated pair
    indicating the dof type (integral, equispaced, spectral, etc)
    and the type of splitting to give a macro-element (Alfeld, Powell-Sabin, iso)
    """
    if variant is None:
        variant = "integral" if integral else "equispaced"
    options = variant.replace(" ", "").split(",")
    assert len(options) <= 2

    default = "integral" if integral else "spectral"
    if integral:
        supported_point_variants = {"integral": None, "point": "point"}
    elif discontinuous:
        supported_point_variants = supported_dg_variants
    else:
        supported_point_variants = supported_cg_variants

    # defaults
    splitting = None
    splitting_args = tuple()
    point_variant = supported_point_variants[default]

    for pre_opt in options:
        opt = pre_opt.lower()
        if opt in supported_splits:
            splitting = supported_splits[opt]
        elif opt.startswith("iso"):
            match = re.match(r"^iso(?:\((\d+)\))?$", opt)
            k, = match.groups()
            call_split = IsoSplit
            splitting_args = (int(k),)
        elif opt.startswith("integral"):
            point_variant = opt
        elif opt in supported_point_variants:
            point_variant = supported_point_variants[opt]
        else:
            raise ValueError("Illegal variant option")

    if discontinuous and splitting is not None and point_variant in supported_cg_variants.values():
        raise ValueError("Illegal variant. DG macroelements with DOFs on subcell boundaries are not unisolvent.")
    if len(splitting_args) > 0:
        splitting = lambda T: call_split(T, *splitting_args, point_variant or "gll")
    return splitting, point_variant
