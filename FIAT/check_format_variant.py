import re
from FIAT.macro import AlfeldSplit, IsoSplit

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


def check_format_variant(variant, degree):
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

    return variant, interpolant_degree


def parse_lagrange_variant(variant, discontinuous=False):
    if variant is None:
        variant = "equispaced"
    options = variant.replace(" ", "").split(",")
    assert len(options) <= 2

    if discontinuous:
        supported_point_variants = supported_dg_variants
    else:
        supported_point_variants = supported_cg_variants

    # defaults
    splitting = None
    point_variant = supported_point_variants["spectral"]

    for pre_opt in options:
        opt = pre_opt.lower()
        if opt == "alfeld":
            splitting = AlfeldSplit
        elif opt == "iso":
            splitting = IsoSplit
        elif opt.startswith("iso"):
            match = re.match(r"^iso(?:\((\d+)\))?$", opt)
            k, = match.groups()
            splitting = lambda T: IsoSplit(T, int(k))
        elif opt in supported_point_variants:
            point_variant = supported_point_variants[opt]
        else:
            raise ValueError("Illegal variant option")

    if discontinuous and splitting is not None and point_variant in supported_cg_variants.values():
        raise ValueError("Illegal variant. DG macroelements with DOFs on subcell boundaries are not unisolvent.")
    return splitting, point_variant
