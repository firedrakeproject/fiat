import re


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
