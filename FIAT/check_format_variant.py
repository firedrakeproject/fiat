import re


def check_format_variant(variant, degree):
    if variant is None:
        variant = "integral"

    match = re.match(r"^integral(?:\((\d+)\))?$", variant)
    if match:
        variant = "integral"
        quad_degree, = match.groups()
        quad_degree = int(quad_degree) if quad_degree is not None else (degree + 1)
        if quad_degree < degree + 1:
            raise ValueError("Warning, quadrature degree should be at least %s" % (degree + 1))
    elif variant == "point":
        quad_degree = None
    else:
        raise ValueError('Choose either variant="point" or variant="integral"'
                         'or variant="integral(Quadrature degree)"')

    return (variant, quad_degree)
