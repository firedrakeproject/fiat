import re
import warnings


def check_format_variant(variant, degree, element):
    if variant is None:
        variant = "point"
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('Variant of ' + element + ' element will change from point evaluation to integral evaluation.'
                      ' You should project into variant="integral"', DeprecationWarning)
        # Replace by the following in a month time
        # variant = "integral"

    if not (variant == "point" or "integral" in variant):
        raise ValueError('Choose either variant="point" or variant="integral"'
                         'or variant="integral(Quadrature degree)"')

    if variant == "integral":
        # quadrature is so high to ensure that the interpolant of curl/divergence-free functions is still curl/divergence-free
        quad_deg = 5 * (degree + 1)
        variant = "integral"
    elif re.match(r'^integral\(\d+\)$', variant):
        quad_deg = int(''.join(filter(str.isdigit, variant)))
        if quad_deg < degree + 1:
            raise ValueError("Warning, quadrature degree should be at least %s" % (degree + 1))
        variant = "integral"
    elif variant == "point":
        quad_deg = None
    else:
        raise ValueError("Wrong format for variant")

    return (variant, quad_deg)
