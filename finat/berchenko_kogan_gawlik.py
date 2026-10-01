import FIAT

from finat.citations import cite
from finat.fiat_elements import ScalarFiatElement, VectorFiatElement


class BerchenkoKoganGawlikH1(ScalarFiatElement):
    """The lowest-order blow-up Whitney 0-forms."""

    is_polynomial = False

    def __init__(self, cell, degree=1):
        cite("BerchenkoKoganGawlik2024")
        super().__init__(FIAT.BerchenkoKoganGawlik(cell, 0, degree=degree))


class BerchenkoKoganGawlikHCurl(VectorFiatElement):
    """The lowest-order blow-up Whitney 1-forms."""

    is_polynomial = False

    def __init__(self, cell, degree=1):
        cite("BerchenkoKoganGawlik2024")
        super().__init__(FIAT.BerchenkoKoganGawlik(cell, 1, degree=degree))


class BerchenkoKoganGawlikHDiv(VectorFiatElement):
    """The lowest-order blow-up Whitney 1-forms, rotated into H(div)."""

    is_polynomial = False

    def __init__(self, cell, degree=1):
        cite("BerchenkoKoganGawlik2024")
        super().__init__(FIAT.BerchenkoKoganGawlik(cell, 1, degree=degree, rotated=True))


class BerchenkoKoganGawlikL2(ScalarFiatElement):
    """The lowest-order blow-up Whitney 2-forms."""

    def __init__(self, cell, degree=1):
        cite("BerchenkoKoganGawlik2024")
        super().__init__(FIAT.BerchenkoKoganGawlik(cell, 2, degree=degree))
