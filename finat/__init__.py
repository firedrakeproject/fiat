from .fiat_elements import Bernstein
from .fiat_elements import Bubble, CrouzeixRaviart, DiscontinuousTaylor
from .fiat_elements import Lagrange, DiscontinuousLagrange, Real
from .fiat_elements import DPC, Serendipity, BrezziDouglasMariniCubeEdge, BrezziDouglasMariniCubeFace
from .fiat_elements import TrimmedSerendipityFace, TrimmedSerendipityEdge
from .fiat_elements import TrimmedSerendipityDiv
from .fiat_elements import TrimmedSerendipityCurl
from .fiat_elements import BrezziDouglasMarini, BrezziDouglasFortinMarini
from .fiat_elements import Nedelec, NedelecSecondKind, RaviartThomas
from .fiat_elements import HellanHerrmannJohnson, Regge
from .fiat_elements import GopalakrishnanLedererSchoberlFirstKind
from .fiat_elements import GopalakrishnanLedererSchoberlSecondKind
from .fiat_elements import FacetBubble
from .fiat_elements import KongMulderVeldhuizen

from .argyris import Argyris
from .aw import ArnoldWinther
from .aw import ArnoldWintherNC
from .hz import HuZhang
from .bell import Bell
from .bernardi_raugel import BernardiRaugel, BernardiRaugelBubble
from .hct import HsiehCloughTocher, ReducedHsiehCloughTocher
from .arnold_qin import ArnoldQin, ReducedArnoldQin
from .christiansen_hu import ChristiansenHu
from .alfeld_sorokina import AlfeldSorokina
from .guzman_neilan import GuzmanNeilanFirstKindH1, GuzmanNeilanSecondKindH1, GuzmanNeilanBubble, GuzmanNeilanH1div
from .powell_sabin import QuadraticPowellSabin6, QuadraticPowellSabin12
from .hermite import Hermite
from .johnson_mercier import JohnsonMercier
from .mtw import MardalTaiWinther
from .morley import Morley
from .trace import HDivTrace
from .direct_serendipity import DirectSerendipity

from .spectral import GaussLobattoLegendre, GaussLegendre, Legendre, IntegratedLegendre, FDMLagrange, FDMQuadrature, FDMDiscontinuousLagrange, FDMBrokenH1, FDMBrokenL2, FDMHermite  # noqa: F401
from .tensorfiniteelement import TensorFiniteElement  # noqa: F401
from .tensor_product import TensorProductElement  # noqa: F401
from .cube import FlattenedDimensions  # noqa: F401
from .discontinuous import DiscontinuousElement  # noqa: F401
from .enriched import EnrichedElement  # noqa: F401
from .hdivcurl import HCurlElement, HDivElement  # noqa: F401
from .mixed import MixedElement  # noqa: F401
from .nodal_enriched import NodalEnrichedElement  # noqa: 401
from .quadrature_element import QuadratureElement, make_quadrature_element  # noqa: F401
from .restricted import RestrictedElement          # noqa: F401
from .runtime_tabulated import RuntimeTabulated  # noqa: F401
from . import quadrature  # noqa: F401
from . import cell_tools  # noqa: F401

# List of supported elements and mapping to element classes
supported_elements = {"Argyris": Argyris,
                      "Bell": Bell,
                      "Bernardi-Raugel": BernardiRaugel,
                      "Bernardi-Raugel Bubble": BernardiRaugelBubble,
                      "Bernstein": Bernstein,
                      "Brezzi-Douglas-Fortin-Marini": BrezziDouglasFortinMarini,
                      "Brezzi-Douglas-Marini Cube Face": BrezziDouglasMariniCubeFace,
                      "Brezzi-Douglas-Marini": BrezziDouglasMarini,
                      "Brezzi-Douglas-Marini Cube Edge": BrezziDouglasMariniCubeEdge,
                      "Bubble": Bubble,
                      "FacetBubble": FacetBubble,
                      "Crouzeix-Raviart": CrouzeixRaviart,
                      "Direct Serendipity": DirectSerendipity,
                      "Discontinuous Lagrange": DiscontinuousLagrange,
                      "Discontinuous Lagrange L2": DiscontinuousLagrange,
                      "Discontinuous Taylor": DiscontinuousTaylor,
                      "Discontinuous Raviart-Thomas": lambda *args, **kwargs: DiscontinuousElement(RaviartThomas(*args, **kwargs)),
                      "DPC": DPC,
                      "DPC L2": DPC,
                      "Hermite": Hermite,
                      "Hsieh-Clough-Tocher": HsiehCloughTocher,
                      "Reduced-Hsieh-Clough-Tocher": ReducedHsiehCloughTocher,
                      "QuadraticPowellSabin6": QuadraticPowellSabin6,
                      "QuadraticPowellSabin12": QuadraticPowellSabin12,
                      "Alfeld-Sorokina": AlfeldSorokina,
                      "Arnold-Qin": ArnoldQin,
                      "Reduced-Arnold-Qin": ReducedArnoldQin,
                      "Christiansen-Hu": ChristiansenHu,
                      "Guzman-Neilan 1st kind H1": GuzmanNeilanFirstKindH1,
                      "Guzman-Neilan 2nd kind H1": GuzmanNeilanSecondKindH1,
                      "Guzman-Neilan H1(div)": GuzmanNeilanH1div,
                      "Guzman-Neilan Bubble": GuzmanNeilanBubble,
                      "Johnson-Mercier": JohnsonMercier,
                      "Lagrange": Lagrange,
                      "Kong-Mulder-Veldhuizen": KongMulderVeldhuizen,
                      "Gauss-Lobatto-Legendre": GaussLobattoLegendre,
                      "Gauss-Legendre": GaussLegendre,
                      "Gauss-Legendre L2": GaussLegendre,
                      "Legendre": Legendre,
                      "Integrated Legendre": IntegratedLegendre,
                      "Morley": Morley,
                      "Nedelec 1st kind H(curl)": Nedelec,
                      "Nedelec 2nd kind H(curl)": NedelecSecondKind,
                      "Raviart-Thomas": RaviartThomas,
                      "Real": Real,
                      "S": Serendipity,
                      "SminusF": TrimmedSerendipityFace,
                      "SminusDiv": TrimmedSerendipityDiv,
                      "SminusE": TrimmedSerendipityEdge,
                      "SminusCurl": TrimmedSerendipityCurl,
                      "Regge": Regge,
                      "HDiv Trace": HDivTrace,
                      "Hellan-Herrmann-Johnson": HellanHerrmannJohnson,
                      "Gopalakrishnan-Lederer-Schoberl 1st kind": GopalakrishnanLedererSchoberlFirstKind,
                      "Gopalakrishnan-Lederer-Schoberl 2nd kind": GopalakrishnanLedererSchoberlSecondKind,
                      "Conforming Arnold-Winther": ArnoldWinther,
                      "Nonconforming Arnold-Winther": ArnoldWintherNC,
                      "Hu-Zhang": HuZhang,
                      "Mardal-Tai-Winther": MardalTaiWinther}
