"""FInite element Automatic Tabulator -- supports constructing and
evaluating arbitrary order Lagrange and many other elements.
Simplices in one, two, and three dimensions are supported."""

# Important functionality
from FIAT.reference_element import ufc_cell, ufc_simplex       # noqa: F401
from FIAT.quadrature import make_quadrature                    # noqa: F401
from FIAT.quadrature_schemes import create_quadrature          # noqa: F401
from FIAT.finite_element import FiniteElement, CiarletElement  # noqa: F401
from FIAT.hdivcurl import Hdiv, Hcurl                          # noqa: F401
from FIAT.mixed import MixedElement                            # noqa: F401
from FIAT.restricted import RestrictedElement                  # noqa: F401
from FIAT.quadrature_element import QuadratureElement          # noqa: F401
from FIAT.tensor_product import TensorProductElement           # noqa: F401
from FIAT.enriched import EnrichedElement                      # noqa: F401
from FIAT.nodal_enriched import NodalEnrichedElement           # noqa: F401
from FIAT.discontinuous import DiscontinuousElement            # noqa: F401

# Import finite element classes
from FIAT.argyris import Argyris
from FIAT.bernardi_raugel import BernardiRaugel
from FIAT.bernstein import Bernstein
from FIAT.bell import Bell
from FIAT.hct import HsiehCloughTocher
from FIAT.alfeld_sorokina import AlfeldSorokina
from FIAT.arnold_qin import ArnoldQin
from FIAT.guzman_neilan import GuzmanNeilanFirstKindH1, GuzmanNeilanSecondKindH1, GuzmanNeilanH1div
from FIAT.christiansen_hu import ChristiansenHu
from FIAT.johnson_mercier import JohnsonMercier
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini
from FIAT.Sminus import TrimmedSerendipityEdge, TrimmedSerendipityFace
from FIAT.SminusDiv import TrimmedSerendipityDiv
from FIAT.SminusCurl import TrimmedSerendipityCurl
from FIAT.brezzi_douglas_fortin_marini import BrezziDouglasFortinMarini
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.discontinuous_taylor import DiscontinuousTaylor
from FIAT.discontinuous_raviart_thomas import DiscontinuousRaviartThomas
from FIAT.serendipity import Serendipity
from FIAT.brezzi_douglas_marini_cube import BrezziDouglasMariniCubeEdge, BrezziDouglasMariniCubeFace
from FIAT.discontinuous_pc import DPC
from FIAT.hermite import CubicHermite
from FIAT.lagrange import Lagrange
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
from FIAT.gauss_legendre import GaussLegendre
from FIAT.gauss_radau import GaussRadau
from FIAT.morley import Morley
from FIAT.nedelec import Nedelec
from FIAT.nedelec_second_kind import NedelecSecondKind
from FIAT.powell_sabin import QuadraticPowellSabin6, QuadraticPowellSabin12
from FIAT.hierarchical import Legendre, IntegratedLegendre
from FIAT.P0 import P0
from FIAT.raviart_thomas import RaviartThomas
from FIAT.crouzeix_raviart import CrouzeixRaviart
from FIAT.regge import Regge
from FIAT.gopalakrishnan_lederer_schoberl import GopalakrishnanLedererSchoberlFirstKind
from FIAT.gopalakrishnan_lederer_schoberl import GopalakrishnanLedererSchoberlSecondKind
from FIAT.hellan_herrmann_johnson import HellanHerrmannJohnson
from FIAT.arnold_winther import ArnoldWinther
from FIAT.arnold_winther import ArnoldWintherNC
from FIAT.hu_zhang import HuZhang
from FIAT.mardal_tai_winther import MardalTaiWinther
from FIAT.bubble import Bubble, FacetBubble
from FIAT.hdiv_trace import HDivTrace
from FIAT.kong_mulder_veldhuizen import KongMulderVeldhuizen
from FIAT.histopolation import Histopolation
from FIAT.fdm_element import FDMLagrange, FDMDiscontinuousLagrange, FDMQuadrature, FDMBrokenH1, FDMBrokenL2, FDMHermite  # noqa: F401

# List of supported elements and mapping to element classes
supported_elements = {"Argyris": Argyris,
                      "Bell": Bell,
                      "Bernardi-Raugel": BernardiRaugel,
                      "Bernstein": Bernstein,
                      "Brezzi-Douglas-Marini": BrezziDouglasMarini,
                      "Brezzi-Douglas-Fortin-Marini": BrezziDouglasFortinMarini,
                      "Bubble": Bubble,
                      "FacetBubble": FacetBubble,
                      "Crouzeix-Raviart": CrouzeixRaviart,
                      "Discontinuous Lagrange": DiscontinuousLagrange,
                      "S": Serendipity,
                      "SminusF": TrimmedSerendipityFace,
                      "SminusDiv": TrimmedSerendipityDiv,
                      "SminusE": TrimmedSerendipityEdge,
                      "SminusCurl": TrimmedSerendipityCurl,
                      "Brezzi-Douglas-Marini Cube Face": BrezziDouglasMariniCubeFace,
                      "Brezzi-Douglas-Marini Cube Edge": BrezziDouglasMariniCubeEdge,
                      "DPC": DPC,
                      "Discontinuous Taylor": DiscontinuousTaylor,
                      "Discontinuous Raviart-Thomas": DiscontinuousRaviartThomas,
                      "Hermite": CubicHermite,
                      "Hsieh-Clough-Tocher": HsiehCloughTocher,
                      "QuadraticPowellSabin6": QuadraticPowellSabin6,
                      "QuadraticPowellSabin12": QuadraticPowellSabin12,
                      "Alfeld-Sorokina": AlfeldSorokina,
                      "Arnold-Qin": ArnoldQin,
                      "Christiansen-Hu": ChristiansenHu,
                      "Guzman-Neilan 1st kind H1": GuzmanNeilanFirstKindH1,
                      "Guzman-Neilan 2nd kind H1": GuzmanNeilanSecondKindH1,
                      "Guzman-Neilan H1(div)": GuzmanNeilanH1div,
                      "Johnson-Mercier": JohnsonMercier,
                      "Lagrange": Lagrange,
                      "Kong-Mulder-Veldhuizen": KongMulderVeldhuizen,
                      "Gauss-Lobatto-Legendre": GaussLobattoLegendre,
                      "Gauss-Legendre": GaussLegendre,
                      "Gauss-Radau": GaussRadau,
                      "Histopolation": Histopolation,
                      "Legendre": Legendre,
                      "Integrated Legendre": IntegratedLegendre,
                      "Morley": Morley,
                      "Nedelec 1st kind H(curl)": Nedelec,
                      "Nedelec 2nd kind H(curl)": NedelecSecondKind,
                      "Raviart-Thomas": RaviartThomas,
                      "Regge": Regge,
                      "HDiv Trace": HDivTrace,
                      "Hellan-Herrmann-Johnson": HellanHerrmannJohnson,
                      "Gopalakrishnan-Lederer-Schoberl 1st kind": GopalakrishnanLedererSchoberlFirstKind,
                      "Gopalakrishnan-Lederer-Schoberl 2nd kind": GopalakrishnanLedererSchoberlSecondKind,
                      "Conforming Arnold-Winther": ArnoldWinther,
                      "Nonconforming Arnold-Winther": ArnoldWintherNC,
                      "Hu-Zhang": HuZhang,
                      "Mardal-Tai-Winther": MardalTaiWinther}

# List of extra elements
extra_elements = {"P0": P0}
