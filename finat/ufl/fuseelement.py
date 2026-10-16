"""Element."""
# -*- coding: utf-8 -*-
# Copyright (C) 2025 India Marsden
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from finat.ufl.finiteelementbase import FiniteElementBase
from ufl.sobolevspace import H1, HCurl, HDiv, L2

_SOBOLEV_SPACES = {"H1": H1, "HCurl": HCurl, "HDiv": HDiv, "L2": L2}


class FuseElement(FiniteElementBase):
    """
    A finite element defined using FUSE.

    :arg triple: An ElementTriple object defined with FUSE
    :arg cell: Optional (defaults to triple.cell) The cell the element is defined on

    """

    def __init__(self, triple, cell=None):
        try:
            import fuse
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FUSE element creation requires the optional 'fuse' dependency. "
            ) from exc
        assert isinstance(triple, fuse.ElementTriple)
        self.triple = triple
        if not cell:
            cell = self.triple.cell.to_ufl()

        degree = self.triple.degree
        super().__init__("FUSE", cell, degree, None, triple.get_value_shape())

    def __repr__(self):
        return repr(self.triple)

    def __str__(self):
        return f"<FuseElem on {self.triple.cell}>"

    @property
    def sobolev_space(self):
        return _SOBOLEV_SPACES[self.triple.sobolev_space()]

    def mapping(self):
        return self.triple.pullback()

    def reconstruct(self, family=None, cell=None, degree=None, quad_scheme=None, variant=None):
        return FuseElement(self.triple, cell=cell)
