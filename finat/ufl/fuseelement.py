"""Element."""
# -*- coding: utf-8 -*-
# Copyright (C) 2025 India Marsden
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from finat.ufl.finiteelementbase import FiniteElementBase


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
        return self.triple.spaces[2].ufl_sobolev_space(self.triple.form_degree,
                                                       self.triple.cell.dim())

    def mapping(self):
        return self.triple.spaces[2].mapping()

    def reconstruct(self, family=None, cell=None, degree=None, quad_scheme=None, variant=None):
        return FuseElement(self.triple, cell=cell)
