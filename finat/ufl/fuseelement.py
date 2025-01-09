"""Element."""
# -*- coding: utf-8 -*-
# Copyright (C) 2025 India Marsden
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from finat.ufl.finiteelementbase import FiniteElementBase

class FuseElement(FiniteElementBase):
    """
    A finite element defined using FUSE.

    TODO: Need to deal with cases where value shape and reference value shape are different
    """

    def __init__(self, triple, cell=None):
        self.triple = triple
        if not cell:
            cell = self.triple.cell.to_ufl()

        # this isn't really correct
        degree = self.triple.spaces[0].degree()
        super(FuseElement, self).__init__("IT", cell, degree, None, triple.get_value_shape())

    def __repr__(self):
        return "FiniteElement(%s, %s, (%s, %s, %s), %s)" % (
               repr(self.triple.DOFGenerator), repr(self.triple.cell), repr(self.triple.spaces[0]), repr(self.triple.spaces[1]), repr(self.triple.spaces[2]), "X")

    def __str__(self):
        return "<Fuse%sElem on %s>" % (self.triple.spaces[0], self.triple.cell)

    def mapping(self):
        if str(self.sobolev_space) == "HCurl":
            return "covariant Piola"
        elif str(self.sobolev_space) == "HDiv":
            return "contravariant Piola"
        else:
            return "identity"

    def sobolev_space(self):
        return self.triple.spaces[1]

    def reconstruct(self, family=None, cell=None, degree=None, quad_scheme=None, variant=None):
        return FuseElement(self.triple, cell=cell)
