# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2021

from FIAT import discontinuous_lagrange


class GaussLegendre(discontinuous_lagrange.DiscontinuousLagrange):
    """Simplicial discontinuous element with nodes at the (recursive) Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        super().__init__(ref_el, degree, variant="gl")
