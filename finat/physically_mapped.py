from abc import ABCMeta, abstractmethod

import gem
import numpy

try:
    from firedrake_citations import Citations
    Citations().add("Kirby2018zany", """
@Article{Kirby2018zany,
  author =       {Robert C. Kirby},
  title =        {A general approach to transforming finite elements},
  journal =      {SMAI Journal of Computational Mathematics},
  year =         2018,
  volume =       4,
  pages =        {197-224},
  doi =          {10.5802/smai-jcm.33},
  archiveprefix ={arXiv},
  eprint =       {1706.09017},
  primaryclass = {math.NA}
}
""")
    Citations().add("Kirby2019zany", """
@Article{Kirby:2019,
  author =       {Robert C. Kirby and Lawrence Mitchell},
  title =        {Code generation for generally mapped finite
                  elements},
  journal =      {ACM Transactions on Mathematical Software},
  year =         2019,
  volume =       45,
  number =       41,
  pages =        {41:1--41:23},
  doi =          {10.1145/3361745},
  archiveprefix ={arXiv},
  eprint =       {1808.05513},
  primaryclass = {cs.MS}
}""")
    Citations().add("Clough1965", """
@inproceedings{Clough1965,
  author =       {R. W. Clough, J. L. Tocher},
  title =        {Finite element stiffness matrices for analysis of plate bending},
  booktitle =    {Proc. of the First Conf. on Matrix Methods in Struct. Mech},
  year =         1965,
  pages =        {515-546},
}
""")
    Citations().add("Argyris1968", """
@Article{Argyris1968,
  author =       {J. H. Argyris and I. Fried and D. W. Scharpf},
  title =        {{The TUBA family of plate elements for the matrix
                  displacement method}},
  journal =      {The Aeronautical Journal},
  year =         1968,
  volume =       72,
  pages =        {701-709},
  doi =          {10.1017/S000192400008489X}
}
""")
    Citations().add("Bell1969", """
@Article{Bell1969,
  author =       {Kolbein Bell},
  title =        {A refined triangular plate bending finite element},
  journal =      {International Journal for Numerical Methods in
                  Engineering},
  year =         1969,
  volume =       1,
  number =       1,
  pages =        {101-122},
  doi =          {10.1002/nme.1620010108}
}
""")
    Citations().add("Ciarlet1972", r"""
@Article{Ciarlet1972,
  author =       {P. G. Ciarlet and P. A. Raviart},
  title =        {{General Lagrange and Hermite interpolation in
                  $\mathbb{R}^n$ with applications to finite element
                  methods}},
  journal =      {Archive for Rational Mechanics and Analysis},
  year =         1972,
  volume =       46,
  number =       3,
  pages =        {177-199},
  doi =          {10.1007/BF0025245}
}
""")
    Citations().add("Morley1971", """
@Article{Morley1971,
  author =       {L. S. D. Morley},
  title =        {The constant-moment plate-bending element},
  journal =      {The Journal of Strain Analysis for Engineering
                  Design},
  year =         1971,
  volume =       6,
  number =       1,
  pages =        {20-24},
  doi =          {10.1243/03093247V061020}
}
""")
    Citations().add("Mardal2002", """
@article{Mardal2002,
        doi = {10.1137/s0036142901383910},
        year = 2002,
        volume = {40},
        number = {5},
        pages = {1605--1631},
        author = {Mardal, K.-A.~ and Tai, X.-C.~ and Winther, R.~},
        title = {A robust finite element method for {Darcy--Stokes} flow},
        journal = {{SIAM} Journal on Numerical Analysis}
}
""")
    Citations().add("Arnold2002", """
@article{Arnold2002,
        doi = {10.1007/s002110100348},
        year = 2002,
        volume = {92},
        number = {3},
        pages = {401--419},
        author = {Arnold, R.~N.~ and Winther, R.~},
        title = {Mixed finite elements for elasticity},
        journal = {Numerische Mathematik}
}
""")
    Citations().add("Arnold2003", """
@article{arnold2003,
        doi = {10.1142/s0218202503002507},
        year = 2003,
        volume = {13},
        number = {03},
        pages = {295--307},
        author = {Arnold, D.~N.~ and Winther, R.~},
        title = {Nonconforming mixed elements for elasticity},
        journal = {Mathematical Models and Methods in Applied Sciences}
}
""")
    Citations().add("Hu2015", """
@article{Hu2015,
      author = {Hu, J.~ and Zhang, S.~},
      title = {A family of conforming mixed finite elements for linear elasticity on triangular grids},
      year = {2015},
      month = jan,
      archiveprefix = {arXiv},
      eprint = {1406.7457},
}
""")
    Citations().add("Arbogast2017", """
@techreport{Arbogast2017,
  title={Direct serendipity finite elements on convex quadrilaterals},
  author={Arbogast, T and Tao, Z},
  year={2017},
  institution={Tech. Rep. ICES REPORT 17-28, Institute for Computational Engineering and Sciences}
}
""")
    Citations().add("Gopalakrishnan2024", """
@article{gopalakrishnan2024johnson,
  title={{The Johnson-Mercier elasticity element in any dimensions}},
  author={Gopalakrishnan, J and Guzman, J and Lee, J J},
  journal={arXiv preprint arXiv:2403.13189},
  year={2024}
}
""")
    Citations().add("Groselj2022", """
@article{groselj2022generalized,
  title={{Generalized C1 Clough--Tocher splines for CAGD and FEM}},
  author={Gro{\v{s}}elj, Jan and Knez, Marjeta},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={395},
  pages={114983},
  year={2022},
  publisher={Elsevier}
}
""")
    Citations().add("PowellSabin1977", """
@article{powell1977piecewise,
  title={Piecewise quadratic approximations on triangles},
  author={Powell, Michael JD and Sabin, Malcolm A},
  journal={ACM Transactions on Mathematical Software},
  volume={3},
  number={4},
  pages={316--325},
  year={1977},
  publisher={ACM New York, NY, USA}
}
""")
    Citations().add("AlfeldSorokina2016", """
@article{alfeld2016linear,
  title={Linear differential operators on bivariate spline spaces and spline vector fields},
  author={Alfeld, Peter and Sorokina, Tatyana},
  journal={BIT Numerical Mathematics},
  volume={56},
  number={1},
  pages={15--32},
  year={2016},
  publisher={Springer}
}
""")
    Citations().add("ArnoldQin1992", """
@article{arnold1992quadratic,
  title={{Quadratic velocity/linear pressure Stokes elements}},
  author={Arnold, Douglas N and Qin, Jinshui},
  journal={Advances in computer methods for partial differential equations},
  volume={7},
  pages={28--34},
  year={1992}
}
""")
    Citations().add("ChristiansenHu2019", """
@article{christiansen2019finite,
  title={A finite element for Stokes with a commuting diagram },
  author={Christiansen, Snorre H and Hu, Kaibo},
  journal={Mathematical Analysis in Fluid and Gas Dynamics},
  volume={2107},
  pages={172--183},
  year={2019}
}
""")
    Citations().add("GuzmanNeilan2018", """
@article{guzman2018infsup,
    author = {Guzm\'{a}n, Johnny and Neilan, Michael},
    title = {{Inf-Sup Stable Finite Elements on Barycentric Refinements Producing Divergence--Free Approximations in Arbitrary Dimensions}},
    journal = {SIAM Journal on Numerical Analysis},
    volume = {56},
    number = {5},
    pages = {2826-2844},
    year = {2018},
    doi = {10.1137/17M1153467}
}
""")
    Citations().add("BernardiRaugel1985", """
@article{bernardi-raugel-0,
    AUTHOR = {Bernardi, Christine and Raugel, Genevi\\`eve},
     TITLE = {Analysis of some finite elements for the {Stokes} problem},
   JOURNAL = {Mathematics of Computation},
    VOLUME = {44},
      YEAR = {1985},
       DOI = {10.1090/S0025-5718-1985-0771031-7},
     PAGES = {{71--79}}
}
""")


except ImportError:
    Citations = None


class NeedsCoordinateMappingElement(metaclass=ABCMeta):
    """Abstract class for elements that require physical information
    either to map or construct their basis functions."""
    pass


class PhysicallyMappedElement(NeedsCoordinateMappingElement):
    """A mixin that applies a "physical" transformation to tabulated
    basis functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if Citations is not None:
            Citations().register("Kirby2018zany")
            Citations().register("Kirby2019zany")

    @abstractmethod
    def basis_transformation(self, coordinate_mapping):
        """Transformation matrix for the basis functions.

        :arg coordinate_mapping: Object providing physical geometry."""
        pass

    def map_tabulation(self, tabulation, coordinate_mapping):
        assert coordinate_mapping is not None

        M = self.basis_transformation(coordinate_mapping)
        # we expect M to be sparse with O(1) nonzeros per row
        # for each row, get the column index of each nonzero entry
        csr = [[j for j in range(M.shape[1]) if not isinstance(M.array[i, j], gem.Zero)]
               for i in range(M.shape[0])]

        def matvec(table):
            # basis recombination using hand-rolled sparse-dense matrix multiplication
            table = [gem.partial_indexed(table, (j,)) for j in range(M.shape[1])]
            # the sum approach is faster than calling numpy.dot or gem.IndexSum
            expressions = [sum(M.array[i, j] * table[j] for j in js) for i, js in enumerate(csr)]
            val = gem.ListTensor(expressions)
            return gem.optimise.aggressive_unroll(val)

        return {alpha: matvec(tabulation[alpha]) for alpha in tabulation}

    def basis_evaluation(self, order, ps, entity=None, coordinate_mapping=None):
        result = super().basis_evaluation(order, ps, entity=entity)
        return self.map_tabulation(result, coordinate_mapping)

    def point_evaluation(self, order, refcoords, entity=None, coordinate_mapping=None):
        result = super().point_evaluation(order, refcoords, entity=entity)
        return self.map_tabulation(result, coordinate_mapping)


class DirectlyDefinedElement(NeedsCoordinateMappingElement):
    """Base class for directly defined elements such as direct
    serendipity that bypass a coordinate mapping."""
    pass


class PhysicalGeometry(metaclass=ABCMeta):

    @abstractmethod
    def cell_size(self):
        """The cell size at each vertex.

        :returns: A GEM expression for the cell size, shape (nvertex, ).
        """

    @abstractmethod
    def jacobian_at(self, point):
        """The jacobian of the physical coordinates at a point.

        :arg point: The point in reference space (on the cell) to
             evaluate the Jacobian.
        :returns: A GEM expression for the Jacobian, shape (gdim, tdim).
        """

    @abstractmethod
    def detJ_at(self, point):
        """The determinant of the jacobian of the physical coordinates at a point.

        :arg point: The point in reference space to evaluate the Jacobian determinant.
        :returns: A GEM expression for the Jacobian determinant.
        """

    @abstractmethod
    def reference_normals(self):
        """The (unit) reference cell normals for each facet.

        :returns: A GEM expression for the normal to each
           facet (numbered according to FIAT conventions), shape
           (nfacet, tdim).
        """

    @abstractmethod
    def physical_normals(self):
        """The (unit) physical cell normals for each facet.

        :returns: A GEM expression for the normal to each
           facet (numbered according to FIAT conventions).  These are
           all computed by a clockwise rotation of the physical
           tangents, shape (nfacet, gdim).
        """

    @abstractmethod
    def physical_tangents(self):
        """The (unit) physical cell tangents on each facet.

        :returns: A GEM expression for the tangent to each
           facet (numbered according to FIAT conventions).  These
           always point from low to high numbered local vertex, shape
           (nfacet, gdim).
        """

    @abstractmethod
    def physical_edge_lengths(self):
        """The length of each edge of the physical cell.

        :returns: A GEM expression for the length of each
           edge (numbered according to FIAT conventions), shape
           (nfacet, ).
        """

    @abstractmethod
    def physical_points(self, point_set, entity=None):
        """Maps reference element points to GEM for the physical coordinates

        :arg point_set: A point_set on the reference cell to push forward to physical space.
        :arg entity: Reference cell entity on which the point set is
                     defined (for example if it is a point set on a facet).
        :returns: a GEM expression for the physical locations of the
                  points, shape (gdim, ) with free indices of the point_set.
        """

    @abstractmethod
    def physical_vertices(self):
        """Physical locations of the cell vertices.

        :returns: a GEM expression for the physical vertices, shape
                (gdim, )."""


zero = gem.Zero()
one = gem.Literal(1.0)


def identity(*shape):
    V = numpy.eye(*shape, dtype=object)
    for multiindex in numpy.ndindex(V.shape):
        V[multiindex] = zero if V[multiindex] == 0 else one
    return V
