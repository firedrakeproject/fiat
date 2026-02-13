try:
    import petsctools

    cite = petsctools.cite
except ImportError:
    petsctools = None

    # dummy function to use when petsctools not available
    def cite(*args, **kwargs):
        pass

if petsctools is not None:
    petsctools.add_citation("Kirby2018zany", """
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
}""")
    petsctools.add_citation("Kirby2019zany", """
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
    petsctools.add_citation("Clough1965", """
@inproceedings{Clough1965,
  author =       {R. W. Clough, J. L. Tocher},
  title =        {Finite element stiffness matrices for analysis of plate bending},
  booktitle =    {Proc. of the First Conf. on Matrix Methods in Struct. Mech},
  year =         1965,
  pages =        {515-546},
}""")
    petsctools.add_citation("Argyris1968", """
@Article{Argyris1968,
  author =       {J. H. Argyris and I. Fried and D. W. Scharpf},
  title =        {{The TUBA family of plate elements for the matrix
                  displacement method}},
  journal =      {The Aeronautical Journal},
  year =         1968,
  volume =       72,
  pages =        {701-709},
  doi =          {10.1017/S000192400008489X}
}""")
    petsctools.add_citation("Bell1969", """
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
}""")
    petsctools.add_citation("Ciarlet1972", r"""
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
}""")
    petsctools.add_citation("Morley1971", """
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
}""")
    petsctools.add_citation("MingXu2006", """
@article{MingXu2006,
  doi = {10.1007/s00211-005-0662-x},
  title={{The Morley element for fourth order elliptic equations in any dimensions}},
  author={Ming, Wang and Xu, Jinchao},
  journal={Numerische Mathematik},
  volume={103},
  number={1},
  pages={155--169},
  year={2006},
  publisher={Springer}
}""")
    petsctools.add_citation("Mardal2002", """
@article{Mardal2002,
        doi = {10.1137/s0036142901383910},
        year = 2002,
        volume = {40},
        number = {5},
        pages = {1605--1631},
        author = {Mardal, K.-A.~ and Tai, X.-C.~ and Winther, R.~},
        title = {A robust finite element method for {Darcy--Stokes} flow},
        journal = {{SIAM} Journal on Numerical Analysis}
}""")
    petsctools.add_citation("Arnold2002", """
@article{Arnold2002,
        doi = {10.1007/s002110100348},
        year = 2002,
        volume = {92},
        number = {3},
        pages = {401--419},
        author = {Arnold, R.~N.~ and Winther, R.~},
        title = {Mixed finite elements for elasticity},
        journal = {Numerische Mathematik}
}""")
    petsctools.add_citation("Arnold2003", """
@article{arnold2003,
        doi = {10.1142/s0218202503002507},
        year = 2003,
        volume = {13},
        number = {03},
        pages = {295--307},
        author = {Arnold, D.~N.~ and Winther, R.~},
        title = {Nonconforming mixed elements for elasticity},
        journal = {Mathematical Models and Methods in Applied Sciences}
}""")
    petsctools.add_citation("Hu2015", """
@article{Hu2015,
      author = {Hu, J.~ and Zhang, S.~},
      title = {A family of conforming mixed finite elements for linear elasticity on triangular grids},
      year = {2015},
      month = jan,
      archiveprefix = {arXiv},
      eprint = {1406.7457},
}""")
    petsctools.add_citation("Arbogast2017", """
@techreport{Arbogast2017,
  title={Direct serendipity finite elements on convex quadrilaterals},
  author={Arbogast, T and Tao, Z},
  year={2017},
  institution={Tech. Rep. ICES REPORT 17-28, Institute for Computational Engineering and Sciences}
}""")
    petsctools.add_citation("Gopalakrishnan2024", """
@article{gopalakrishnan2024johnson,
  title={{The Johnson-Mercier elasticity element in any dimensions}},
  author={Gopalakrishnan, J and Guzman, J and Lee, J J},
  journal={arXiv preprint arXiv:2403.13189},
  year={2024}
}""")
    petsctools.add_citation("Groselj2022", """
@article{groselj2022generalized,
  title={{Generalized C1 Clough--Tocher splines for CAGD and FEM}},
  author={Gro{\v{s}}elj, Jan and Knez, Marjeta},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={395},
  pages={114983},
  year={2022},
  publisher={Elsevier}
}""")
    petsctools.add_citation("PowellSabin1977", """
@article{powell1977piecewise,
  title={Piecewise quadratic approximations on triangles},
  author={Powell, Michael JD and Sabin, Malcolm A},
  journal={ACM Transactions on Mathematical Software},
  volume={3},
  number={4},
  pages={316--325},
  year={1977},
  publisher={ACM New York, NY, USA}
}""")
    petsctools.add_citation("AlfeldSorokina2016", """
@article{alfeld2016linear,
  title={Linear differential operators on bivariate spline spaces and spline vector fields},
  author={Alfeld, Peter and Sorokina, Tatyana},
  journal={BIT Numerical Mathematics},
  volume={56},
  number={1},
  pages={15--32},
  year={2016},
  publisher={Springer}
}""")
    petsctools.add_citation("ArnoldQin1992", """
@article{arnold1992quadratic,
  title={{Quadratic velocity/linear pressure Stokes elements}},
  author={Arnold, Douglas N and Qin, Jinshui},
  journal={Advances in computer methods for partial differential equations},
  volume={7},
  pages={28--34},
  year={1992}
}""")
    petsctools.add_citation("ChristiansenHu2019", """
@article{christiansen2019finite,
  title={A finite element for Stokes with a commuting diagram },
  author={Christiansen, Snorre H and Hu, Kaibo},
  journal={Mathematical Analysis in Fluid and Gas Dynamics},
  volume={2107},
  pages={172--183},
  year={2019}
}""")
    petsctools.add_citation("GuzmanNeilan2018", """
@article{guzman2018infsup,
    author = {Guzm\'{a}n, Johnny and Neilan, Michael},
    title = {{Inf-Sup Stable Finite Elements on Barycentric Refinements Producing Divergence--Free Approximations in Arbitrary Dimensions}},
    journal = {SIAM Journal on Numerical Analysis},
    volume = {56},
    number = {5},
    pages = {2826-2844},
    year = {2018},
    doi = {10.1137/17M1153467}
}""")
    petsctools.add_citation("BernardiRaugel1985", """
@article{bernardi-raugel-0,
    AUTHOR = {Bernardi, Christine and Raugel, Genevi\\`eve},
     TITLE = {Analysis of some finite elements for the {Stokes} problem},
   JOURNAL = {Mathematics of Computation},
    VOLUME = {44},
      YEAR = {1985},
       DOI = {10.1090/S0025-5718-1985-0771031-7},
     PAGES = {{71--79}}
}""")
    petsctools.add_citation("Geevers2018new", """
@article{Geevers2018new,
 title={New higher-order mass-lumped tetrahedral elements for wave propagation modelling},
 author={Geevers, Sjoerd and Mulder, Wim A and van der Vegt, Jaap JW},
 journal={SIAM journal on scientific computing},
 volume={40},
 number={5},
 pages={A2830--A2857},
 year={2018},
 publisher={SIAM},
 doi={https://doi.org/10.1137/18M1175549},
}
""")
    petsctools.add_citation("Chin1999higher", """
@article{chin1999higher,
 title={Higher-order triangular and tetrahedral finite elements with mass lumping for solving the wave equation},
 author={Chin-Joe-Kong, MJS and Mulder, Wim A and Van Veldhuizen, M},
 journal={Journal of Engineering Mathematics},
 volume={35},
 number={4},
 pages={405--426},
 year={1999},
 publisher={Springer},
 doi={https://doi.org/10.1023/A:1004420829610},
}
""")
    petsctools.add_citation("Walkington2010", """
@article{Walkington2010,
  author = {Walkington, Noel J.},
  title = {{A \\$C^1\\$ Tetrahedral Finite Element without Edge Degrees of Freedom}},
  journal = {SIAM Journal on Numerical Analysis},
  volume = {52},
  number = {1},
  pages = {330-342},
  year = {2014},
  doi = {10.1137/130912013},
}""")
    petsctools.add_citation("WuXu2019", """
@article{WuXu2019,
  title={Nonconforming finite element spaces for $2m^{\\mathrm{th}}$ order partial differential equations on $\\mathbb{R}^n$ simplicial grids when $\\mathbb{m} = \\mathbb{n} + 1$},
  author={Wu, Shuonan and Xu, Jinchao},
  journal={Mathematics of Computation},
  volume={88},
  number={316},
  pages={531--551},
  year={2019}
}""")
    petsctools.add_citation("Alfeld1984", """
@article{Alfeld1984,
  title={{A bivariate C2 Clough-Tocher scheme}},
  author={Alfeld, Peter},
  journal={Computer Aided Geometric Design},
  volume={1},
  number={3},
  pages={257--267},
  year={1984},
  publisher={Elsevier}
}""")
    petsctools.add_citation("BrambleZlamal1970", """
@article{BrambleZlamal1970,
  title={Triangular elements in the finite element method},
  author={Bramble, James H and Zl{\'a}mal, Milo{\v{s}}},
  journal={Mathematics of Computation},
  volume={24},
  number={112},
  pages={809--820},
  year={1970}
}""")
    petsctools.add_citation("Xie2008", """
@article{Xie2008,
 title={{Uniformly-stable finite element methods for Darcy-Stokes-Brinkman models}},
 author = {Xiaoping Xie and Jinchao Xu and Guangri Xue},
 journal = {Journal of Computational Mathematics},
 number = {3},
 pages = {437--455},
 volume = {26},
 year = {2008}
}
}""")
