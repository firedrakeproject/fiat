#!/usr/bin/env python

import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info < (3, 0):
    print("Python 3.0 or higher required, please upgrade.")
    sys.exit(1)

version = "2019.2.0.dev0"

url = "https://bitbucket.org/fenics-project/fiat/"
tarball = None
if 'dev' not in version:
    tarball = url + "downloads/fenics-fiat-%s.tar.gz" % version

setup(
    name="fenics-fiat",
    description="FInite element Automatic Tabulator",
    version=version,
    author="Robert C. Kirby et al.",
    author_email="fenics-dev@googlegroups.com",
    url=url,
    download_url=tarball,
    license="LGPL v3 or later",
    packages=["FIAT"],
    install_requires=[
        "setuptools", "numpy", "recursivenodes", "scipy", "sympy"
    ]
)

# FInAT:
# from distutils.core import setup
# import sys
#
# if sys.version_info < (3, 5):
#     print("Python 3.5 or higher required, please upgrade.")
#     sys.exit(1)
#
#
# setup(name="FInAT",
#       version="0.1",
#       description="FInAT Is not A Tabulator",
#       author="Imperial College London and others",
#       author_email="david.ham@imperial.ac.uk",
#       url="https://github.com/FInAT/FInAT",
#       license="MIT",
#       packages=["finat", "finat.ufl"],
#       # symengine is optional, but faster than sympy.
#       extras_require={'full': ['symengine']},
#       install_requires=[
#           'numpy>=1.16',
#           'sympy',
#       ])
