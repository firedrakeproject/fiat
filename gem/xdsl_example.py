from firedrake import *

import numpy as np
from petsc4py import PETSc

import tsfc

# ----- SETUP -----

# Create a 2D square with 8 triangles
mesh = UnitSquareMesh(2, 2)

# Create a continuous Lagrange degree 3 function space (1 DoF per vertex, 2 per edge and 1 per cell)
# See https://defelement.org/elements/examples/triangle-lagrange-equispaced-3.html
V_P3 = FunctionSpace(mesh, "P", 3)

# Create a Lagrange degree 1 function space (1 DoF per vertex)
# See https://defelement.org/elements/examples/triangle-lagrange-equispaced-1.html
V_P1 = FunctionSpace(mesh, "P", 1)

# Create a known function in P1
f = Function(V_P1)

# Set the values to the expression x*y
x, y = SpatialCoordinate(mesh)
f.interpolate(x*y)
# print(f.dat.data_ro)  # uncomment to see the values in f

# Now create an unknown function in P3. This means that assembling a form containing it
# will be a vector.
v = TestFunction(V_P3)

# Build symbolic expressions representing integrals (called one forms)
L_mass = f * v * dx
L_laplace = inner(grad(v), grad(f)) * dx

# ----- EVALUATE USING FIREDRAKE -----

# And assemble (i.e. numerically evaluate) them
result_mass = assemble(L_mass)
# print(result_mass.dat.data_ro)  # uncomment to see values
result_laplace = assemble(L_laplace)
# print(result_laplace.dat.data_ro)  # uncomment to see values

# ----- EVALUATE MANUALLY -----

# But let's do this manually instead, starting with the mass form.
# We'll do the outer loop over cells in Python for now.

# Compile the local kernel - this bit is your job
cellwise_kernel, = tsfc.compile_form(L_mass)
# NOTE: cellwise_kernel here isn't actually callable so the below is only pseudocode
# for now, but you can look at it by running
# print(cellwise_kernel.ast)
# or (for the C)
# import loopy as lp; print(lp.generate_code_v2(cellwise_kernel.ast).device_code())

# Get global data structures, allocating a result tensor
f_data = f.dat.data_ro
result = Cofunction(V_P3.dual())
result_data = result.dat.data

# Get the indirection maps
map_P1 = V_P1.cell_node_list
num_cells, num_nodes_per_cell_P1 = map_P1.shape
map_P3 = V_P3.cell_node_list
_, num_nodes_per_cell_P3 = map_P3.shape

# Allocate temporaries
f_temp = np.empty(num_nodes_per_cell_P1, dtype=PETSc.ScalarType)
result_temp = np.empty(num_nodes_per_cell_P3, dtype=PETSc.ScalarType)

# Now compute
for c in range(num_cells):
    # f is input to the kernel
    f_temp[...] = f_data[map_P1[c]]
    result_temp[...] = 0.0

    cellwise_kernel(result_temp, f_temp)

    result_data[map_P3[c]] += result_temp