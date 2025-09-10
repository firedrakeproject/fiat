from firedrake import *

mesh = UnitSquareMesh(1,1)
cg1_space = FunctionSpace(mesh, "CG", 1)
hdiv_space = FunctionSpace(mesh, "RT", 1)
hcurl_space = FunctionSpace(mesh, "N1curl", 3)
arg_space = FunctionSpace(mesh, "ARG", 5)

@pytest.fixture(params=(dim for dim in range(1, 4)))
def mesh(request):
    dim, extruded = request.param
    return UnitSquareMesh(dim,dim)


@pytest.fixture(params=("lagrange", "raviart", "nedelec", "argyris", "sin"))
def testcase(request, mesh):
    case = request.param
    cg1_space = 
    x = SpatialCoordinate(mesh)

    f = Function(cg1_space)
    f.assign(1)

    g = Function(cg1_space)
    g.interpolate(x[0] + 4*x[1])
    v = TestFunction(hdiv_space)
    form = dot(v,grad(g))*dx
    #form = v*dx
    form_a = assemble(form)
    if case == "lagrange":
        

def gpu_kernel(mesh, space, form):
    with device("gpu") as compute_device:
        # For checking our work
        x = SpatialCoordinate(mesh)
        f = Function(cg1_space)
        g = Function(cg1_space)
        f.assign(1)
        g.interpolate(x[0] + 4*x[1])
        v = TestFunction(hdiv_space)
        form = dot(v,grad(g))*dx
        #form = v*dx
        form_a = assemble(form)
        import os
        if len(compute_device.kernel_string) == 0:
            raise ValueError("Missing kernel, please run firedrake-clean")
        with open("temp_kernel_grad.py",'w') as file:
            file.write("import cupy as cp\n")
            for i, kernel in enumerate(compute_device.kernel_string):
                file.write(kernel.replace("cupy_kernel", f"cupy_kernel{i}") + "\n")
