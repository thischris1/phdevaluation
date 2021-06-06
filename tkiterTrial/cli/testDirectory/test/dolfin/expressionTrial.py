
from dolfin import *
#import matplotlib as plt
# Read mesh from file and create function space
mesh = Mesh("mesh.xml.gz")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Code for C++ evaluation of conductivity
conductivity_code = """

class Conductivity : public UserExpression
{
public:

  // Create expression with 3 components
  Conductivity() : Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const uint D = cell.topological_dimension;
    const uint cell_index = cell.index;
    values[0] = (*c00)[cell_index];
    values[1] = (*c01)[cell_index];
    values[2] = (*c11)[cell_index];
  }

  // The data stored in mesh functions
  boost::shared_ptr<MeshFunction<double> > c00;
  boost::shared_ptr<MeshFunction<double> > c01;
  boost::shared_ptr<MeshFunction<double> > c11;

};
"""
#print (conductivity_code)
# Define conductivity expression and matrix
c00 = MeshFunction("double", mesh, "c00.xml.gz")
c01 = MeshFunction("double", mesh, "c01.xml.gz")
c11 = MeshFunction("double", mesh, "c11.xml.gz")

c = Expression(cppcode=conductivity_code)
c.c00 = c00
c.c01 = c01
c.c11 = c11
C = as_matrix(((c[0], c[1]), (c[1], c[2])))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
a = inner(C*grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
plot(u, interactive=True)















