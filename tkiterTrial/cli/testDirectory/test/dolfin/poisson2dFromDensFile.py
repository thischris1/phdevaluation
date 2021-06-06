from fenics import *
import matplotlib.pyplot as plt
import numpy as np
#import DensityFromFileClass


class DensityFromFileClass(UserExpression):
    data =np.array([])
    fileName= ""
    size = 0
    def readData(self,fileName):

        print ("Do something here")
        filedata = np.loadtxt(fileName,usecols= (0,1,2), delimiter = ' ')
        density = filedata[:,2]
        self.size = int(np.sqrt(density.size))
        
        self.data = density.reshape(self.size,self.size)

        return (0)
    def eval(self, value, x):
        if self.fileName == "":
            self.readData("density.dat")

        xval = int(x[0])
        yval = int(x[1])
        value[0] = self.data[xval,yval]
    def value_shape(self):
        return (1,)

def boundary(x):
     return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Create mesh and define function space
mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, 'Lagrange', 1)
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)
# Define boundary condition
#u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
#u_D = Expression('-0.25*exp(-1.*(x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))', degree=10)
u = TrialFunction(V)
v = TestFunction(V)
f = DensityFromFileClass()
#g = Expression('sin(5*x[0])',degree=2)
u_D = Expression("1+x[0]*x[0]+2*x[1]*x[1]", degree=2)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
#f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
L = -0.1*f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
plt.show()
