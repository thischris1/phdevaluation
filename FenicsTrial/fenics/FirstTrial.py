'''
Created on Mar 22, 2021

@author: root
'''
from dolfin import * 
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from sympy import interactive
from matplotlib.pyplot import title
aPoint =  Point(0,0)
domain = Rectangle(Point(0.0,0.0),Point(1.0,1.0))
mesh = generate_mesh(domain,20)
#plot(mesh)
#plt.show()

V = FunctionSpace(mesh, 'Lagrange', 3)

#build essential boundary conditions
def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,Constant(0.0) , u0_boundary)

#define functions
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("-1*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree=2)

B = Expression("-5.0*x[1]+0.0*x[0]",degree=2)
#define problem
a = (inner(grad(u), grad(v)) + f*u*v -B*u*v)*dx
m = u*v*dx

A = PETScMatrix() 
M = PETScMatrix()
_ = PETScVector()
L = Constant(0.)*v*dx

assemble_system(a, L, bc, A_tensor=A, b_tensor=_)
#assemble_system(m, L,bc, A_tensor=M, b_tensor=_)
assemble_system(m, L, A_tensor=M, b_tensor=_)

#create eigensolver
eigensolver = SLEPcEigenSolver(A,M)
eigensolver.parameters['spectrum'] = 'smallest magnitude'
eigensolver.parameters['tolerance'] = 1.e-15

#solve for eigenvalues
eigensolver.solve(1)
groundr,groundc,groundrx,groundcx = eigensolver.get_eigenpair(0)
u = Function(V)
u.vector()[:] = groundrx
arr = u.vector().get_local()
densArray = arr*np.conj(arr)
print (densArray.shape)
densFunc = Function(V)
densFunc.vector().set_local(densArray)
plot(u,interactive=True, title="State")
plt.show()
plot (densFunc, interactive=True, title="Density")
plt.show()
# solve poisson


#density = u*np.conj(u)
# now poisson with this density 


if __name__ == '__main__':
    pass