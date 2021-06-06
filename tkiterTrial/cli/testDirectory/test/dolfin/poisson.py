from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import expression

class MyExpression0(Expression):
    def eval(self, value, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        value[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)
        value[1] = 250.0*exp(-(dx*dx + dy*dy)/0.01)
  

f0 = MyExpression0()



mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "Lagrange", 1)

u0 = Function(V)
bc = DirichletBC(V, u0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",
               degree=1)
g = Expression("sin(5*x[0])", degree=1)
a = inner(grad(u), grad(v))*dx()
L = f*v*dx() + g*v*ds()

u = Function(V)
M= u*dx()
tol = 1.e-05
problem = LinearVariationalProblem(a, L, u, bc)
solver = AdaptiveLinearVariationalSolver(problem, M)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"
solver.solve(tol)

solver.summary()

# Plot solution(s)
plot(u.root_node(), title="Solution on initial mesh")
plot(u.leaf_node(), title="Solution on final mesh")
plt.show()
print ("done")
