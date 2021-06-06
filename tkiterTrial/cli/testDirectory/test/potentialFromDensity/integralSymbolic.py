from sympy import *
from mpmath import *
mp.dps = 15
mp.pretty = True
print (sqrt(9))
x,y = symbols('x y')
expr1 = integrate((x**0.5),x)
print (expr1)
#g(x,y) = sqrt(x+y)
#f = integrate(cos(x+y),x,y)
#print f
expr = integrate(((x**2+y**2)**-0.5), (x, 0,1), (y, 0, 1))
#expr = integrate(1/(sqrt(x*x+y*y)),(x,0,0.1),(y,0,0.1))
print (expr) 
