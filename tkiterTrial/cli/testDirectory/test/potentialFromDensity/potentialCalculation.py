#/usr/bin/python
import Potential
import sys
import os
import numpy as np
import matplotlib.pyplot  as plt 
fileName = "testDensity.dat"
print "Start"
if len(sys.argv) > 1:
    fileName = sys.argv[1]

myPotential = Potential.Potential(fileName)
size = myPotential.density.size
potList = np.array([])
potListPBC = np.array([])
xSpace = np.linspace(0.5,1.0, 20)
ySpace = np.linspace(0.5,1.0,20)
for x in xSpace:
    pot = myPotential.getPotentialatXY(x,x)
    print x,x,pot
    potPBC = myPotential.getPotentialAtXYPBC(x,x)
    potList = np.append(potList,pot)
    potListPBC = np.append(potListPBC,potPBC)
xSpace = xSpace*myPotential.cellSizeLo
plt.plot(xSpace,potList, 'ro', label =' ohne PBC')


plt.plot(xSpace,potListPBC, label = 'PBC')
plt.legend()
plt.show()
