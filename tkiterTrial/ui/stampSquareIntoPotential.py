#!/usr/bin/python3

import numpy as np
import Gaussian
import sys, os
import matplotlib.pyplot as plt
fileName = "CheckGaussianArray.dat"
outfilename = "CheckGaussianArray_cut.dat"
if len(sys.argv) > 1:
    fileName = sys.argv[1]
if len (sys.argv) > 2:
    outfilename = sys.argv[2]
potentials = np.loadtxt(fileName)
(x,y,potOrig) = Gaussian.createPotentialFromGaussianFile(fileName)
print (potentials.shape)
#subArray = data[np.where(data[:,1]*data[:,2] == vmaxSigma)]
subPotentials = np.where((np.abs((potentials[:,0]-0.5)) > 0.2))



index = 0
for aPotential in potentials:
    
    xDiff = np.abs(aPotential[0]-0.5)
    yDiff = np.abs(aPotential[1]-0.5)
    totalDistance = np.sqrt(xDiff*xDiff + yDiff*yDiff)
    print (xDiff, yDiff, totalDistance)
    if totalDistance < 0.2:
        print ("Delete",aPotential)
        potentials = np.delete(potentials,index,0)
        index = index-1
        
    print ("Line", index)
    index = index +1
print (potentials.shape)
#np.savetxt(, potentials)

out = open(outfilename,"w")
for row in potentials:
    for col in row:
        out.write(str(col))
        out.write (" \t ")
    out.write("\n")

out.close()
(x,y,pot) = Gaussian.createPotentialFromGaussianFile(outfilename)

plt.contourf(x,y,pot)
plt.plot(potentials[:,0],potentials[:,1],"ro")
plt.show()
Gaussian.calculateAutoCorrelation(pot,True)
