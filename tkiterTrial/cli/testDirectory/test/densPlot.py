import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy import interpolate
from scipy import integrate 
data = np.loadtxt("density.dat",usecols= (0,1,2), delimiter = ' ')
print data.shape




fig = plt.figure()
ax = fig.gca(projection='3d')
density = data[:,2]


print ("Maximum"       + str(np.max(density)))
print density.shape
newSum = 0.0
for val in density:
    newSum = newSum + (val*0.01*0.01)

print "Summe2"
print newSum
newDens = density.reshape(100,100)

print newDens.shape
#fourier = np.fft.fft2(newDens)
# Make data.
Xa = np.arange(0.0, 1.0, 0.010101)
Ya = np.arange(0.0, 1.0, 0.010101)
#Xa = data[:,0]
#Ya = data[:,1]
X, Y = np.meshgrid(Xa, Ya)

Xb = np.arange(0.0,1.0,0.010101)
Yb = np.arange(0.0,1.0,0.010101)

meshX, meshY = np.meshgrid(Xb,Yb)
print X.shape
print "Function value"
densFunction = interpolate.interp2d(Xa,Ya,newDens, kind='cubic')
print "Start interpolation"
# Plot the surface.
#densNewVals = densFunction(Xa,Ya)
densNewVals = densFunction(Xb,Yb)
sum = 0.0
sum1 = 0.0
Xaa = np.arange(0.0,1.0,0.05)

for xval in Xaa:
    for yval in Xaa:
        #        print xval
        sum = sum + densFunction(xval,yval)*0.05*0.05
        
print " Summe = " + str(sum)
testDens = densFunction(0.5,0.5)
print testDens
fourierDens = np.fft.fft2(densNewVals)
print "Fourier"
print fourierDens.shape
integral = integrate.nquad(densFunction,[[0,1.0],[0,1.0]])
print ("Integral")
print integral

print newDens.shape
print X.shape
print Y.shape

surf = ax.plot_surface(X, Y, newDens, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#surf = ax.plot_surface(meshX, meshY, fourierDens, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
#ax.set_xlim(-0.1,0.1)
#ax.set_ylim(-0.1,0.1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()

#surf2 = ax.plot_surface(X,Y, np.absolute(fourier), cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
plt.show()

# calculate potential

potX = np.arange(0.0,1.0,0.1)
potY = np.arange(0.0,1.0,0.1)

for x in potX:
    for y in potY:
        print (x,y)
        print densFunction(x,y)
