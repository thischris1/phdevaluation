import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy import interpolate
from scipy import integrate
import sys

fileName = 'density.dat'
if len(sys.argv) > 1:
    fileName = sys.argv[1]
if fileName == 'test':
    density = np.ones(10000)
else:
    data = np.loadtxt(fileName,usecols= (0,1,2), delimiter = ' ')
    print data.shape
#ax = fig.gca(projection='3d')
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

#Xb = np.arange(0.0,1.0,0.010101)
#Yb = np.arange(0.0,1.0,0.010101)


print X.shape
print "Function value"
densFunction = interpolate.interp2d(Xa,Ya,newDens, kind='cubic')
print "Start interpolation"
epsilon = 1e-06
potEpsilon1 = 0.01
size = int(1/potEpsilon1)
potEpsilon2 = 0.05
potX = np.arange(0.5,1.0,potEpsilon1)
#potY = np.arange(0.0,1.0,potEpsilon)






potXLarge = np.arange(-10.0,10.0,potEpsilon2)
potYLarge = np.arange(-10.0,10.0,potEpsilon2)
def mapCoordinateToUnitCell(x,y):

    newx = x - np.floor(x)
    newy = y - np.floor(y)
    
   # if x <= 1.0:
   #     if x >= 0.0:
   #         if y <= 1.0:
   #             if y >= 0.0:
   #                 return (x,y)
   # newx = x
   # newy = y
   # if (x > 1.0):
   #     newx = x -1.0
   # elif  x < 0.0:
   #     newx = x + 1.0
   # if (y > 1.0):
   #     newy = y -1.0
   # elif y < 0.0:
   #     newy = y + 1.0
 #   print "New position = (Old, new)"
 #   print (x,y,newx,newy)
    return (newx,newy)
        
potential = []
for x in potX:
    y = x
   # print (x,y)
    densAtX_y= densFunction(x,y)
    #print ("Density at " + str(x) + " "+str(y)+ "  " + str(densAtX_y))
    aufPunkt= np.array([x,y])
    # loop over all points (except the current one)
    localPot = 0.0
    count = 0
    for xinner in potXLarge:
        for yinner in potYLarge:
            innerpoint = np.array([xinner,yinner])
            distance = LA.norm(innerpoint - aufPunkt)
            if distance > 1e-12:
                (remapx,remapy) = mapCoordinateToUnitCell(xinner,yinner)
                offsetDens = densFunction(remapx,remapy)
               # print ("Offset density = " + str(offsetDens))
               # print ("Distance = " + str(distance))
                distSquare = distance*distance
                potContribution =  epsilon*(densAtX_y*offsetDens)/distSquare
                if x == 0.5:
                    print ("Potential contribution" + str(potContribution))
                localPot = localPot + potContribution
                #print localPot
                count = count +1
            else:
                print ("Skipping point " + str (xinner) + "  " +str(yinner)+ " at x = " + str(x)+" , " +str(y)) 
                
    print ("local potential at " + str(x) + " " + str(y) + " " + str(localPot))
    print ("Count = "+str(count))
    potential.append(float(localPot))
    localPot = 0.0
print potential
print potX
np.savetxt('potential.dat', np.column_stack((potX, potential)))
potArray = np.asarray(potential)
fig = plt.figure()

plt.plot(potX,potential)


#surf = ax.plot_surface(meshX, meshY, potReshaped, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_xlim(0.25,0.75)
#ax.set_ylim(0.25,0.75)


plt.show()
