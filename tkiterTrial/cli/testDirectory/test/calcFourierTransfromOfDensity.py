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
data = np.loadtxt("density.dat",usecols= (0,1,2), delimiter = ' ')
print data.shape





fig = plt.figure()
ax1 =  fig.add_subplot(211,projection='3d')
ax2 =   fig.add_subplot(212,projection='3d')
density = data[:,2]


print ("Maximum"       + str(np.max(density)))
print density.shape
newSum = 0.0
for val in density:
    newSum = newSum + (val*0.01*0.01)

print "Summe2"
print newSum
newDim = int(np.sqrt(density.shape))
print "Newdimension = " + str(newDim)
#newDens = density.reshape(100,100)
newDens = density.reshape(newDim,newDim)
Xa = np.arange(0.0, 1.0, 1.0/newDim)
Ya = np.arange(0.0, 1.0, 1.0/newDim)
#Xa = data[:,0]
#Ya = data[:,1]

X, Y = np.meshgrid(Xa, Ya)
print newDens.shape
# fourier transformation

fourierTransform = np.fft.fftshift(np.fft.fft2(newDens))

print fourierTransform.shape
fourierSpacex = np.linspace(-1.0, 1.0, endpoint=True, num=50)
fourierSpacey= np.linspace(-1.0, 1.0, endpoint=True, num=50)
fourierSpacex, fourierSpacey = np.meshgrid(fourierSpacex,fourierSpacey)

surf1 = ax1.plot_surface(X,Y,newDens, linewidth=0, antialiased=True, rstride=2, cstride=2)

surf2 = ax2.plot_surface(fourierSpacex, fourierSpacey,fourierTransform , cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

# maximum
print np.max(fourierTransform)
print np.argmax(fourierTransform)
# set maximal value to 100
#fourierTransform[np.argmax(fourierTransform)] = 100.0
#print ("Values bigger than 0.0 = ", fourierTransform[fourierTransform  > 0.1])
#print ("Indices = ", np.nonzero(fourierTransform  > 1.0))
Xa = np.arange(0.0, 1.0, 1.0/newDim)
Ya = np.arange(0.0, 1.0, 1.0/newDim)
#Xa = data[:,0]
#Ya = data[:,1]
backtransform = np.fft.ifft2(fourierTransform)
X, Y = np.meshgrid(Xa, Ya)
diff = newDens - backtransform
print X.shape
surf = ax1.plot_surface(X, Y,diff , cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

surf2=ax2.plot_surface(X,Y,backtransform, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
exit()
#fourier = np.fft.fft2(newDens)
# Make data.


Xb = np.arange(0.0,1.0,0.010101)
Yb = np.arange(0.0,1.0,0.010101)


print X.shape
print "Function value"
densFunction = interpolate.interp2d(Xa,Ya,newDens, kind='cubic')
print "Start interpolation"
epsilon = 1e-06
potEpsilon = 0.1
size = int(1/potEpsilon)
potX = np.arange(0.0,1.0,potEpsilon)
potY = np.arange(0.0,1.0,potEpsilon)

cell1 = np.array([1.0,1.0])
cell7 = np.array([1.0,0.0])
cell5 = np.array([1.0,-1.0])
cell8 = np.array([0.0,-1.0])
cell6 = np.array([-1.0,-1.0])
cell2 = np.array([-1.0,0.0])
cell3 = np.array([-1.0,1.0])
cell4 = np.array([0.0,1.0])




potXLarge = np.arange(-5.0,5.0,potEpsilon)
potYLarge = np.arange(-5.0,5.0,potEpsilon)
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
    print "New position = (Old, new)"
    print (x,y,newx,newy)
    return (newx,newy)
        
potential = []
for x in potX:
    for y in potY:
        print (x,y)
        densAtX_y= densFunction(x,y)
        print densAtX_y
        aufPunkt= np.array([x,y])
        # loop over all points (except the current one)
        localPot = 0.0
        for xinner in potXLarge:
            for yinner in potYLarge:
                innerpoint = np.array([xinner,yinner])
                distance = LA.norm(innerpoint - aufPunkt)
                if distance > 1e-05:
                    (remapx,remapy) = mapCoordinateToUnitCell(xinner,yinner)
                    offsetDens = densFunction(remapx,remapy)
                    print ("Distance = " + str(distance))
                    distSquare = distance*distance
                    potContribution =  epsilon*(densAtX_y*offsetDens)/distSquare
                    print ("Potential contribution" + str(potContribution))
                    localPot = localPot + potContribution
                    print localPot
        print ("local potential at " + str(x) + " " + str(y) + " " + str(localPot))
        
        potential.append(localPot)
        localPot = 0.0
#print potential
potArray = np.asarray(potential)
print potArray.shape
meshX, meshY = np.meshgrid(potX,potY)
potReshaped = potArray.reshape(size,size)
surf = ax.plot_surface(meshX, meshY, potReshaped, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_xlim(0.25,0.75)
#ax.set_ylim(0.25,0.75)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show()
