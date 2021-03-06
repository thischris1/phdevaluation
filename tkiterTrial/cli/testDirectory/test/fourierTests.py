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

def sincos(x,y):
    return np.sin(x)*np.sin(y)

def gaussian(x,y):
    exponent = x*x +y*y
    value = np.exp(-1*exponent)
    return -0.4*value 




fig = plt.figure()
#ax1 = fig.gca(projection='3d')
ax1 = fig.add_subplot(211,projection='3d')
ax2 = fig.add_subplot(212,projection='3d')
#ax2 = fig.gca(projection='3d')

x = np.linspace(-5, 5, endpoint=True, num=100)
y = np.linspace(-5, 5, endpoint=True, num=100)
fourierSpacex = np.linspace(-1.0/5.0, 1.0/5.0, endpoint=True, num=100)
fourierSpacey= np.linspace(-1.0/5.0, 1.0/5.0, endpoint=True, num=100)
x, y = np.meshgrid(x, y)
z = sincos(x,y)
#z = gaussian(x,y)


fourierSpacex, fourierSpacey = np.meshgrid(fourierSpacex,fourierSpacey)

fourierTransform = np.fft.fft2(z)
surf = ax1.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=2, cstride=2)

#plt.show()
surf2 = ax2.plot_surface(fourierSpacex, fourierSpacey, fourierTransform, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=2, cstride=2)
#ax2.axes.set_xlim3d(left=0, right=1) 
#ax2.axes.set_ylim3d(bottom=0, top=1) 
plt.show()




print fourierTransform.shape


# maximum
print np.max(fourierTransform)
print np.argmax(fourierTransform)
exit()

# set maximal value to 100
fourierTransform[np.argmax(fourierTransform)] = 100.0
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
surf = ax.plot_surface(X, Y,diff , cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

surf2=ax.plot_surface(X,Y,backtransform, cmap=cm.coolwarm, linewidth=0, antialiased=False)
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
