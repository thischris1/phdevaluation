#!/usr/bin/python3

import scipy.optimize as opt
import numpy as np
import pylab as plt
from scipy import signal
from scipy.optimize import curve_fit


def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)  + c*((y-yo)**2)))
    return g.ravel()


data  = np.loadtxt("PotentialArray.dat")
pot = data[:,2]
potSize = len(pot)
print (potSize)
potSize = int(np.sqrt(potSize))
print (potSize)
pot = np.reshape(pot,(potSize,potSize))

corr = signal.correlate2d(pot,pot).ravel()
corrSize = int(len(corr))
print ("Corr size ="+str(corrSize))

# Create x and y indices
x = np.linspace(0, 200, 201)
y = np.linspace(0, 200, 201)
x, y = np.meshgrid(x, y)


#create data
data = twoD_Gaussian((x, y), 3, 100, 100, 20, 40, 0, 10)
initial_guess = (3,100,100,20,40,0,10)

#data_noisy = data + 0.2*np.random.normal(size=data.shape)
data_noisy = corr
print (data_noisy.shape)
print (corr.shape)

popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)
#print (popt, pcov)
# plot twoD_Gaussian data generated above

data_fitted = twoD_Gaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
#ax.colorbar()
plt.show()
sigmax = np.abs(popt[3])
sigmay = np.abs(popt[4])
print (sigmax,sigmay)
