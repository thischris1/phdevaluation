'''
Created on Feb 2, 2020

@author: chris
'''

import Gaussian
import numpy as np 

if __name__ == '__main__':
    pass
trialMax = 200
sigmaCount  = 100
impCount = 5000

sigma = 0.1
trials = np.arange(0,20)
overallValues = np.empty((0,8))
try:
    (potLargeLcorr, lcorrLarge, potSmallLcorr, lcorrSmall) = Gaussian.findSmallestAndLargestCorrelationLength(0.01, 4000, sigma, trialMax)
except:
    print ("fehler")
    exit()
potSize = int(np.sqrt(len(potLargeLcorr)))
    
potLargeLcorr = np.reshape(potLargeLcorr, (potSize, potSize))
potSmallLcorr = np.reshape(potSmallLcorr, (potSize, potSize))
try:
    smallSigmaX, smallSigmaY, dataSmall = Gaussian.calculateAutoCorrelation2dFromPotentialData(potSmallLcorr, False)
    print ("Small")
    print (smallSigmaX, smallSigmaY)    
        
except:
    smallSigmaX = -10
    smallSigmaY = -10
    print ("Not Found")
try :
    largeSigmax, largesigmay, dataLarge = Gaussian.calculateAutoCorrelation2dFromPotentialData(potLargeLcorr, False)
    print (largeSigmax, largesigmay)
        
except:
    print ("Not found")
    largeSigmax = -100
    largesigmay = -100
print ("Large")

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import pylab as plt

x = np.linspace(0, 9.871, 201)
y = np.linspace(0, 9.871, 201)
x, y = np.meshgrid(x, y)

plt.subplot(2,2,1)
#plt.imshow(dataSmall.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()))
plt.contourf(dataSmall.reshape(201, 201),cmap=plt.cm.jet,extent=(x.min(), x.max(), y.min(), y.max()))
titleStringSmall = 'l_{corr} = '+str(round(lcorrSmall,2))+' \sigma_{x}= ' +str(round(smallSigmaX,2))+', \sigma_{y} = ' +str(round(smallSigmaY,2))
plt.title("$l_{corr}$ = "+str(round(lcorrSmall,2))+ " $\sigma_{x}= $" +str(round(smallSigmaX,2))+', $\sigma_{y}$ = ' +str(round(smallSigmaY,2)))
plt.subplot(2,2,2)
#plt.imshow(dataLarge.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()))
plt.contourf(dataLarge.reshape(201, 201),cmap=plt.cm.jet,extent=(x.min(), x.max(), y.min(), y.max()))
plt.title ("$l_{corr} =$" +str(round(lcorrLarge,2))+ " $\sigma_{x}= $" +str(round(largeSigmax,2))+', $\sigma_{y}$ = ' +str(round(largesigmay,2)))
plt.colorbar()
plt.tight_layout()
plt.show()
