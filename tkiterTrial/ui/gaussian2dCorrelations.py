'''
Created on Jan 21, 2020

@author: chris
'''

import Gaussian
import numpy as np 

if __name__ == '__main__':
    pass
trialMax = 50
sigmaCount  = 100
impCount = 5000
sigmas = np.linspace(0.005, 0.15, sigmaCount)
trials = np.arange(0,20)
overallValues = np.empty((0,8))
for sigma in sigmas:
    try:
        (potLargeLcorr, lcorrLarge,potSmallLcorr, lcorrSmall) = Gaussian.findSmallestAndLargestCorrelationLength(0.01, 4000, sigma, trialMax)
    except:
        continue
    potSize = int(np.sqrt(len(potLargeLcorr)))
    
    potLargeLcorr = np.reshape(potLargeLcorr, (potSize,potSize))
    potSmallLcorr = np.reshape(potSmallLcorr, (potSize,potSize))
    try:
        smallSigmaX, smallSigmaY,dataSmall  =Gaussian.calculateAutoCorrelation2dFromPotentialData(potSmallLcorr, False)
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
    values = np.array([ impCount, sigma, lcorrLarge, lcorrSmall, largeSigmax, largesigmay, smallSigmaX,smallSigmaY])
    overallValues = np.vstack((overallValues, values))

print (overallValues)
np.savetxt("results.dat", overallValues)
    
    
