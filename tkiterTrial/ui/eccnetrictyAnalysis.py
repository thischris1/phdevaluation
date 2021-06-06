'''
Created on Feb 2, 2020

@author: chris
'''


if __name__ == '__main__':
    pass

import Gaussian
import numpy as np 

if __name__ == '__main__':
    pass
trialMax = 50
sigmaCount  = 25
impCount = 4000
Vmax = 0.1
trials = np.arange(trialMax)
sigmas = np.linspace(0.001, 0.2, sigmaCount)
overallValues = np.empty((0,7))
averageValues = np.empty((0,7))
for sigma in sigmas:
    sigmaValues = np.empty((0,7))
    print (sigma)
    # calculate potential
    for trial in trials:
        print (sigma,trial)
       
       
        
        (pot,lcorr) = Gaussian.procedure(trial,impCount,Vmax,sigma,True)
        potSize = int(np.sqrt(len(pot)))
        potential = np.reshape(pot, (potSize, potSize))
        try:
            sigmaX, sigmaY, dataSmall = Gaussian.calculateAutoCorrelation2dFromPotentialData(potential, False)
        except:
            continue
        results = np.array([trial,sigma,lcorr,sigmaX,sigmaY,impCount,Vmax])
        overallValues = np.vstack((overallValues,results))
        sigmaValues = np.vstack((sigmaValues,results))
    # 2d fit 
    res =np.array([sigma, np.mean(sigmaValues[:,3]),np.var(sigmaValues[:,3]),np.mean(sigmaValues[:,4])
                   ,np.var(sigmaValues[:,4]),np.mean(sigmaValues[:,5]),np.var(sigmaValues[:,5])])
    averageValues = np.vstack((averageValues,res))
    # store results

print (overallValues)
np.savetxt("eccentricityResultsSmall.dat",overallValues)
np.savetxt("eccentricty_averageValues.dat", averageValues)
    
