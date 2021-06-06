
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import rc
import cli.Gaussian as gauss
from potentialFind import InitialGuess as guess
from potentialFind  import AllRunInformation as ai

from potentialFind import Direction as direc
import sys
#from numba import njit, prange
import shutil

import sys
from potentialFind.InitialGuess import InitialGuess
from potentialFind.AllRunInformation import AllRunInformation

import potentialFind.RunInformation as info

# fileName  = "results_all_cluster.dat"
# fileName = "testREsults_6_18_hc_vosigma66e-5.dat"

fileName = "../cli/CompleteData_5_6_imps_multipleCosigmas.dat"
aGuess = guess.InitialGuess(fileName)
aGuess.resultFileName = fileName
runInfo = AllRunInformation()

def performForLoop(epsVar, epsLcorr, lcorr, variance, sigmaStart, vmaxStart, np, totalCount, results):
    found = False
    dirNumber = 1
    print ("Vmax, sigma")
    print (vmaxStart, sigmaStart)
    dirName = "./dir_"+str(dirNumber)
    impFileNameOld = dirName+"/CheckGaussianArray.dat"
    impFileNameNew = "./impurities_var"+str(variance)+"_lcorr_"+str(lcorr)
    for index in np.arange(20):
        data = gauss.procedure(1, 4800, vmaxStart, sigmaStart)
        totalCount = totalCount - 1
        print(data)
        foundLcorr = data[3]
        foundVariance = data[1]
        #inf = info.RunInformation(vmaxStart,sigmaStart) 
        #runInfo.addStep(inf)
        print(foundLcorr, foundVariance)
        if (np.abs(foundLcorr - lcorr) < epsLcorr):
            print ("Found proper lcorr")
            if np.abs(foundVariance - variance) - epsVar:
                found = True
                print ("Ich habs") 
                shutil.copyfile(impFileNameOld, impFileNameNew)
                aGuess.saveCalculatedData()
                break
        results = np.vstack((results, data))
        if totalCount < 0:
            found = True
    
    return results, found

epsVar = 1.2e-06
epsLcorr = 0.9

def recalculateSigmaVmax(oldvmax, oldSigma, targetVar, targetLcorr, epsVar, epsLcorr, results):
    varMean = np.mean(results[:,1])
    lcorrMean = np.mean(results[:,3])
    sigma = oldSigma
    vmax = oldvmax
    vmaxDir = None
    sigmaDir = None
    
    aGuess.addCalculatedData(oldSigma, oldvmax, lcorrMean, varMean)
    print ("Needed Locr, variance ", targetLcorr, lcorrMean, targetVar, varMean)
    print("Old Values, sigma, vmax",oldSigma,oldvmax)
    # ist irgendeiner der parameter ok?
    if np.abs(lcorrMean - targetLcorr) < 1.5*epsLcorr:
        print ("Leave sigma")
        sigmaDir = runInfo.getLastSigmaDirection()
    else:
        if lcorrMean < targetLcorr:
            print("Too small lcorr, groesser machen (needed)", targetLcorr, lcorrMean)
            if runInfo.getLastSigmaDirection() == direc.Direction.INIT:
                sigma = oldSigma*2.0
                 # im ersten Durchgang: Grosse Schritte
            else:
                if runInfo.getTotalSigmaChanges() == 0:     
                        
                    print ("Grosse Schritte")
                    sigma = oldSigma*2.0
                else:
                    print ("Kleine schritte")
                    eps = 1.0 + (0.5 / runInfo.getTotalSigmaChanges())
                    sigma =oldSigma*eps
            sigmaDir =  direc.Direction.INCREASING   
        else:
            sigmaDir = direc.Direction.DECREASING
            print("Too large lcorr, kleiner machen(needed)")
            if runInfo.getLastSigmaDirection() == direc.Direction.INIT:
                sigma = oldSigma *0.5
                print ("Grosse schritte")
            else:
                if runInfo.getTotalSigmaChanges()==0:
                    sigma = oldSigma *0.5
                    print ("Grosse schritte")
                else:
                    print ("Kleine Schritte")
                    eps = 1.0 - (0.5 / runInfo.getTotalSigmaChanges())
                    sigma = oldSigma *eps
            
    # mehr intelligenz hier?
    if np.abs(varMean-targetVar) < 1.5*epsVar:
        print ("Variance kommt hin")
        vmaxDir = runInfo.getLastVamxDirection()
    else:
        if varMean > targetVar:
            vmaxDir = direc.Direction.DECREASING
            print ("too large variance, vmax kleiner machen")
            if runInfo.getTotalVmaxChange() == 0:
                print ("Grosse Schritte")
                vmax = oldvmax*0.75
            else:
                print ("Kleine Schritte")
                eps = 1.0 - (0.5 / runInfo.getTotalVmaxChange())
                vmax = oldvmax*eps
        else:
            vmaxDir = direc.Direction.INCREASING
            print ("TOo small variance, vmax groesser machen")
            if runInfo.getTotalVmaxChange() == 0:
                print ("Grosse schritte")
                vmax = oldvmax*1.33333 
            else:
                eps = 1.0 + (0.5 / runInfo.getTotalVmaxChange())
                vmax = oldvmax*eps
                
                print ("kleine schritte")
    Stepinfo = info.RunInformation(sigma,vmax)
    Stepinfo.set_vmax_direction(vmaxDir)
    Stepinfo.set_sigma_direction(sigmaDir)
    print(Stepinfo.toString())
    runInfo.addStep(Stepinfo)
    print ("alles")
    print(runInfo.toString())
    print ("Step nUmmer " + str(runInfo.stepCount()))
    print ("New values, sigma, vmax",sigma,vmax)
    return sigma,vmax


if len(sys.argv) < 5:
    print ("Nicht genug info")
    sys.exit()

lcorr = float(sys.argv[2])
variance = float(sys.argv[1])
sigmaStart = float(sys.argv[3])
vmaxStart = float(sys.argv[4])

print("Looking for a potential with lcorr, variance", lcorr,variance)
allResults = np.empty((0,4))

lcorrArray = np.linspace(17, 19,5)
varArray = np.linspace(0.0001, 0.0003,10)
runInfo = ai.AllRunInformation()
for lcorr in lcorrArray:
    for variance in varArray:
# lookup file 
        
        runInfo.clean()
        (vmaxStart, sigmaStart) = aGuess.lookupNearestNeighbours(lcorr,variance, epsLcorr, 1e-06)
        print ("Initial guess")
        print (vmaxStart, sigmaStart)
        if sigmaStart < 0.075:
            sigmaStart = 0.2
         
        found = False
        totalCount = 50
        vmax = vmaxStart
        sigma = sigmaStart
        runInfo.start(sigma, vmax)
        while found == False:
            results = np.empty((0,4))
            
            results, found = performForLoop(epsVar, epsLcorr, lcorr, variance, sigma, vmax, np, totalCount, results) 
            allResults = np.vstack((allResults,results))
            sigma,vmax = recalculateSigmaVmax(vmax, sigma, variance, lcorr, epsVar, epsLcorr, results)
            
            
        print ("nach dem while!")

plt.plot(allResults[:,3], allResults[:,1],'o')
plt.plot(lcorr, variance,'s',ms=30, label='target')

plt.plot(allResults[:,3],'o',label='lcorr')
plt.show()
plt.plot(allResults[:,1],'o', label ="variance")

plt.show()
    