#!/usr/bin/python3

import numpy as np
import sys, os
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
from dataFile import  DataFile
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# gnuplot indices 
NeIndex = 0  #1
NmIndex = 1 #2
interactionIndex = 2 #4 
impCountIndex = 3  #5
VmaxIndex = 4 #6
VminIndex = 5  #7
sigmaIndex = 6 #8
lcorrIndex = 7  #9
twodcorrelationx = 8 #10
twodcorrelationy = 9 #11
varIndex = 10
evMaxIndex = 11 # 12
vvMaxIndex = 12 # 13
gsIndex = 13  #14
gapStateIndex = 16  #17
twodmaxIndex = 19

#fileName="HC-5-15-4000imps-vo-sigma-gap-var.dat"
fileName = "CompleteData_5_6_imps_multipleCosigmas.dat"
#fileName  = "results_all_cluster.dat"
#fileName = "testREsults_6_18_hc_vosigma66e-5.dat"

mData = DataFile(fileName)
#data = np.genfromtxt(fileName, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    
data = mData.getDataForElectronsInteraction(6,1)
print (data[:,1].shape)

data = data[~np.isnan(data).any(axis=1)]
new_array = [tuple(row) for row in data]
uniques = np.unique(new_array)

# Nimp = column 0 sigma = columm 1 vmax = colum 2 potvariance = colum3 gap = column4 spectrum --> rest
print (data.shape)
sigmas = data[:,sigmaIndex]
vomax = np.round(data[:,sigmaIndex]*data[:,VmaxIndex],7)
vmax = data[:,2]
allSigmas = np.unique(sigmas)
allVmax = np.unique(vmax)
allVmaxSigmas = np.unique(vomax)
print (allSigmas)
print (allVmaxSigmas)
resultList =[]
for vmaxSigma in allVmaxSigmas:
    subArray = data[np.where(data[:,sigmaIndex]*data[:,VmaxIndex] == vmaxSigma)]
    subArray = subArray[~np.isnan(subArray).any(axis=1)]
    print (subArray.shape)
    varMean = np.mean(subArray[:,varIndex])
    varvar = np.var(subArray[:,varIndex])**2
    gapMean = np.mean(subArray[:,gapStateIndex]-0.21)
    gapVariance = np.var(subArray[:,gapStateIndex])**2
    bandwidthMean = np.mean(subArray[:,gsIndex]-subArray[:,gsIndex+2])
    bandwidthVar = np.var(subArray[:,gsIndex]-subArray[:,gsIndex+2])*np.var(subArray[:,gsIndex]-subArray[:,gsIndex+2])
    
    print(vmaxSigma, gapMean , gapVariance,varMean,varvar)
    tuple = [vmaxSigma, gapMean , gapVariance,varMean,varvar,bandwidthMean, bandwidthVar]
    resultList.append(tuple)


# loop over whole array, use only those values which fit
resultArr = np.asarray(resultList)
#resultArr = resultArr[~np.isnan(resultArr).any(axis=1)]
# FIt Data to polynomial
p4bandwidthw = np.poly1d(np.polyfit(resultArr[:,0], resultArr[:,5],6))
print ("Fit result Bandwidth")
print(p4bandwidthw)
fitx = np.linspace(np.amin(resultArr[:,0]),np.amax(resultArr[:,0]),15)
print (fitx)
fitybw = p4bandwidthw(fitx)
p4variance = np.poly1d(np.polyfit(resultArr[:,0], resultArr[:,3],6))
print ("Variance fit")
print (p4variance)
fityvariance = p4variance(fitx)

plt.plot(data[:,1]*data[:,2],data[:,4]-0.21,'ro')
plt.errorbar(resultArr[:,0],resultArr[:,1], yerr=resultArr[:,2], mfc='red',mec='green', ms = 20, mew=4, lw = 3 )
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('$V_{max}\sigma$')
plt.ylabel('$E_{gap}$ [enu]')
plt.show()
plt.errorbar(resultArr[:,0],resultArr[:,3], yerr=resultArr[:,4], mfc='red',mec='green', ms = 20, mew=4, lw = 3)
plt.plot(fitx,fityvariance, '--', lw = 3)
plt.xlabel('$V_{max}\sigma$')
plt.ylabel('$var(V(r))$')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.show()
plt.errorbar(resultArr[:,0],resultArr[:,5],yerr=resultArr[:,6],ms = 20, mew=4, lw = 3)
plt.plot(fitx,fitybw,'--', lw = 3)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('$V_{max}\sigma$')
plt.ylabel('Bandwidth')
plt.show()
plt.plot(data[:,3], data[:,4],'ro')
plt.xlabel('var(V(r))')
plt.ylabel('$E_{gap}$')
plt.show()
plt.plot(data[:,3],((data[:,8]-data[:,5])/(data[:,8]-data[:,7])-1), 'ro')
plt.xlabel('var(V(r))')
plt.ylabel('splitting')
plt.show()


exit()











sigmaList = ()
VmaxList = ()
VmaxSigmaList =()
for row in data:
    aSigma = row[1]
    if not aSigma in sigmaList:
        print ("neues Sigma"+str(aSigma))
        sigmaList = np.append(sigmaList,aSigma)
    avmax = row[2]
    if not avmax in VmaxList:
        VmaxList = np.append(VmaxList,avmax)
        print("neues Vmax" + str(avmax))
    vmaxSigma = np.round(aSigma*avmax,7)

    if not vmaxSigma in VmaxSigmaList:
        VmaxSigmaList = np.append(VmaxSigmaList,vmaxSigma)
        print (vmaxSigma)
print (VmaxSigmaList)
print (allVmaxSigmas)
resultArray=[]
for VmaxSigma in VmaxSigmaList:
    count = 0
    gapList =[]
    varianceList =[]
    for row in data:
        tempVsigma = np.round(row[1]*row[2],7)
        if VmaxSigma == tempVsigma:
            count =count +1
            gapList.append(row[4])
            varianceList.append(row[3])

    print (VmaxSigma,count)
    print (VmaxSigma, np.mean(np.asarray(gapList)) , np.var(np.asarray(gapList)))
plt.plot(data[:,1]*data[:,2],data[:,3],'ro')
plt.show()
