'''
Created on Feb 18, 2020

@author: chris
'''
import unittest
import Gaussian
import Spectrum
import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


fileName = "HC5-15-Nimp-testData.dat"
data= np.loadtxt(fileName)                         
vosigmaArary = [1e-05*4000]
impArray = [100, 1000,2000,3000,4000,5000,6000,10000,20000,30000]
facArray=[0.95,1.0,1.05]
trialmax = 30

sigma = 0.05
dirCount = 0
allArray=[]
allgapList=[]
for nimpLoop in impArray:
    gapArray=[]
    for fac in facArray:
        nimp = int(nimpLoop*fac)
        subArray = data[np.where(data[:,0]==nimp)]
       
        gapAverage = np.mean(subArray[:,5])
        #print (nimp,gapAverage)
        gapArray.append((nimp,gapAverage))
        allArray.append((nimp,gapAverage))
        allgapList.append((nimp,gapAverage))
    gaps = np.array(gapArray)
    #print ("gaps ", gaps)
    diffgap = np.diff(gaps[:,1])
    diffImps = np.diff(gaps[:,0])
    print (nimpLoop,diffgap,diffImps, diffgap/diffImps)
    print (np.mean(np.abs(diffgap)/np.abs(diffImps)))

allgapArray = np.array(allgapList)
plt.plot(data[:,0],data[:,5], "ro")
plt.plot(allgapArray[:,0],allgapArray[:,1], "b+", markersize=20)
plt.show()
#plt.plot(

