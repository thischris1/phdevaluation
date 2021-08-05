#!/usr/bin/python

from numpy.random import uniform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.optimize import curve_fit
import pylab as p
import sys
from scipy.interpolate import interp1d
import os.path 
files = []

fitStart = -1
fitEnd = 1000
max = -10
def aPower(x,a,b,c):
    retVal = c*np.power(x,a)+b
    return retVal

def quadraticOnly(x,a,b):
    retVal = a*x*x+b
    return retVal


def checkFitPoints(fitStart, fitEnd,maxPos):
    if (fitEnd - fitStart > 3):
        #passt
        return fitStart, fitEnd
    else:    
        # zu dicht beieinander
        print ("Fix fitintervall")
        if fitEnd < 3:
            fitEnd = maxPos -1
            fitStart = 2
            return fitStart, fitEnd


def fitAndPlot2(fileName, normalize, leftValinit = 0,plot =False):
    
    print (fileName)
    
    corrFile=np.loadtxt(fileName)
    
  
    print ("First in corrfile x",corrFile[0,:])
    r = corrFile[:,0]
    evCorr = corrFile[:,1]
    r = r[:400]
    evCorrTemp = evCorr[:400]
    if np.argmax(evCorrTemp) < 1.5:
        evCorr = np.delete(evCorrTemp,[0])
        r = np.delete(r,[0]) 
        print ("Delete an element")
    else:
        evCorr = evCorrTemp
    
    
    if normalize == True:
        r = r+1
        evCorr = evCorr/r
    
    maxPos = np.argmax(evCorr[leftValinit:])
    maxCorrVal = np.amax(evCorr[leftValinit:])
    maxCorrVal = evCorr[maxPos]
    print ("MAximum value is " + str(maxCorrVal) + " at " + str(maxPos))
    print (maxPos)
    leftVal = maxPos
    rightVal = maxPos
    # look for threshold value
    threshold = maxCorrVal*0.75
    # erst nach links
    for index in  range(maxPos):
        print (str(maxPos-index))
        if evCorr[maxPos-index] > threshold:
            continue
        else:
            leftVal = maxPos-index
            break
    
    print ("Left end is " + str(leftVal))
     # now for right... 
    for index in range(3*maxPos):
        try:
            if evCorr[maxPos+index] > threshold:
                continue
        except IndexError:
            print ("Width = unknown")
            rightVal = maxPos+20
        else:
            rightVal = maxPos+index
            break
    
    print ("Right value is " +str(rightVal))
        
    width = rightVal -leftVal
    print ("Width = " + str(width))
    if plot is True:
        plt.plot(r,evCorr)
        plt.show()
    return (maxPos, width)
