'''
Created on Feb 18, 2020

@author: chris
'''
import unittest
import Gaussian
import Spectrum
import numpy as np
import os

def arrayToString(aArray):
    retVal = ""
    for val in aArray:
        retVal = retVal + " " + str(val)
    return retVal
                                
class Test(unittest.TestCase):


    def testName(self):
        vosigmaArary = [1e-05*4000]
        impArray = [20000,30000]
        facArray=[0.95,1.0,1.05]
        trialmax = 30
        resFile = open('resultFileTest_NimpDiff.dat','a')
        sigma = 0.05
        dirCount = 0
        for nimpLoop in impArray:
            for fac in facArray:
                nimp = int(nimpLoop*fac)
                Vmax = 1e-05*4000 /(nimp*sigma)
                for index in range(0,trialmax):
                	x,y, pot, spectrum = Gaussian.createPotentialFromGaussian(dirCount,nimp,sigma,Vmax,'f')
                	potVar = np.var(pot)
                	dirCount = dirCount+1
                	gap = Spectrum.calculateGap(spectrum)
                	specString = arrayToString(spectrum)
                	os.chdir('..')
                	print (nimp, sigma,Vmax,potVar,gap,specString)
                	resFile.write(str(nimp)+" "+str(sigma)+ " "+str(Vmax)+" "+str(potVar)+" " +str(gap) + " " +specString+"\n")
        resFile.close()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
