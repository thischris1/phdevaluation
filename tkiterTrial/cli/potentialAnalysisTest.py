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
        vosigmaArary = np.linspace(1e-06, 8e-06, 8)
        trialmax = 30
        resFile = open('resultFileTest_coulomb.dat', 'a')
        sigmaArray = np.linspace(0.001, 0.3, 10)
        dirCount = 0
        for vosigma in vosigmaArary:
            for sigma in sigmaArray:
                Vmax = vosigma / sigma
                for index in range(0, trialmax):
                    x, y, pot, spectrum = Gaussian.createPotentialFromGaussian(dirCount, 4000, sigma, Vmax, 'f', 'Coulomb')
                    potVar = np.var(pot)
                    dirCount = dirCount + 1
                    gap = Spectrum.calculateGap(spectrum)
                    specString = arrayToString(spectrum)
                    os.chdir('..')
                    print (4000, sigma, Vmax, potVar, gap, specString)
                    resFile.write("4000 " + str(sigma) + " " + str(Vmax) + " " + str(potVar) + " " + str(gap) + " " + specString + "\n")
        resFile.close()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
