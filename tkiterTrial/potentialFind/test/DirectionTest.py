'''
Created on Jan 9, 2021

@author: chris
'''
import unittest
import potentialFind.RunInformation as info
import potentialFind
from potentialFind import InitialGuess as guess
from potentialFind  import AllRunInformation as ai
from potentialFind import Direction as direc
import numpy as np
from potentialFind.RunInformation import RunInformation
from singleton_decorator import singleton

from unittest.mock import MagicMock 
from potentialFind.InitialGuess import InitialGuess
from numpy.ma.testutils import assert_equal


class Test(unittest.TestCase):
    
    

    def testInit(self):
        direction = potentialFind.Direction
        print (direction)
        for dire in direction.Direction:
            print (dire)

        
    def testSecond(self):
        runInfo = info.RunInformation()
        assert (runInfo.get_sigma_direction() == potentialFind.Direction.Direction.INIT)
        assert (runInfo.sigmaDirection == potentialFind.Direction.Direction.INIT)
        runInfo.increaseSigma()
        assert (runInfo.sigmaDirection == potentialFind.Direction.Direction.INCREASING)
        runInfo.decreaseVmax()
        assert (runInfo.vmaxDirection == potentialFind.Direction.Direction.DECREASING)
        runInfo.lastSigma = 0.123
        assert (runInfo.lastSigma == 0.123)
        pass
    def testLookup(self):
        fileName = "/home/chris/eclipse-workspace/tkiterTrial/cli/CompleteData_5_6_imps_multipleCosigmas.dat"
        #aGuess2= InitialGuess(fileName)
        print ("testlookup")
        aGuess = guess.InitialGuess(fileName)
        #aGuess.resultFileName = fileName
        (sigma,vmax)= aGuess.lookupNearestNeighbours(19.386011926538675,  1.521352355266200368e-04, 0.0001, 1e-05,)
        assert_equal(sigma, 1.406121510658760454e-01, "wrong sigma")  
        assert vmax > 0.0
        print ("First result")
        print (20, 0.0001,sigma,vmax)
        (sigma,vmax)= aGuess.lookupNearestNeighbours(25, 2.5e-05, 0.3, 1e-05)
        assert sigma > 0.0
        assert vmax > 0.0
        print (sigma,vmax)
        (sigma,vmax)= aGuess.lookupNearestNeighbours(1.38, 1.5e-05, 0.3, 5e-06)
        assert sigma > 0.0
        assert vmax > 0.0
        print (sigma,vmax)
        (sigma,vmax) = aGuess.lookupNearestNeighbours(2.0, 1.5e-05,0.001, 5e-06)
        print (sigma,vmax)
        
    def testLoop(self):
        fileName = "../../cli/CompleteData_5_6_imps_multipleCosigmas.dat"
        aGuess = guess.InitialGuess(fileName)
        aGuess.resultFileName = fileName
        (sigma,vmax)= aGuess.lookupNearestNeighbours(20, 2.5e-05, 0.3, 1e-05)
        lvarrrange = np.linspace(0.000005, 0.00002, 10)
        lcorrRange = np.linspace(1.1,2.1,5)
        epsCorr = 0.1
        epsVar = 1e-06
        for avar in lvarrrange:
            for lcorr in lcorrRange:
                (sigma,vmax) = aGuess.lookupNearestNeighbours(lcorr, avar, epsCorr, epsVar)
                print (sigma,vmax)
    def testListOfDirections(self):
        allRuns = []
        runInfo = info.RunInformation()
        for a in range(0,10):
            runInfo = info.RunInformation()
            runInfo.lastSigma=0.1*a
            runInfo.lastVmax = 10*a            
            allRuns.append(runInfo)
        assert len(allRuns) == 10
        testObj = allRuns.pop()
        print (testObj.lastSigma)
        assert testObj.lastSigma == 0.9
        
    def testAlLRunInformation(self):
        testObj = ai.AllRunInformation()
        testObj.clean()
        assert testObj.stepCount() == 0
        assert testObj.hasSigmaDirectionChanged() == False
        testInfo =RunInformation()
        testInfo.lastSigma = 0.1
        testInfo.lastVmax = 0.01
        testInfo.sigmaDirection = potentialFind.Direction.Direction.INCREASING
        testInfo.vmaxDirection = potentialFind.Direction.Direction.INCREASING
        testObj.addStep(testInfo)
        assert testObj.stepCount() == 1
        assert testObj.hasSigmaDirectionChanged() == False
        testInfo.decreaseVmax()
        testInfo.increaseSigma()
        testObj.addStep(testInfo)
        assert testObj.stepCount() == 2
        testObj.addStep(RunInformation(0.005, 1.3e-04))
        assert testObj.stepCount() == 3
        assert testObj.hasSigmaDirectionChanged() == True
        (i,j)= testObj.getTotalCountOfDirChange()
        print (i,j)
        testInfo.increaseVmax()
        testInfo.descreaseSigma()
        testObj.addStep(testInfo)
        assert testObj.stepCount() == 4
        (i,j)= testObj.getTotalCountOfDirChange()
        print (i,j)
        testInfo.increaseVmax()
        testInfo.descreaseSigma()
        testObj.addStep(testInfo)
        assert testObj.stepCount() == 5
        (i,j)= testObj.getTotalCountOfDirChange()
        print (i,j)
        testInfo.decreaseVmax()
        testInfo.descreaseSigma()
        testObj.addStep(testInfo)
        print (testObj.toString())
        assert testObj.stepCount() == 6
        (i,j)= testObj.getTotalCountOfDirChange()
        print (i,j)
        
    def testAddCalcData(self):
        fileName = "/home/chris/eclipse-workspace/tkiterTrial/cli/CompleteData_5_6_imps_multipleCosigmas.dat"
        aGuess2= InitialGuess(fileName)
        aGuess2.addCalculatedData(0.1, 0.2, 1.0, 1e-06)
        assert  aGuess2 is not None
        assert aGuess2.calculatedData is not None
        aGuess2.addCalculatedData(0.2,0.4,2.0,2e-06)
        print (aGuess2.calculatedData.shape)
        assert aGuess2.calculatedData.shape[0] == 2
    def testPrint(self):
        testObj = ai.AllRunInformation()
        testObj.clean()
        assert testObj.stepCount() == 0
        assert testObj.hasSigmaDirectionChanged() == False
       
        testInfo0 =RunInformation()
        testInfo1 =RunInformation()
        testInfo2 =RunInformation()
        print (testObj.toString())
        assert testObj.toString() == "empty"
        testInfo0.set_last_sigma(0.1)
        testInfo0.set_last_vmax(1e-05)
        testObj.addStep(testInfo0)
        assert testObj.stepCount() == 1
        assert testObj.hasSigmaDirectionChanged() == False
        print (testObj.toString())
        testInfo1.set_last_sigma(0.05)
        testInfo1.set_sigma_direction(direc.Direction.DECREASING)
        testInfo1.set_last_vmax(2e-05)
        testInfo1.increaseVmax()
        testObj.addStep(testInfo1)
        assert testObj.stepCount() == 2
        
        print (testObj.toString())
        assert testObj.hasSigmaDirectionChanged() == True
        testInfo2.increaseSigma(0.075)
        testInfo2.decreaseVmax(1e-07)
        testObj.addStep(testInfo2)
        assert testObj.stepCount() == 3
        
        print (testObj.toString())
        assert testObj.hasSigmaDirectionChanged() == True
        assert_equal (testObj.getTotalCountOfDirChange() , (2,2))
        testInfo3 =RunInformation(0.1, 0.2)
        testInfo3.increaseSigma(1)
        testInfo3.decreaseVmax(3)
        testObj.addStep(testInfo3)
        assert testObj.hasSigmaDirectionChanged() == False
        assert testObj.hasVmaxDirectionChanged()  == False
        print (testObj.toString())
        assert_equal (testObj.getTotalCountOfDirChange() , (2,2))
    def testlongRun(self):
        testObj = ai.AllRunInformation()
        testObj.clean()
        testObj.start(1, 10)
        for i in range(10):
            testInfo = RunInformation(i, 10*i)
            testInfo.increaseSigma()
            testInfo.increaseVmax()
            testObj.addStep(testInfo)
        assert testObj.hasSigmaDirectionChanged() == False
        assert testObj.hasVmaxDirectionChanged() == False
        assert_equal(11, testObj.stepCount())
        print (testObj.toString())
        for i in range(10):
            testInfo = RunInformation(10-i, 100 - 10*i)
                        
            testInfo.decreaseSigma()
            testInfo.decreaseVmax()
            testObj.addStep(testInfo)
            if i == 0:
                assert testObj.hasSigmaDirectionChanged() == True
                assert testObj.hasVmaxDirectionChanged() == True
            else:
                assert testObj.hasVmaxDirectionChanged() == False
                assert testObj.hasSigmaDirectionChanged() == False
        print (testObj.toString())
        assert testObj.hasSigmaDirectionChanged() == False
        assert testObj.hasVmaxDirectionChanged() == False
        assert_equal(21, testObj.stepCount())
        sc, vc = testObj.getTotalCountOfDirChange()
        assert_equal(sc, 1, "Sigma changes")
        assert_equal(vc, 1, "vmax changes")
        print (testObj.toString())
        print ("Done")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testInit']
    unittest.main()