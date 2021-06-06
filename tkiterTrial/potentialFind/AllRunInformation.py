'''
Created on Jan 15, 2021

@author: chris
'''
import potentialFind.RunInformation as inf
import potentialFind.Direction as dr
from singleton_decorator import singleton

@singleton
class AllRunInformation(object):
    '''
    classdocs
    '''
    runs = None

    def __init__(self,  optional_start = None):
        '''
        Constructor
        '''
        self.runs=[]
        if optional_start is None:
            return 
        else:
            self.runs.append(optional_start)
    def clean(self):
        if self.runs is not None:
            self.runs.clear()
    def start(self, sigma, vmax):
        startStep = inf.RunInformation(sigma,vmax)
        startStep.set_sigma_direction(dr.Direction.INIT)
        startStep.set_vmax_direction(dr.Direction.INIT)
        self.runs.append(startStep)
    def toString(self):
        if self.runs == None:
             
            return "empty"
        if len(self.runs) == 0:
            return "empty"
        retVal ="" 
        for step in self.runs:
               retVal = retVal+"\n"+ str(step.toString())
        return retVal
    def stepCount(self):
        return len(self.runs)
    
    def addStep(self,aStep):
        if len(self.runs) == 0:
            aStep.sigmaDirection = dr.Direction.INIT
            aStep.vmaxDirection = dr.Direction.INIT
        self.runs.append(aStep)
        
    def lastDirection (self):
        if self.runs is None or len(self.runs) < 2:
            retVal = dr.Direction.INIT
            return retVal
        return self.runs[-1]
    
    def hasSigmaDirectionChanged(self):
        if self.runs is None or len(self.runs) < 2:
            return False
        if self.runs[-1].sigmaDirection != self.runs[-2].sigmaDirection:
            return True
        return False
    def hasVmaxDirectionChanged(self):
        if self.runs is None or len(self.runs) < 2:
            return False
        if self.runs[-1].vmaxDirection != self.runs[-2].vmaxDirection:
            return True
        return False
        
    def getLastSigmaDirection(self):
        if self.runs is None or len(self.runs) < 2:
            return dr.Direction.INIT
        return self.runs[-1].sigmaDirection
    
    def getLastVamxDirection(self):
        if self.runs is None or len(self.runs) < 2:
            return dr.Direction.INIT
        return self.runs[-1].vmaxDirection
    
    def getTotalCountOfDirChange(self):
        sigmaChange = 0
        vmaxChange = 0
        lastSigmaDir = None
        lastvmaxDir = None
        for run in self.runs:
            
            if run.sigmaDirection == dr.Direction.INIT and run.vmaxDirection == dr.Direction.INIT:
                lastSigmaDir = run.sigmaDirection
                lastvmaxDir = run.vmaxDirection
                # am anfang
                continue
            if lastSigmaDir == dr.Direction.INIT:
                # 2. Schritt
                lastSigmaDir = run.sigmaDirection
                lastvmaxDir = run.vmaxDirection
                continue
            if lastSigmaDir != run.sigmaDirection:
                # alle anderen Schritte
                sigmaChange = sigmaChange+1
            if lastvmaxDir != run.vmaxDirection:
                vmaxChange = vmaxChange +1
            lastSigmaDir = run.sigmaDirection
            lastvmaxDir = run.vmaxDirection    
        return sigmaChange, vmaxChange
    
    def getTotalSigmaChanges(self):
        if self.runs is None or len(self.runs) == 0:
            return 0
        sc, vc = self.getTotalCountOfDirChange()
        return sc 
    def getTotalVmaxChange(self):
        if self.runs is None or len(self.runs) == 0:
            return 0
        sc, vc = self.getTotalCountOfDirChange()
        return vc