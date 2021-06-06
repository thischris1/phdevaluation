'''
Created on Feb 16, 2020

@author: chris
'''
import os, sys
import numpy as np
import Gaussian
import CorrelationFit
from symbol import except_clause
class resultDirectory(object):
    '''
    classdocs
    '''
    Ne = 0
    def __init__(self, dirName):
        '''
        Constructor
        '''
        self.potFile = ""
        self.spectrumFile =""
        self.evCorrelationFile =""
        self.vvCorrelationFile =""
        self.randomFile = ""
        self.gaussianFile=""
        self.Ne = 0
        self.Nm = 0
        self.interaction ="Unknown"
        self.home = dirName
        self.sigmax = "Unknown"
        self.Vmax = "Unknown"
        self.impCount = "Unknown"
        self.sigmay = "Unknown"
    
        for aFile in os.listdir(dirName):
            print (aFile)
            if "PotentialArray" in aFile:
                self.potFile = aFile
                continue
            if "spectrum" in aFile:
                self.spectrumFile = aFile
                continue
            if "evCorrelation_" in aFile and not "temp" in aFile:
                # ignore temp file
                self.evCorrelationFile = aFile
                continue
            if "vortexVortexCorrelation_" in aFile and not "temp" in aFile:
                self.vvCorrelationFile = aFile
                continue
            if "gaussian.par" in aFile:
                self.gaussianFile = aFile
                continue
            if "random.dat" in aFile:
                self.randomFile = aFile
                continue
        print ("Done reading")
        os.chdir(dirName)
        self.evaluate()

    
    
    def print(self): 
        print("============================== Content of " + self.home+" ===============")
        print (self.evCorrelationFile, self.vvCorrelationFile, self.potFile, self.gaussianFile, self.randomFile,self.spectrumFile)
        print ("============================== EnDE ===========================")
        
        
    def evaluate(self):
        # get ne, Nm
        # getSigma
        try:
            if self.randomFile:
                try:
                    self.impCount, self.Vmax, self.sigmax, self.sigmay = Gaussian.readRandom(self.randomFile)
                except:
                    print("Reading failed")
                    
                print (self.impCount, self.sigmax, self.sigmay, self.Vmax)
    
            # def get lcorr and sigmax,sigmay
            if self.potFile:
                self.lcorr = Gaussian.calculateAutoCorrelationFromFile(self.potFile)
                self.results2d = Gaussian.calculateAutoCorrelation2dFromFile(self.potFile)
                self.results2d = self.results2d[:2]
                self.potvariance = Gaussian.calculatePotentialVarianceFromFile(self.potFile)
            
                print (self.lcorr, self.results2d)
            else:
                self.lcorr = "UNKNOWN"
                self.results2d="UNKNOWN"
            # getSpectrum
            if self.spectrumFile:
                self.spectrum =Gaussian.readSpectrumFile(self.spectrumFile)
                print(self.spectrum)
            else:
                self.spectrum = 'UNKNOWN'
            if self.evCorrelationFile:
                try:
                    self.evMax = CorrelationFit.fitAndPlot2(self.evCorrelationFile,True,plot = False)
                except:
                    self.evMax= "UNKNOWN"
            else:
                self.evMax= "UNKNOWN"
            if self.vvCorrelationFile:
                try:
                    self.vvMax = CorrelationFit.fitAndPlot2(self.vvCorrelationFile,True,plot = False)
                except:
                    self.vvMax= "UNKNOWN"
            else:
                self.vvMax= "UNKNOWN"
            
            if self.gaussianFile:
                (self.Ne, self.Nm, self.interaction) = Gaussian.readGaussianPar(self.gaussianFile)  
        except:
            self.lcorr = "UNKNOWN"
            self.results2d="UNKNOWN"    
            self.spectrum = 'UNKNOWN'
            self.evMax= "UNKNOWN"
            self.vvMax= "UNKNOWN"
    
    def electronNumber(self):
        return (self.Ne)
            
            
    def printValues(self, resultFile):
        print (self.Ne, self.Nm, self.interaction, self.impCount, self.Vmax, self.sigmax, self.sigmay,self.lcorr, np.ravel(self.results2d),  self.evMax[0], self.vvMax[0],  self.spectrum)
        resultFile.write(str(self.Ne) + " ," +str(self.Nm)+ " ," + str(self.interaction))
        resultFile.write(str(self.impCount) + " ," + str(self.Vmax)+ " ," + str(self.sigmax) + " ," + str(self.sigmay) + " ," + str(self.lcorr)+ " ,")
        resultFile.write( str(np.ravel(self.results2d))+ " ," +  str(self.evMax[0])+ " ," + str(self.vvMax[0])+ " ," +  str(self.spectrum)+ " , " +str(self.potvariance)+ "\n")
    
                