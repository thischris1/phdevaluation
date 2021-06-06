'''
Created on Apr 6, 2020

@author: chris
'''
from _operator import ne
import os
import Gaussian, CorrelationFit
import numpy as np
class ParseDirectory(object):
    '''
    classdocs
    '''
   

    def __init__(self, dirName):
        '''
        Constructor
        '''
        self.__Ne = 0
        self.dir = dirName
        self.__Nm = 0
        self.__potFile = ""
        self.__spectrumFile =""
        self.__evCorrelationFile =""
        self.__vvCorrelationFile =""
        self.__randomFile = ""
        self.__gaussianFile=""
        self.__impurityFile=""
        
        self.__interaction ="Unknown"
        self.home = dirName
        self.__sigma = "Unknown"
        self.__Vmax = "Unknown"
        self.__Vmin = "Unknown"
        self.__impCount = "Unknown"
        self.__potVariance = "Unknown"
       
        for aFile in os.listdir(dirName):
            print (aFile)
            if "impurities" in aFile:
                self.__impurityFile = aFile
                continue
            if "PotentialArray" in aFile or "PotentialArray_" in aFile:
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
    
    def evaluate(self):
        # get ne, Nm
        # getSigma
        try:
            if self.randomFile:
                try:
                    self.__impCount,  self.__sigma,self.__Vmax, self.__Vmin = Gaussian.readRandom(self.randomFile)
                except:
                    print("Reading failed")
                    
            else:
                if self.__impurityFile:
                    try:
                        self.__impCount,  self.__sigma,self.__Vmax, self.__Vmin = Gaussian.readAndParseImpurity(self.__impurityFile)
                    except:
                        print("reading faild of imp")
            # def get lcorr and sigmax,sigmay
            if self.potFile:
                self.lcorr = Gaussian.calculateAutoCorrelationFromFile(self.potFile)
                self.results2d = Gaussian.calculateAutoCorrelation2dFromFile(self.potFile)
                self.results2d = self.results2d[:2]
                self.__potVariance = Gaussian.calculatePotentialVarianceFromFile(self.potFile)
            
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
                (self.__Ne, self.__Nm, self.__interaction) = Gaussian.readGaussianPar(self.gaussianFile)  
        except:
            self.lcorr = "UNKNOWN"
            self.results2d="UNKNOWN"    
            self.spectrum = 'UNKNOWN'
            self.evMax= "UNKNOWN"
            self.vvMax= "UNKNOWN"
    
    def printValues(self, resultFile):
        print (self.__Ne, self.__Nm, self.__interaction, self.__impCount, self.__Vmax, self.__Vmin, self.__sigma,self.lcorr, np.ravel(self.results2d),  self.evMax[0], self.vvMax[0],  self.potVariance,self.spectrum)
        resultFile.write(str(self.__Ne) + " ," +str(self.__Nm)+ " ," + str(self.__interaction)+" , ")
        if self.__interaction == "COULOMB":
            resultFile.write(" 0 ")
        if self.__interaction == "SRI":
            resultFile.write(" 1 ")
            
        resultFile.write(str(self.__impCount) + " ," + str(self.__Vmax)+ " ," + str(self.__Vmin) + " ," + str(self.__sigma) + " ," + str(self.lcorr)+ " ,")
        resultFile.write( str(np.ravel(self.results2d))+ " ," +  str(self.evMax[0])+ " ," + str(self.vvMax[0])+ " ," +  str(self.__potVariance)+" , "+str(self.spectrum)+ "\n")
 
    def printHeadLine(self,resultFile):
        
        resultFile.write ("Ne Nm interaction impCount Vmax, Vmin, sigma , lcorr, 2dcorrelation, evMax, vvMax, potentialVariance, spectrum \n")
   
    
    ######################## PROPERTIES
    def get_pot_file(self):
        return self.__potFile


    def get_spectrum_file(self):
        return self.__spectrumFile


    def get_ev_correlation_file(self):
        return self.__evCorrelationFile


    def get_vv_correlation_file(self):
        return self.__vvCorrelationFile


    def get_random_file(self):
        return self.__randomFile


    def get_gaussian_file(self):
        return self.__gaussianFile


    def get_interaction(self):
        return self.__interaction


    def get_sigma(self):
        return self.__sigma


    def get_vmax(self):
        return self.__Vmax

    def get_vmin(self): 
        return self.__Vmin
    
    def get_imp_count(self):
        return self.__impCount


    def get_sigmay(self):
        return self.__sigmay


    def set_pot_file(self, value):
        self.__potFile = value


    def set_spectrum_file(self, value):
        self.__spectrumFile = value


    def set_ev_correlation_file(self, value):
        self.__evCorrelationFile = value


    def set_vv_correlation_file(self, value):
        self.__vvCorrelationFile = value


    def set_random_file(self, value):
        self.__randomFile = value


    def set_gaussian_file(self, value):
        self.__gaussianFile = value


    def set_interaction(self, value):
        self.__interaction = value


    def set_sigma(self, value):
        self.__sigma = value


    def set_vmax(self, value):
        self.__Vmax = value
    
    def set_vmin(self, value):
        self.__Vmin = value


    def set_imp_count(self, value):
        self.__impCount = value




    def del_pot_file(self):
        del self.__potFile


    def del_spectrum_file(self):
        del self.__spectrumFile


    def del_ev_correlation_file(self):
        del self.__evCorrelationFile


    def del_vv_correlation_file(self):
        del self.__vvCorrelationFile


    def del_random_file(self):
        del self.__randomFile


    def del_gaussian_file(self):
        del self.__gaussianFile


    def del_interaction(self):
        del self.__interaction


    def del_sigma(self):
        del self.__sigma


    def del_vmax(self):
        del self.__Vmax
        
    def del_vmin(self):
        del self.__Vmin


    def del_imp_count(self):
        del self.__impCount


    def del_sigmay(self):
        del self.__sigmay


    def get_nm(self):
        return self.__Nm


    def set_nm(self, value):
        self.__Nm = value


    def del_nm(self):
        del self.__Nm

        
    def __getNe(self):
        return self.__Ne
    def __setNe(self, ne):
        self.__Ne = ne
    
    def get_pot_variance(self):
        return self.__potVariance
    def del_pot_variance(self):
        del self.__potVariance
    
    
    def set_pot_variance(self, value):
        self._potVariance = value
            
    ne=property(__getNe,__setNe)   
    Nm = property(get_nm, set_nm, del_nm, "Nm's docstring")
    potFile = property(get_pot_file, set_pot_file, del_pot_file, "potFile's docstring")
    potVariance = property(get_pot_variance, set_pot_variance, del_pot_variance, "potVariance docstring")
    spectrumFile = property(get_spectrum_file, set_spectrum_file, del_spectrum_file, "spectrumFile's docstring")
    evCorrelationFile = property(get_ev_correlation_file, set_ev_correlation_file, del_ev_correlation_file, "evCorrelationFile's docstring")
    vvCorrelationFile = property(get_vv_correlation_file, set_vv_correlation_file, del_vv_correlation_file, "vvCorrelationFile's docstring")
    randomFile = property(get_random_file, set_random_file, del_random_file, "randomFile's docstring")
    gaussianFile = property(get_gaussian_file, set_gaussian_file, del_gaussian_file, "gaussianFile's docstring")
    interaction = property(get_interaction, set_interaction, del_interaction, "interaction's docstring")
    sigma = property(get_sigma, set_sigma, del_sigma, "sigma's docstring")
    Vmax = property(get_vmax, set_vmax, del_vmax, "Vmax's docstring")
    Vmin = property(get_vmin, set_vmin, del_vmin, "Vmin's docstring")
    impCount = property(get_imp_count, set_imp_count, del_imp_count, "impCount's docstring")
    
