'''
Created on Jan 10, 2021

@author: chris
'''
import numpy as np
from singleton_decorator import singleton
import os



@singleton
class InitialGuess(object):
    '''
    classdocs
    '''
    resultFileName =  "../cli/CompleteData_5_6_imps_multipleCosigmas.dat"
    
    __allData = None
    __reducedData = None
    __calculatedData = None
    __theData = None
    VmaxIndex = 4  # 6
    VminIndex = 5  # 7
    sigmaIndex = 6  # 8
    lcorrIndex = 7  # 9
    twodcorrelationx = 9  # 10
    twodcorrelationy = 10  # 11
    evMaxIndex = 11  # 12
    vvMaxIndex = 12  # 13
    gsIndex = 14  # 14
    varIndex = 12
    impCountIndex = 3
    red_sigmaIndex = 0
    red_vmaxIndex = 1
    red_lcorrIndex=2
    red_varIndex = 3
    red_dataAvailableIndex = 4
    
    def addCalculatedData(self,sigma,vmax,lcorr,var):
        tuple = np.asarray([ sigma,vmax,lcorr,var, False])
        tuple = tuple[~np.isnan(tuple)]
        if self.__calculatedData is None:
            self.__calculatedData = tuple
            return
        print (self.__calculatedData.shape)
        print (tuple.shape)
        try:
            self.__calculatedData = np.vstack((self.__calculatedData, tuple))
        except:
            print("return")
        # save every 20 steps 
        rows = self.__calculatedData.shape[0]
        if rows % 20 == 0:
            np.savetxt("./calculatdData.txt", self.__calculatedData)
    
        
    def saveCalculatedData(self):
        np.savetxt("./calculatdData.txt", self.__calculatedData)
        
    def lookupNearestNeighbours(self,neededLcor, neededvar, epsCorr, epsVar):
        
        
        #if self.allData is None:
           
        print (self.__allData.shape)
        sigma = 0.0
        vmax = 0.0
        epsFloat = 1e-10
        print (self.__theData[0,self.lcorrIndex], self.__theData[0,self.varIndex])
        found = False
        if self.__calculatedData is not None:
            allData = np.vstack((self.__reducedData, self.__calculatedData))
        else:
            allData = self.__reducedData
        searchEps = epsFloat
        while not found:
         #   partArray1 = theData[np.where(np.abs(theData[:,lcorrIndex] -neededLcor)<epsCorr)]
         #   partArray1 = self.__reducedData[np.where(np.abs(self.__reducedData[:,self.red_lcorrIndex] -neededLcor)<epsCorr)]
            partArraySingle = allData[np.where(np.abs(allData[:,self.red_lcorrIndex] -neededLcor)<epsFloat)]
            if (len(partArraySingle)> 0):
                secondPart = partArraySingle[np.where(np.abs(partArraySingle[:,self.red_varIndex] - neededvar) < epsFloat)]
                if len(secondPart) > 0:
                    return secondPart[0,[self.red_sigmaIndex,self.red_vmaxIndex]]  
            partArray1 = allData[np.where(np.abs(allData[:,self.red_lcorrIndex] -neededLcor)<epsCorr)]
            
            if len(partArray1) == 0:
                epsCorr = epsCorr*2.0
                continue
            #finalarray = partArray1[np.where(np.abs(partArray1[:,varIndex] - neededvar) < epsVar)]
            finalarray = partArray1[np.where(np.abs(partArray1[:,self.red_varIndex] - neededvar) < epsVar)]
            print ("Gefunden: " + str(finalarray.shape[0]))
            if len(finalarray) == 0:
                epsVar = epsVar*2.0
                continue
            if len(finalarray)>= 5:
                epsVar = epsVar*0.9
                continue
            if len(finalarray) == 1:
                sigma = finalarray[0,self.red_sigmaIndex]
                print (finalarray[0,:])
                vmax = np.abs(finalarray[0,self.red_vmaxIndex])
                print (sigma,vmax)
                found = True
                break
            if len(finalarray) < 5:
                sigma = np.mean(finalarray[:,self.red_sigmaIndex])
                vmax = np.mean(finalarray[:,self.red_vmaxIndex])
                found = True
                break
                # mittelwert nehmen
            sigma = np.mean(finalarray[:,self.red_sigmaIndex])
            vmax = np.mean(finalarray[:,self.red_vmaxIndex])
                            
        return (sigma,vmax)


    def __init__(self, fileName=None):
        '''
        Constructor
        '''
        if  fileName is not None:
            self.resultFileName = fileName
            self.readData()
        
        if os.path.exists("./calculatdData.txt"):
            self.__calculatedData = np.loadtxt("./calculatdData.txt")
            
            print (self.__calculatedData.shape)


    def readData(self):
        print ("read data")
        self.__allData = np.genfromtxt(self.resultFileName, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
        
        self.__theData = self.__allData[self.__allData[:, self.impCountIndex] == 4800]   
        
        self.__reducedData =  self.__theData[:,[self.sigmaIndex, self.VmaxIndex,self.lcorrIndex, self.varIndex]]
        num_rows, n_cols = self.__reducedData.shape
        trueColumn = np.ones((1,num_rows))
        self.__reducedData = np.hstack((self.__reducedData,trueColumn))
    def get_result_file_name(self):
        return self.__resultFileName


    def set_result_file_name(self, value):
        self.__resultFileName = value
        self.readData()


    def del_result_file_name(self):
        del self.__resultFileName


    def get_all_data(self):
        return self.__allData


    def get_reduced_data(self):
        return self.__reducedData


    def get_calculated_data(self):
        return self.__calculatedData


    def set_all_data(self, value):
        self.__allData = value


    def set_reduced_data(self, value):
        self.__reducedData = value


    def set_calculated_data(self, value):
        self.__calculatedData = value


    def del_all_data(self):
        del self.__allData


    def del_reduced_data(self):
        del self.__reducedData


    def del_calculated_data(self):
        del self.__calculatedData

    allData = property(get_all_data, set_all_data, del_all_data, "allData's docstring")
    reducedData = property(get_reduced_data, set_reduced_data, del_reduced_data, "reducedData's docstring")
    calculatedData = property(get_calculated_data, set_calculated_data, del_calculated_data, "calculatedData's docstring")
    resultFileName = property(get_result_file_name, set_result_file_name, del_result_file_name, "resultFileName's docstring")
        