'''
Created on Oct 12, 2020

@author: chris
'''
import numpy as np
from xdiagnose.xorglog import loadfile

class DataFile(object):
    '''
    classdocs
    '''
    __fileName = "CompleteData_5_6_imps_multipleCosigmas.dat"
    __data = None 
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
    evMaxIndex = 10 # 12
    vvMaxIndex = 12 # 13
    gsIndex = 13  #14
    gapStateIndex = 13  #17
    twodmaxIndex = 18
    def __init__(self, params):
        '''
        Constructor
        '''
        if not params:
            # load file 
            print("Use default path")
        else:
            self.__fileName = params
        self.loadFile()
    def get_file_name(self):
        return self.__fileName


    def get_data(self):
        return self.__data


    def set_file_name(self, value):
        self.__fileName = value


    def set_data(self, value):
        self.__data = value


    def del_file_name(self):
        del self.__fileName


    def del_data(self):
        del self.__data

    
    
    def loadFile(self):
        self.__data = np.genfromtxt(self.__fileName, usecols=(0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    

        print (self.__data[:,1].shape)   
        extraColumn = self.addMaxtwoColumn()
        self.__data = np.append(self.__data,extraColumn,axis=1)
        print (self.__data[0,:])
    def addMaxtwoColumn(self):
        maxArray =[]
        for aData in self.__data:
        
        
            if aData[self.twodcorrelationx]> aData[self.twodcorrelationy]:
                maxArray = np.append(maxArray,aData[self.twodcorrelationx])
            else:
                maxArray= np.append(maxArray,aData[self.twodcorrelationy])
        maxArray = np.expand_dims(maxArray, -1)
        print (maxArray.shape)
        
        return (maxArray)
    def getDataForElectrons(self, Ne):
        if Ne == 5:
            return self.__data[self.__data[:,self.impCountIndex]== 4000]
        
        if Ne == 6: 
            return self.__data[self.__data[:,self.impCountIndex]== 4800]
    def getDataForInteraction(self,ia):
        return (self.__data[self.__data[:,self.interactionIndex] == ia])
    
    def filterDataForInteraction(self,indata,ia):
        return (indata[indata[:,self.interactionIndex] == ia])
    
    def getDataForElectronsInteraction(self,Ne,ia):
        tempArray = self.getDataForElectrons(Ne)
        return self.filterDataForInteraction(tempArray, ia)
    fileName = property(get_file_name, set_file_name, del_file_name, "fileName's docstring")
    
    data = property(get_data, set_data, del_data, "data's docstring")
