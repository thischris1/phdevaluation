'''
Created on Jan 9, 2021

@author: chris
'''
import potentialFind.Direction as d 
class RunInformation(object):
    '''
    classdocs
    '''
    __Sigma = 0.0
    __Vmax = 0.0
   
    __sigmaDirection = d.Direction.INIT
    __vmaxDirection = d.Direction.INIT 
    def __init__(self,sigma =None, Vmax = None):
        
        if sigma is not None:
            self.__Sigma = sigma
        if Vmax is not None:
            self.__Vmax = Vmax 
              
    def increaseVmax(self, n_vmax = None):
        self.vmaxDirection = d.Direction.INCREASING
        if n_vmax is not None:
            __Vmax = n_vmax
        
    def decreaseVmax(self, n_vmax=None):
        self.vmaxDirection = d.Direction.DECREASING
        if n_vmax is not None:
            __Vmax = n_vmax
    def increaseSigma(self, n_sigma = None):
        self.sigmaDirection = d.Direction.INCREASING
        if n_sigma is not None:
            __Sigma = n_sigma
    def decreaseSigma(self, n_sigma = None):
        self.sigmaDirection = d.Direction.DECREASING
        if n_sigma is not None:
            __Sigma = n_sigma
    def get_last_sigma(self):
        return self.__Sigma
    def toString(self):
        retVal = "Sigma " + str(self.__Sigma) +" Direction = "+ str(self.__sigmaDirection)+" vmax = " + str(self.lastVmax) +" Direction = "+ str(self.get_vmax_direction())
        return retVal
    def get_last_vmax(self):
        return self.__Vmax


    def get_sigma_direction(self):
        return self.__sigmaDirection


    def get_vmax_direction(self):
        return self.__vmaxDirection


    def set_last_sigma(self, value):
        self.__Sigma = value


    def set_last_vmax(self, value):
        self.__Vmax = value


    def set_sigma_direction(self, value):
        self.__sigmaDirection = value


    def set_vmax_direction(self, value):
        self.__vmaxDirection = value


    def del_last_sigma(self):
        del self.__Sigma


    def del_last_vmax(self):
        del self.__Vmax


    def del_sigma_direction(self):
        del self.__sigmaDirection


    def del_vmax_direction(self):
        del self.__vmaxDirection

    lastSigma = property(get_last_sigma, set_last_sigma, del_last_sigma, "lastSigma's docstring")
    lastVmax = property(get_last_vmax, set_last_vmax, del_last_vmax, "lastVmax's docstring")
    sigmaDirection = property(get_sigma_direction, set_sigma_direction, del_sigma_direction, "sigmaDirection's docstring")
    vmaxDirection = property(get_vmax_direction, set_vmax_direction, del_vmax_direction, "vmaxDirection's docstring")
    
        