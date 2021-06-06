import numpy as np

import Density

class Potential():
    density = None
    fillingFactor = 0.333333
    Ne = 5
    Nm = 0
    cellSizeLo = 0.0

    def setNoElectrons(self,new_Ne):
        self.Ne = new_Ne
        self.Nm = int((round(new_Ne/self.fillingFactor)))
        self.cellSizeLo = np.sqrt(2*np.pi*self.Nm)
        
    def __init__(self, fileName):
        print ("Populate density")
        self.density = Density.Density(fileName)
        self.setNoElectrons(5)
        print ("Potential is done")

    def getPotentialatXY(self,x0,y0):
        # loop over unit cell
        xSpace = np.linspace(0.0,1.0,self.density.size)
        ySpace = np.linspace(0.0,1.0,self.density.size)
        potential = 0.0
        for x in xSpace:
            for y in ySpace:
                pot =  self.getPotentialContribution(x0,y0,x,y)
#                print (x,y,pot)
                potential = potential + self.getPotentialContribution(x0,y0,x,y)
        return potential
    def getPotentialContribution(self,x0,y0,x1,y1):
        # calculate distance
        distance = self.getDistance(x0,y0,x1,y1)
        #print ("Distance from "+ str(x0) + ", " +str(y0)+ "to " + str(x1)+" , " + str(y1) + " is " + str(distance))
        chargeAtx1y1= self.density.getChargeInArea(x1,y1)
        if distance < 1e-12:
            distance = 1.0/self.density.size*self.cellSizeLo
        potContribution = chargeAtx1y1/distance
        

        return (potContribution)
    def getDistance(self,x0,y0,x1,y1):
        distance = (x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)
        distance = np.sqrt(distance)*self.cellSizeLo
        return distance
    def getPotentialAtXYPBC(x,y):
        pot = getPotentialAtXY(x,y)
        return pot
