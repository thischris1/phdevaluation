import numpy as np


class Density():
    fileName = ""
    data = np.array([])
    size = 0
    epsilon = 0.0

    def __init__(self,fileName):
        filedata = np.loadtxt(fileName,usecols= (0,1,2), delimiter = ' ')
        print ("Loading data from "+ fileName)
        density = filedata[:,2]
        self.size = int(np.sqrt(density.size))
        
        self.data = density.reshape(self.size,self.size)
        self.epsilon = 1.0/(self.size-1)

# access methods
    def getDensityAtXY(self, x,y):
        # distinguish types
        # if int
        if type(x) == int:
            return self.data[x,y]
        else:
        # if float umrechnen
            reducedX = x- int(x)
            reducedY = y - int(y)
            newx,newy = self.xyFromCoordtoArray(reducedX,reducedY)
            return self.data[newx,newy]
            
    def xyFromCoordtoArray(self, x,y):
        newX = np.abs(int (round(x * (self.size -1))))
        newY = np.abs(int (round(y* (self.size -1))))
        return (newX,newY)
    
    def getSize(self):
        return (self.size)
# physics here: Acces the charge in an epsion x epsilon area element
    def getChargeInArea(self,x,y):
        density = self.getDensityAtXY(x,y)
        areaElement = self.epsilon*self.epsilon
        return density*areaElement
    
# calculate integral of density (by summing over data) and dividing throguh epsiolonSquare
    def getChargeInUnitCell(self):
        summe = np.sum(self.data)
      
        print ("Summe ohne Gewichtung "+str(summe))
        epsSquare = np.power(self.epsilon,2)
        print ("Epsionquare " + str(epsSquare))
          # another way
        weighteddens = self.data*epsSquare
        summe2 = np.sum(weighteddens)
        print (summe2)
        return (summe*epsSquare)
