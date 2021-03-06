import numpy as np
from dolfin import *
from fenics import *

class DensityFromFileClass(UserExpression):
    data =np.array([])
    fileName= ""
    size = 0
    def readData(self,fileName):

        print ("Do something here")
        filedata = np.loadtxt(fileName,usecols= (0,1,2), delimiter = ' ')
        density = filedata[:,2]
        self.size = int(np.sqrt(density.size))
        
        self.data = density.reshape(self.size,self.size)

        return (0)
    def eval(self, value, x):
        if self.fileName == "":
            self.readData("density.dat")

        xval = int(x[0])
        yval = int(x[1])
        #value[0] = 500.0*exp(-(dx*dx)/0.02)
        value[0] = self.data[xval,yval]
    def value_shape(self):
        return (1,)

#f0 = DensityFromFileClass()

