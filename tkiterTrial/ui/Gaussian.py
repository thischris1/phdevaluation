'''
Created on Sep 15, 2019

@author: chris
'''
import numpy as np
import os,sys
from scipy import signal
from scipy.optimize import curve_fit
import scipy.optimize as opt

    
def getPotential(imp, sigma, x,y):
    strength = imp[2]   
    sigsquare = sigma*sigma
    posx = imp[0]
    posy= imp[1]
    
    exponent = (posx-x)*(posx-x)+(posy-y)*(posy-y)
    exponent = exponent/sigsquare*-1.0
        
    return (strength*np.exp(exponent))

def getPotentialSum(imps,sigma,x,y):
    retVal = 0.0
    for anImp in imps:
        retVal = retVal + getPotential(anImp,sigma,x,y)
    return retVal
    
# returns the width and the positions of the impurities
def readGaussianFile(fileName):
    gaussArray = np.loadtxt(fileName, usecols=(0,1,2,4))
    impsx = gaussArray[:,0]
    impsy = gaussArray[:,1]
    sigma = gaussArray[0,2]
    strengths = gaussArray[:,3]
    
    imps = np.delete(gaussArray, 2, 1)
    return imps, sigma

def createPotentialFromGaussianFile(fileName):
    imps,sigma = readGaussianFile(fileName)
    x = np.linspace(0.0,1.0,101)
    y = np.linspace(0.0,1.0,101)
    x,y = np.meshgrid(x,y)
    pot = getPotentialSum(imps, sigma, x, y)
    return x,y,pot
    
def readPotentialFile(fileName):
    data = np.loadtxt(fileName)
    x = data[:,0]
    y = data[:,1]
    pot = data[:,2]
    x,y =np.meshgrid(x,y)
    return (x,y,pot)
# get , potentialvalue at x,y from file  
def getPotentialFromFile(fileName, xPos, yPos):
    x,y,potential = readPotentialFile(fileName)
    xSize = len(x)
    ySize = len(y)
    xIndex = int(xPos*xSize)
    yIndex = int(yPos*ySize) 
    return potential[xIndex,yIndex]

def generateRandomPotential(impCount, Vmax, sigma, Debug=False):

    if Debug == True:
       print ("generate a random potential with "+str(impCount)+" impurities")
    x =  np.arange(101)
    y = np.arange(101)
    x,y = np.meshgrid(x,y)
    imps = np.random.rand(impCount,2)
    strengths = np.random.rand(impCount,1)-0.5
    strengths = strengths*Vmax

    fimps = np.hstack((imps,strengths))
    potential = getPotentialSum(fimps,sigma,x/100.0,y/100.0)
   # print (potential)
#    potential = np.reshape(101,101)
    return strengths, fimps, potential

def procedure(i, impCount, Vmax,sigma, returnPotential=False):
    # print (i)
    path = "dir_"+str(i)
    # check if dir exists ( 1 ... numCount
    if (not os.path.isdir(path)):
       #do nothing
       os.mkdir(path)
    # change into dir
    os.chdir(path)
    createGaussian()
    #shutil.copy("../gaussian.par", ".")
    # create random.dat 
    writeRandom(impCount, sigma, Vmax)
   
    # create if necessary 
   
    # run gaussian -p
    os.system("gaussian -p")
    # evaluate
    
    data = readPotentialFile("PotentialArray.dat")
    pot = data[2]
    #print ("PotentialGroesse")
    #print (len(pot))
    lcorr = calculateAutoCorrelationFromFile("PotentialArray.dat")
    
    os.chdir("..")
    #artMean, artVariance,artMax = Gaussian.createAndEvaluatePotential(Vmax, impCount, sigma, x, y)
    #print ("Artifical")
    #print (artMean, artVariance, artMax)
   
    # print (np.mean(pot),np.var(pot),np.amax(pot))
    if returnPotential == True:
        return (pot,lcorr)
    else:
        return (np.mean(pot),np.var(pot),np.amax(pot), lcorr)
    
def createGaussian():
    f = open("gaussian.par","w")
    f.write ("./bs# root-name of basis file \n ./state_5_15_hc# root-name of vector file \n ./dnY0.0# root-name of density file \n")
    f.write ("./ldY0.0# root-name of landau-diagonal file\n")
    f.write ("./pot# root-name of potential file\n")
    f.write ("5       # Ne: Nr. of electrons\n")
    f.write ("15              # Nm: Nr. of flux quanta (i.e. Ne/Nm=filling factor)\n")
    f.write ("0               # spinYes: 0=spin polarized, 1=not necessarily sp. pol.\n")
    f.write ("0               # reqSz: dtto, with total Sz (applies only if spinYes=1)\n")
    f.write ("2               # mat_type: 0=FULL_REAL, 1=SPARSE_REAL, 2=FULL_CPLX\n")
    f.write ("1               # type of vector-file to generate: 0->ascii, 1->raw binary\n")
    f.write ("7              # eigsToFind: Nr. of eigvals/eigvecs to be found\n")
    f.write ("1.0             # a: size of the system (vert.)\n")
    f.write ("1.0       # b: size of the system (horiz.)\n")
    f.write ("0.0             # bli: related to finite thickness\n")
    f.write ("0               # type of barrier potential: 0 -> gaussian, 1 -> delta\n")
    f.write ("1               # type of e-e interaction: 0 -> Coulomb, 1 -> hardcore\n")
    f.write ("-2.0            # energy-offset\n")
    f.write ("0.0             # flux of solenoid1 in units of h/e\n")
    f.write ("0.0             # flux of solenoid2 in units of h/e\n")
    f.write ("100             # xkmax: Sum from -kmax to kmax for Barrier in x-direction (resp. hole)\n")
    f.write ("100             # ykmax: Sum from -kmax to kmax for Barrier in in y-direction (r\n")
    f.write ("random.dat")
    f.close()
                 
def writeRandom(impCount,sigma, strength):
    negstrength = -1.0*strength
    f = open("random.dat", "w")
    f.write(str(impCount)+"\n")
    f.write (str(sigma)+"\n")
    f.write (str(strength)+"\n")
    f.write (str(negstrength)+"\n")
         
    f.close()

def createPotentialFromGaussian(i, impCount, sigma, Vmax):

    path = "dir_"+str(i)
    # check if dir exists ( 1 ... numCount
    if (not os.path.isdir(path)):
       #do nothing
       os.mkdir(path)
    # change into dir
    os.chdir(path)
    createGaussian()
    #shutil.copy("../gaussian.par", ".")
    # create random.dat 
    writeRandom(impCount, sigma, Vmax)
   
    # create if necessary 
   
    # run gaussian -p
    os.system("gaussian -p")
    # evaluate
    
    data = readPotentialFile("PotentialArray.dat")
    pot = data[2]
    potSize = int(np.sqrt(len(pot)))
    #print (potSize)
    x = np.arange(potSize-1)
    y = np.arange(potSize-1)
    x = x/float(potSize-1)
    y = y/float(potSize-1)
    x = np.append(x,1.0)
    y = np.append(y,1.0)
    

    (x,y) = np.meshgrid(x,y)
    pot = np.reshape(pot,(potSize,potSize))
    #print (x)
    return x,y,pot

def calculateAutoCorrelationFromFile(fileName):
    pot = readPotentialFile(fileName)[2]
    # resize pot
    #print (pot.shape)
    potSize = len(pot)
    #print (potSize)
    potSize = int(np.sqrt(potSize))
    #print (potSize)
    pot = np.reshape(pot,(potSize,potSize))
    return calculateAutoCorrelation(pot, False)

def calculateAutoCorrelationData(pot):
    corr = signal.correlate2d(pot,pot)
    #    extract middle x
    corrSize = len(corr)
    cutX = corr[int(corrSize*0.5),:]
    cutLength = len(cutX)
    x = np.arange(corrSize)
    potX = np.arange(0,corrSize,10000)

   # print (cutX.shape)
    # fit to Gaussian
    p0 = [1., corrSize*0.5, 1.]
    coeff, var_matrix = curve_fit(gauss, x, cutX, p0=p0)
    fitted = gauss(x,*coeff)
    return (corr, cutX, np.abs(coeff[2]),fitted, np.sqrt(np.abs(var_matrix[2,2])))


def calculateAutoCorrelation(pot, plotting = False):
    corr = signal.correlate2d(pot,pot)
    #    extract middle x
    corrSize = len(corr)
    cutX = corr[int(corrSize*0.5),:]
    x = np.arange(corrSize)
    potX = np.arange(0,corrSize,10000)

   # print (cutX.shape)
    # fit to Gaussian
    p0 = [1., corrSize*0.5, 1.]
    coeff, var_matrix = curve_fit(gauss, x, cutX, p0=p0)
    fitted = gauss(x,*coeff)
    
#    fitted = gauss(potX,*coeff)
    if plotting ==True:
        import matplotlib.pyplot as plt
        plt.plot(x,cutX, label ='data')
        plt.plot(x,fitted, label='Fitted', linewidth=3)
        plt.legend()
        plt.show()
    #print (coeff)
    return np.abs(coeff[2])

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    
def findSmallestAndLargestCorrelationLength( Vmax, Nimp, sigma, reps):
    potSmallLcorr = None
    potLargeLcorr = None
    lcorrSmall = 1000.0
    lcorrLarge = -1000.0
    smallIndex = 1
    largeIndex = 1
    for index in np.arange(0,reps):
        (pot,lcorr) = procedure(index,Nimp,Vmax,sigma,True)
        # start 
        if index == 1:
            potSmallLcorr = pot
            potLargeLcorr = pot
            lcorrSmall = lcorr
            lcorrLarge = lcorr
            continue
        if lcorrSmall > lcorr:
            potSmallLcorr = pot
            lcorrSmall = lcorr
            smallIndex = index
            continue
        if lcorrLarge < lcorr:
            potLargeLcorr = pot
            lcorrLarge = lcorr
            largeIndex = index
    print ("Small",smallIndex)
    print ("Large", largeIndex)
    return (potLargeLcorr, lcorrLarge,potSmallLcorr, lcorrSmall)


def calculateAutoCorrelation2dFromFile(fileName, plot =False):
    data  = np.loadtxt(fileName)
    pot = data[:,2]
    potSize = len(pot)
    print (potSize)
    potSize = int(np.sqrt(potSize))
    print (potSize)
    pot = np.reshape(pot,(potSize,potSize))
    return (calculateAutoCorrelation2dFromPotentialData(pot, plot))
   
    
def calculateAutoCorrelation2dFromPotentialData(pot, plot = False):
    
    corr = signal.correlate2d(pot,pot).ravel()
    return calculateAutoCorrelation2dFromACFplot(corr, plot)

def calculateAutoCorrelation2dFromACFplot(corr, plot = False):
    
    corrSize = int(len(corr))
    print ("Corr size ="+str(corrSize))

# Create x and y indices
    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    x, y = np.meshgrid(x, y)

    passTuple = (x,y)
#create data
    data = twoD_Gaussian(passTuple,3, 100, 100, 20, 40, 0, 10)
    initial_guess = (3,100,100,20,40,0,10)

#data_noisy = data + 0.2*np.random.normal(size=data.shape)
    data_noisy = corr
    
    

    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)
    data_fitted = twoD_Gaussian((x, y), *popt) 
#print (popt, pcov)
# plot twoD_Gaussian data generated above
    if (plot == True): 
        import pylab as plt
       
        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, data_fitted.reshape(201, 201), 1, colors='w')
#ax.colorbar()
        plt.show()
    sigmax = np.abs(popt[3])
    sigmay = np.abs(popt[4])
    print (sigmax,sigmay)
    return (sigmax, sigmay,data_fitted)

def twoD_Gaussian(xyTuple,amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xyTuple[0]
    y = xyTuple[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)  + c*((y-yo)**2)))
    return g.ravel()



    
