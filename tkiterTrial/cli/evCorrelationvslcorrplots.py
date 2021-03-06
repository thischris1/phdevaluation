'''
Created on Apr 9, 2020

@author: chris
'''
import numpy as np
import sys, os
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import rc
from numpy import double

import scipy 
from pylab import *
from scipy.optimize import curve_fit






#from cli.clusterEvaluation import allData


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# berechne den mittelwwet und die Varianz der Spalte index in data
# gib den Mittelwert und die Varianz zurueck

def calcSplittingCoefficent(data):
    bandwidth = data[:,gsIndex+1]- data[:,gsIndex+3]
    correlationGap = data[:,gsIndex+1]- data[:,gsIndex+4]
    
    splittingFactor = bandwidth / correlationGap
    print("Splitting")
    print (data[0,:])
    print ("Ground state")
    print (data[0,gsIndex])
    print (bandwidth[0],correlationGap[0], splittingFactor[0])
    print ("Shapes")
    
    splittingFactor = np.expand_dims(splittingFactor, -1)
    print (np.shape(splittingFactor))
    return splittingFactor
    



def calculateMeanAndVarianceSingleEntity( data, index):
    tempArray = []
    if data.size == 0:
        return
    print (data.shape)
    for aDatum in data:
        tempVal = aDatum[index]
        if np.isnan(tempVal):
            continue
        tempArray= np.append(tempArray,tempVal)
    
    tempArray = np.asarray(tempArray)
    mean = np.mean(tempArray)
    var = np.var(tempArray)
    if np.isnan(mean) or np.isnan(var):
        return
    return (mean,var)

# Berechne varainz und mittelwert in den Spalten in indexlist von data  
# Gibt ein Array der Form [[mean1, var1],[mean2,var2],...] zurueck
#
def calculateMeanAndVarianceListOfEntities( data, indexlist):
    retArray = []
    for index in indexlist:
        (mean, var) = calculateMeanAndVarianceSingleEntity(data, index)
        retArray = np.hstack((retArray,[mean,var]))
    return retArray

def calculateMeanFormVmaxSigmaList( vosigmaList, data, indexList, plot = True):
    # indices to be meaned and aard given in indexList,
    # 
    indexSize = int(len(indexList))*2 +1
    resultArray = np.empty([0,indexSize])
    exVoSigmaList = []
    print (resultArray.shape)
    for vosigma in vosigmaList:
        vosigmaConstArray = data[np.where(np.abs(data[:,VmaxIndex]*data[:,sigmaIndex] - vosigma) < 1e-07)]
        if vosigmaConstArray.size == 0:
         
            continue
        exVoSigmaList.append(vosigma)
        retVal = calculateMeanAndVarianceListOfEntities(vosigmaConstArray, indexList)
        retVal= np.append(retVal,vosigma)
        print (retVal.shape)
        print (resultArray.shape)
        resultArray = np.vstack((resultArray,retVal))
    if plot:
        plt.plot(exVoSigmaList, resultArray[:,0], "ro")
        plt.xlabel("$V_{max}\sigma$")
        plt.ylabel("$r_{ev}$")
        plt.show()

def plotrevvvOverlcorrAllVoSigma(data, vosigmaList):
    
    print (data.shape)
    for vosigma in vosigmaList:
        vosigmaConstArray = data[np.where(np.abs(data[:,VmaxIndex]*data[:,sigmaIndex] - vosigma) < 1e-09)]
        print (vosigmaConstArray.shape)
        if vosigmaConstArray.size == 0:
         
            continue
        vosigmastring = '{:.2e}'.format(vosigma)
        titleString ="V  sigma = "+ str(vosigmastring)
        plt.plot(vosigmaConstArray[:,lcorrIndex],vosigmaConstArray[:,evMaxIndex], "ro", label=titleString)
        
        plt.xlabel("l_{corr}")
        plt.ylabel("r_{ev}")
        plt.legend(titleString)
    plt.show()

def addMaxtwoColumn(data):
    maxArray =[]
    for aData in data:
        
        
        if aData[twodcorrelationx]> aData[twodcorrelationy]:
            maxArray = np.append(maxArray,aData[twodcorrelationx])
        else:
            maxArray= np.append(maxArray,aData[twodcorrelationy])
    maxArray = np.expand_dims(maxArray, -1)
    print (maxArray.shape)
    return (maxArray)
def plotLcorrOverGap(data,Ne,ia):
    
    # find max and min value of gap
    lcorrArray = np.array([])
    gapMin = np.min(data[:,gapStateIndex]-data[:,gsIndex])
    gapMax = np.max(data[:,gapStateIndex]-data[:,gsIndex])
    deltaGap = (gapMax- gapMin)/10.0
    print (gapMin, gapMax, deltaGap)
    figs,axs = plt.subplots(2,1 )
    titleString = " Electrons"
    titleString = str(Ne)+titleString
    if ia == 0:
        titleString = titleString+" Coulomb interaction"
    else:
        titleString = titleString+" SRI"
    plt.suptitle(titleString)
    axs[0,0].plot(data[:,gapStateIndex]-data[:,gsIndex], data[:,evMaxIndex], 'ro')
    for aGap in np.linspace(gapMin, gapMax, 10):
        print (aGap)
        # get subarray data
        lines = data[np.where((np.abs(data[:,gapStateIndex]-data[:,gsIndex])-aGap) < deltaGap)]
        lcorravg = np.nanmean(lines[:,evMaxIndex])
        lcorrVar = np.nanvar(lines[:,evMaxIndex])
        lcorrArray = np.append(lcorrArray,lcorravg)
    axs[0,0].plot(np.linspace(gapMin, gapMax, 10),lcorrArray, 'g+',ms=20)
    plt.show()     

def getScaleFactor(Ne):
    retval = np.sqrt(Ne*3*2*np.pi) /1000.0 
    return retval

def getUnitcellSizeinLo(Ne):
    return np.sqrt(Ne*3*2*np.pi)

def plotrevVsLcorr(data,Ne,ia):
    
    
    if len(ia) > 1:    
        # beide ww typen
        print (ia)
        if len(ne) > 1:
            # wenn auch alle elec Typen: 2x2
            figs,axs = plt.subplots(2,2 )
        else:
        # sonst 2x1
            figs,axs=plt.subplots(2,1)
    else:
        if len(ne) > 1:
            figs,axs=plt.subplots(2,1)
        else:
            figs,axs= plt.plot()
    
    titleString = " Electrons"
    titleString = str(Ne)+titleString
    if ia == 0:
        titleString = titleString+" Coulomb interaction"
    else:
        titleString = titleString+" SRI"
    plt.suptitle(titleString)
    plt.plot(data[:,lcorrIndex]/getUnitcellSizeinLo(Ne), data[:,evMaxIndex]*getScaleFactor(Ne), 'ro')
    plt.show()
    
def expfunc(x, a, c, d):
    return a*np.exp(-c*x)+d   


def plotrevVsLcorrperVoSigma(data):
    
    plt.show()

def get_color():
    for item in ['r', 'g', 'b', 'c', 'm', 'y', 'k','#eeefff','r', 'g', 'b', 'c', 'm', 'y', 'k']:
        yield item

def plotrevOverLcorrRaw(data,Ne,ia, vosigmalList = None,maximum = False, Fit =False):
    offset = 0.0   
    if Ne == 5: 
        theData = allData[allData[:,impCountIndex]== 4000]
        
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    unitCellSize = np.sqrt(2*np.pi*3*Ne)
    #theData.vstack
    electronNumberDataArray =theData[theData[:,0]==Ne]  
    # withount invalid data
    electronNumberDataArray = electronNumberDataArray[np.where(np.abs(electronNumberDataArray[:,splittingIndex] <0.5))]
    print ("Remove array with too much splitting")
    HCArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
    if vosigmalList is None:
        vosigma = electronNumberDataArray[:,sigmaIndex]*electronNumberDataArray[:,VmaxIndex]
        vosigmalList = np.unique(vosigma.round(decimals = 7))
    
   # figs,axs = plt.subplots(2,1)
    titleString = " Electrons"
    titleString = str(Ne)+titleString
    if ia == 0:
        titleString = titleString+" Coulomb interaction"
        offset = 0.6
    else:
        titleString = titleString+" SRI"
    plt.suptitle(titleString)   
    colors = get_color()  
   # axs[0,0].plot(HCArray[:,lcorrIndex], HCArray[:,evMaxIndex], 'o')
   # axs[0,1].plot(HCArray[:,lcorrIndex], HCArray[:,vvMaxIndex]*0.5,'+')
    print(vosigmalList)
    acolor = next(colors)  
    for aVmaxSigma in vosigmalList:
        acolor = next(colors)
        vosigmaConstArray = HCArray[np.where(np.abs(HCArray[:,VmaxIndex]*HCArray[:,sigmaIndex] - aVmaxSigma) < 1e-6)]
        vosigmaConstArray = np.delete(vosigmaConstArray,2, axis = 1)
        print (aVmaxSigma)
        #vosigmaConstArray = vosigmaConstArray[~np.isnan(vosigmaConstArray)]
        #print (vosigmaConstArray.shape)
        # fit
        vosigmaConstArray = vosigmaConstArray[~np.isnan(vosigmaConstArray).any(axis=1), :]
        
        plt.plot(vosigmaConstArray[:,lcorrIndex-1]*9.87/200, vosigmaConstArray[:,evMaxIndex-1]*9.87/1000, 'o', ms = 5, mfc =acolor)
        if len(vosigmaConstArray) < 10:
            print ("Not enough data")
            continue

        try:
            coeffs, errors = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  vosigmaConstArray[:,lcorrIndex-1], vosigmaConstArray[:,evMaxIndex-1],  p0=(4, -0.1,0.5))
            #coeffs, errors = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  vosigmaConstArray[:,lcorrIndex-1]*9.87/200, vosigmaConstArray[:,evMaxIndex-1]*9.87/1000,  p0=(4, -0.1,0.5))
            
            fitX = np.linspace(0, 40, 1000)
            fitData = coeffs[0]*np.exp(coeffs[1]*fitX)+coeffs[2]
            fitX = fitX*9.97/200
            fitData = fitData * 9.87/1000
            print (coeffs)
            try:
                finalCoeffs, finalErrors = scipy.optimize.curve_fit(lambda t1,a1,b1,c1: a1*np.exp(b1*t1)+c1, fitX, fitData,  p0=(1, coeffs[1]/0.049,0.5))
                plotData = finalCoeffs[0]*np.exp(finalCoeffs[1]*fitX)+finalCoeffs[2]
                print (finalCoeffs) 
                labelString = "$V_{max}\sigma = "+ str(aVmaxSigma)+"$ Fit to $e^{ " +  str(round(finalCoeffs[1],2))+"}$"
                plt.plot(fitX,plotData,"-", label = labelString, color = acolor,ms=20, lw=3)
            except e:
                print ("Secondary fit failed")
            
                
        except:
            print ("FitFehler")
            print("Unexpected error:", sys.exc_info()[0])  
            print (vosigmaConstArray.shape)
    plt.xlabel("$l_{corr} [l_{0}]$",fontsize=16)
    plt.ylabel("$r_{ev} [l_{0}]$",fontsize=16)
    plt.legend()
    plt.show()

def plotrevOverLcorr(data,Ne,ia, maximum = False, Fit =False):
    # plot Ne = 5 , Interaction = HC, all points

    # only fitting impurities
    offset = 0.0
    if Ne == 5:
        theData = allData[allData[:,impCountIndex]== 4000]
        
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    unitCellSize = np.sqrt(2*np.pi*3*Ne)
    #theData.vstack
    electronNumberDataArray =theData[theData[:,0]==Ne]  
    # withount invalid data
    electronNumberDataArray = electronNumberDataArray[np.where(np.abs(electronNumberDataArray[:,splittingIndex] <0.5))]
    print ("Remove array with too much splitting")
    HCArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
    vosigma = electronNumberDataArray[:,sigmaIndex]*electronNumberDataArray[:,VmaxIndex]
    vosigmalList = np.unique(vosigma.round(decimals = 7))
    print (vosigmalList)
   # figs,axs = plt.subplots(2,1)
    titleString = " Electrons"
    titleString = str(Ne)+titleString
    if ia == 0:
        titleString = titleString+" Coulomb interaction"
        offset = 0.6
    else:
        titleString = titleString+" SRI"
    plt.suptitle(titleString)   
    color = get_color()  
   # axs[0,0].plot(HCArray[:,lcorrIndex], HCArray[:,evMaxIndex], 'o')
   # axs[0,1].plot(HCArray[:,lcorrIndex], HCArray[:,vvMaxIndex]*0.5,'+')
    for aVmaxSigma in vosigmalList:
        vosigmaConstArray = HCArray[np.where(np.abs(HCArray[:,VmaxIndex]*HCArray[:,sigmaIndex] - aVmaxSigma) < 1e-6)]
        # fit all data to an exponential
        
        
        sigmaList = np.unique(vosigmaConstArray[:,sigmaIndex])
        print (aVmaxSigma)
        print (sigmaList)
       
        if len(sigmaList) > 3:
            correlationVsSigmaArray = np.empty((0,9))
            max = 0.0
            for aSigma in sigmaList:
                sigmaData = vosigmaConstArray[vosigmaConstArray[:,sigmaIndex] == aSigma]
                
                #print (sigmaData.shape)
                meanRev = np.nanmean(sigmaData[:,evMaxIndex] )*unitCellSize/1000 - offset
                #meancorrMax = np.mean(np.maximum(sigmaData[:,twodcorrelationx],sigmaData[:twodcorrelationy]))
               # print (meancorrMax)
                meanrvv = np.nanmean(sigmaData[:,vvMaxIndex]) * 0.5
                varRev = np.sqrt(np.nanvar(sigmaData[:,evMaxIndex]*unitCellSize/1000,dtype=np.float64))
                varRvv = np.sqrt(np.nanvar(sigmaData[:,vvMaxIndex]*unitCellSize/100, dtype=np.float64))
                meantwod = np.nanmean(sigmaData[:,twodmaxIndex])*unitCellSize/200
                vartwod = np.nanvar(sigmaData[:,twodmaxIndex])*unitCellSize/200
                meanLcorr = np.nanmean(sigmaData[:,lcorrIndex]*unitCellSize/200)
               
                #meanLcorr2D = np.nanmean(np.amax(sigmaData[0:[]]))
                varLcorr = np.nanvar(sigmaData[:,lcorrIndex]*unitCellSize/200)
                #print ("LCorr 1D vs 2D", meanLcorr, varLcorr, meantwod, vartwod)
                if np.isnan(varRev):
                    print ("NAN in VAR")
                    print  (sigmaData[:,evMaxIndex])
                    print ("ENDE")
                print ("variance = ", varRev, varRvv, vartwod,aSigma)
                print ("mean = " ,meanRev, meanrvv, meantwod,aSigma)
                # einpacken in ein array
                resultArray = np.array([aSigma,meanRev,varRev,meanrvv,varRvv,meantwod,vartwod, meanLcorr, varLcorr])
                correlationVsSigmaArray = np.vstack((correlationVsSigmaArray, resultArray)) 
            if Fit =='P':
                z = np.polyfit(correlationVsSigmaArray[:,7], correlationVsSigmaArray[:,1], 3)
                p = np.poly1d(z)
                fitX = np.linspace(0, np.max(correlationVsSigmaArray[:,7])*1.2, 20)
                fitData = p(fitX)
                print (z)
            if Fit == 'E':
                try:
                    coeffs, errors = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  correlationVsSigmaArray[:,7], correlationVsSigmaArray[:,1],  p0=(4, -0.1,0.5))
                    fitX = np.linspace(0, np.max(correlationVsSigmaArray[:,7])*1.2, 20)
                    fitData = coeffs[0]*np.exp(coeffs[1]*fitX)+coeffs[2]
                    coeffs2, error2 = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(b*t)+c,  vosigmaConstArray[:,lcorrIndex], vosigmaConstArray[:,evMaxIndex],  p0=(4, -0.1,0.5))
                    fitData2 = coeffs2[0]*np.exp(coeffs2[1]*fitX)+coeffs2[2] 
                    print(coeffs)
                except RuntimeError:
                    Fit = False  
            
            
            acolor = next(color)
            if Fit is not False:
                plt.plot(fitX,fitData, lw=3, color = acolor)
                plt.plot(fitX,fitData2, lw = 3, color = acolor , label="Full fit")
            print (correlationVsSigmaArray.shape)
            if max < np.amax(correlationVsSigmaArray[:,1])*1.2:
                max = np.amax(correlationVsSigmaArray[:,1])*1.2
            #axs[2,0].plot(vosigmaConstArray[:,sigmaIndex]*9.871, vosigmaConstArray[:,evMaxIndex], 'ro')
            aVmaxSigmaString = '{:.2e}'.format(aVmaxSigma)
            if maximum == False:
                plt.errorbar(correlationVsSigmaArray[:,7],correlationVsSigmaArray[:,1],xerr= np.sqrt(correlationVsSigmaArray[:,8]), yerr=np.sqrt(correlationVsSigmaArray[:,2]), marker='s',
          ms=20, mew=4,color = acolor,linestyle='None',label =aVmaxSigmaString)
            else:
                plt.errorbar(correlationVsSigmaArray[:,5],correlationVsSigmaArray[:,1],xerr= np.sqrt(correlationVsSigmaArray[:,6]), yerr=np.sqrt(correlationVsSigmaArray[:,2]), marker='s',
          ms=20, mew=4,color = acolor,linestyle='None',label =aVmaxSigmaString)
                
           #plt.plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray[:,evMaxIndex], )
            
            plt.xlabel('$l_{corr} [l_{0}]$',fontsize=16)
            if ia == 0:
                plt.ylabel('$\Delta r_{ev}r_{ev}^{homo}[l_{0}]$',fontsize=16)
            else:
                plt.ylabel('$r_{ev}[l_{0}]$',fontsize=16)
            plt.legend(numpoints=1)
            plt.xlim(left=0)
           
            plt.xlim(right=3.0)
            plt.ylim(bottom=0)
            plt.ylim(top=max)
           # axs[2,1].plot(vosigmaConstArray[:,sigmaIndex]*9.871, vosigmaConstArray[:,vvMaxIndex], 'bo')  
           # plot lcorr (x), rev 
#             axs[0,1].errorbar(correlationVsSigmaArray[:,5]*9.871/100, correlationVsSigmaArray[:,1],xerr=correlationVsSigmaArray[:,6], yerr=correlationVsSigmaArray[:,2],  marker='s',
#           ms=20, mew=4,linestyle='None',label=str(aVmaxSigma))
#             axs[0,1].set_ylim(bottom=0)
#             axs[0,1].set_ylim(top = max)
#             axs[0,1].set_xlim(left=0)
#             axs[0,1].legend(numpoints=1)
#             axs[0,1].set_title("$r_{ev}(l_{corr}^{2d})$")
        else:
            continue
        
        
        # perform a fit to an exponential for all data
    fitArray = HCArray[~np.isnan(HCArray[:,lcorrIndex])]
    fitArray = fitArray[~np.isnan(fitArray[:,evMaxIndex])]
    # if coulomb interaction remove small evMaxVlaues
    if ia == 0:
        fitArray = fitArray[np.where(fitArray[:,evMaxIndex]> 10)]
    #fitArray = fitArray[np.where(fitArray[:,evMaxIndex]>2)]
    popt, pcov = curve_fit(expfunc, fitArray[:,lcorrIndex], fitArray[:,evMaxIndex], p0=(1, 1e-6, 1))
    print (popt)
    print ("PCOV =")
    print (pcov)
    expxData = np.linspace(0.0,np.max(fitArray[:,lcorrIndex]),100)
    expyData = expfunc(expxData, *popt)
    print (popt[:])
    newSigma = popt[1]*unitCellSize/200.0
    newSigma = round(newSigma,3)
    #plot(expxData,expyData,label="exp. Fit", lw =4)
    labelString = str(aVmaxSigma)
    labelString = '{:.2e}'.format(aVmaxSigma)      #  axs[1,0].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,evMaxIndex], '*', label = labelString)
        #axs[1,1].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,vvMaxIndex], '+', label = labelString)
    plt.legend()
   
    plt.show()
    
    plot(fitArray[:,lcorrIndex]*unitCellSize/200,fitArray[:,evMaxIndex]*unitCellSize/1000,'ro')
    plot(expxData*unitCellSize/200,expyData*unitCellSize/1000,label="$$\propto e^{"+str(newSigma)+" x}$$", lw =4)
    plt.xlabel('$l_{corr} [l_{0}]$',fontsize=16)
    if ia == 0:
        plt.ylabel('$\Delta r_{ev}r_{ev}^{homo}[l_{0}]$',fontsize=16)
    else:
        plt.ylabel('$r_{ev}[l_{0}]$',fontsize=16)
    labelString = str(aVmaxSigma)
      #  axs[1,0].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,evMaxIndex], '*', label = labelString)
        #axs[1,1].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,vvMaxIndex], '+', label = labelString)
    plt.legend()
    plt.show()
    #sys.exit()
    
def revVsLcorrIacomparison(allData,Ne, vosigmaList):
    ialist =(0,1)
    if Ne == 5:
        theData = allData[allData[:,impCountIndex]== 4000]
        
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    unitCellSize = np.sqrt(2*np.pi*3*Ne)
    print (vosigmaList)
    #theData.vstack
    electronNumberDataArray =theData[theData[:,0]==Ne] 
    # alles mit nE 
    print (electronNumberDataArray.shape)
    HCArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==1]
    print (HCArray.shape)
    CoulArray  = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==0]
    print (CoulArray.shape)
    iaString  =""
    color = get_color()
    for ia in ialist:
        print (ia)
        offset = 0.0
        if ia == 0:
            iaString = " ,Coulomb"
            offset = 0.6
        else:
            iaString = " ,SRI"
       
        iaArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
        # get Data
        vosigmalist = iaArray[:,sigmaIndex]*iaArray[:,VmaxIndex]
        vosigmalList = np.unique(vosigmalist.round(decimals = 7))
        print (vosigmalList)
        for vosigma in vosigmaList:
            print (vosigma)
            
            vosigmaConstArray = iaArray[np.where(np.abs(iaArray[:,VmaxIndex]*iaArray[:,sigmaIndex] - vosigma) < 1e-10)]
            sigmaList = np.unique(vosigmaConstArray[:,sigmaIndex])
            print (sigmaList)
       
            if len(sigmaList) > 3:
                correlationVsSigmaArray = np.empty((0,9))
                max = 0.0
                for aSigma in sigmaList:
                    sigmaData = vosigmaConstArray[vosigmaConstArray[:,sigmaIndex] == aSigma]
                    print (sigmaData.shape)
                    meanRev = np.nanmean(sigmaData[:,evMaxIndex] )*unitCellSize/1000 - offset
                    #meancorrMax = np.mean(np.maximum(sigmaData[:,twodcorrelationx],sigmaData[:twodcorrelationy]))
                   # print (meancorrMax)
                    meanrvv = np.nanmean(sigmaData[:,vvMaxIndex]) * 0.5
                    varRev = np.sqrt(np.nanvar(sigmaData[:,evMaxIndex]*unitCellSize/1000,dtype=np.float64))
                    varRvv = np.sqrt(np.nanvar(sigmaData[:,vvMaxIndex]*unitCellSize/100, dtype=np.float64))
                    meantwod = np.nanmean(sigmaData[:,twodmaxIndex])*unitCellSize/200
                    vartwod = np.nanvar(sigmaData[:,twodmaxIndex])*unitCellSize/200
                    meanLcorr = np.nanmean(sigmaData[:,lcorrIndex]*unitCellSize/200)
                   
                    #meanLcorr2D = np.nanmean(np.amax(sigmaData[0:[]]))
                    varLcorr = np.nanvar(sigmaData[:,lcorrIndex]*unitCellSize/200)
                    print ("LCorr 1D vs 2D", meanLcorr, varLcorr, meantwod, vartwod)
                    if np.isnan(varRev):
                        print ("NAN in VAR")
                        print  (sigmaData[:,evMaxIndex])
                        print ("ENDE")
                    print ("variance = ", varRev, varRvv, vartwod,aSigma)
                    print ("mean = " ,meanRev, meanrvv, meantwod,aSigma)
                    # einpacken in ein array
                    resultArray = np.array([aSigma,meanRev,varRev,meanrvv,varRvv,meantwod,vartwod, meanLcorr, varLcorr])
                    correlationVsSigmaArray = np.vstack((correlationVsSigmaArray, resultArray)) 
            
            
            vosigmaString = '{:.2e}'.format(vosigma)
            plt.errorbar(correlationVsSigmaArray[:,5],correlationVsSigmaArray[:,1],xerr= np.sqrt(correlationVsSigmaArray[:,6]), yerr=np.sqrt(correlationVsSigmaArray[:,2]), marker='s',
          ms=20, mew=4,linestyle='None',label =str(vosigmaString)+iaString)
            
    rightLimit = np.max(correlationVsSigmaArray[:,1])*1.5
    plt.ylim(bottom=0)
    plt.xlim(right=3)
    plt.xlim(left=0.0)
    plt.xlabel('$l_{corr} [l_{0}]$',fontsize=16)
    
    plt.ylabel('$\Delta r_{ev}r_{ev}^{homo}[l_{0}]$',fontsize=16)
    
    plt.legend(numpoints=1)
    
    plt.show()
def revOverLcorrConstGap(data, Ne,ia):
    
    return
def plotlCorrOnetwoD(data, Ne):
    if Ne == 5:
        theData = allData[allData[:,impCountIndex]== 4000]
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    
    print (np.amax(theData[:,twodcorrelationx]), np.amax(theData[:,twodcorrelationy]))
    maxYindex = np.argmax(theData[:,twodcorrelationy])
    print (theData[maxYindex,:])
    plt.plot(theData[:,lcorrIndex], theData[:,twodcorrelationx], label='2Dx')
    plt.plot(theData[:,lcorrIndex], theData[:,twodcorrelationy], label = "2dy")
    plt.legend()
    plt.show()
    return


def showNonsenseValues(data):
    # lcorr too small or too large
    print ("One line")
    print (data[1,:])
    strippedData = np.delete(data, 2, axis=1)
    print (strippedData[1,:])
    bogusData = strippedData[np.where(strippedData[:,lcorrIndex] < 1.0)]
    print (bogusData)
    
    # everything with Nan in it
    print(strippedData[np.isnan(strippedData[:[NeIndex]]).any(axis=1)])
    print (len(strippedData))
    print (len(strippedData[np.isnan(strippedData).any(axis=1)]))

    
    return

def getVarianceData(data):
    retVal = data[np.abs(data[:,varIndex]) < 0.001 ]
    return retVal


def plotrevVsVariance(allData,Ne,ia,originalData=True):
    if Ne == 5:
        theData = allData[allData[:,impCountIndex]== 4000]
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    
    electronNumberDataArray =theData[theData[:,0]==Ne]  
    HCArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
    vosigma = electronNumberDataArray[:,sigmaIndex]*electronNumberDataArray[:,VmaxIndex]
    vosigmalList = np.unique(vosigma.round(decimals = 6))
    varMax = np.amax(HCArray[:,varIndex])
    
    print (varMax)
    varMax = 0.00003
    varianceBins = 40
    print (vosigmalList)
    if originalData:
    
        for aVmaxSigma in vosigmalList:
            vosigmaConstArray = HCArray[np.where(np.abs(HCArray[:,VmaxIndex]*HCArray[:,sigmaIndex] - aVmaxSigma) < 1e-09)]
            plt.plot(vosigmaConstArray[:,varIndex], vosigmaConstArray[:,evMaxIndex]/1000*9.871,'o', label=str(aVmaxSigma))
        
    
    varianceArray = np.linspace(0.0, varMax, varianceBins)
    varianceDelta = varMax/varianceBins
    print ("variancedelta and variances")
    print (varianceDelta)
    print(varianceArray)
    results =np.empty((0,4),double)
     
    for aVariance in varianceArray:
        var1Array = HCArray[np.where(HCArray[:,varIndex]>aVariance)]
        varArray = var1Array[np.where(var1Array[:,varIndex]<aVariance+varianceDelta)]
        print (len(varArray))
        print (calculateAverage(varArray, lcorrIndex))
        lcorrMean = np.nanmean(varArray[:,lcorrIndex])
        lcorrVar = np.nanvar(varArray[:,lcorrIndex])
        varMean = np.nanmean(varArray[:,varIndex])
        varVariance = np.nanvar(varArray[:,varIndex])
        print (varMean, lcorrMean, lcorrVar, varVariance)
        results = np.vstack((results, np.array([[aVariance, lcorrMean, lcorrVar,varVariance]])))
    print (results)
    print (results[:,0])
    plt.errorbar(results[:,0], results[:,1]/1000*9.871, xerr=results[:,3], yerr=results[:,2]/1000*9.871,marker = 's',label="Average",ms=20)
    plt.xlim(0,varMax*1.5)
    plt.ylim(0,np.max(results[:,1]*20.0)/1000*9.871)
    plt.xlabel('var (V(r))')
    plt.ylabel('$r_{ev} [l_{0}]$')
    plt.legend()
    plt.show()
def plotrevVsgap(allData, Ne, ia):
    if Ne == 5:
        theData = allData[allData[:,impCountIndex]== 4000]
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    
    electronNumberDataArray =theData[theData[:,0]==Ne]  
    iaArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
    print (iaArray[1,gsIndex], iaArray[1,gapStateIndex], iaArray[1,gsIndex]- iaArray[1,gapStateIndex])
    plt.plot(iaArray[:,gsIndex]- iaArray[:,gapStateIndex],iaArray[:,evMaxIndex], 'o')
    plt.show()
    return
def plotrevOvervariancefixLcorr(data, Ne, ia, lcorrs, epsilon = 0.1):
    if Ne == 5:
        theData = allData[allData[:,impCountIndex]== 4000]
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    
    electronNumberDataArray =theData[theData[:,0]==Ne]  
    iaArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
    epsilonString = str(round(epsilon,2))
    lcorrmax = np.max(iaArray[:,lcorrIndex])
    colors = get_color()  
    print ("Max lccorr =" + str(lcorrmax))
    for anlcorr in lcorrs:
        print (anlcorr)
        aColor = next(colors)
        anLcorr = anlcorr*200/9.871
        tempData = iaArray[np.where(np.abs(iaArray[:,lcorrIndex] -anLcorr) < epsilon) ]
        rescaleTemp = tempData[:,twodmaxIndex]*9.871/1000
        fitData = np.polyfit(tempData[:,varIndex], rescaleTemp, 1)
        fitPoly = np.poly1d(fitData)
        mean = np.nanmean(rescaleTemp)
        var = np.sqrt(np.nanvar(rescaleTemp))
        print (fitData, mean, var)
        print (tempData[0,:])
        labelString = "$l_{corr} = " +str(anlcorr)+" \pm" + epsilonString +"l_{0}$"
        #plt.plot(tempData[:,varIndex],tempData[:,twodmaxIndex], 'o', color = aColor,label =labelString )
        plt.plot(tempData[:,varIndex],rescaleTemp, 'o', color = aColor,label =labelString )
        plt.plot(tempData[:,varIndex], fitPoly(tempData[:,varIndex]), '-', color = aColor, label ='_nolegend_' )
        #plt.errorbar(tempData[:,varIndex], fitPoly(tempData[:,varIndex]), np.sqrt(var),color = aColor, label ='_nolegend_')
        plt.fill_between(tempData[:,varIndex], mean-var, mean+var,color=aColor, alpha =0.3)
    plt.xlim(left= 0.0)
    plt.xlim(right=1e-05)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xlabel('var (V(r))')
    plt.ylabel('$r_{ev} [l_{0}]$')
    plt.legend()
    plt.legend()
    plt.savefig("revOverVarianceFixLCorr.png")
    plt.show()
def calculateAverage(data, index):
    
    length = len(data)
    value = 0.0
    count = 0
    print ("calcualte average, anzahl der Daten", length)
    for row in data:
        if row[index]> 1:
            value = value + row[index]
        
            count = count +1
    if count == 0:
        return -100000 
    return value/count
def gnuplotComparisonPlot(data, Ne, ia):
# plot Ne = 5 , Interaction = HC, all points

    # only fitting impurities
    if Ne == 5:
        theData = allData[allData[:,impCountIndex]== 4000]
        
    if Ne == 6: 
        theData =  allData[allData[:,impCountIndex]== 4800]
    
    electronNumberDataArray =theData[theData[:,0]==Ne]  
    HCArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
    vosigma = electronNumberDataArray[:,sigmaIndex]*electronNumberDataArray[:,VmaxIndex]
    vosigmalList = np.unique(vosigma.round(decimals = 6))
    print (vosigmalList)
    figs,axs = plt.subplots(2,2 )
    titleString = " Electrons"
    titleString = str(Ne)+titleString
    if ia == 0:
        titleString = titleString+" Coulomb interaction"
    else:
        titleString = titleString+" SRI"
    plt.suptitle(titleString)   
  
   # axs[0,0].plot(HCArray[:,lcorrIndex], HCArray[:,evMaxIndex], 'o')
   # axs[0,1].plot(HCArray[:,lcorrIndex], HCArray[:,vvMaxIndex]*0.5,'+')
    for aVmaxSigma in vosigmalList:
        vosigmaConstArray = HCArray[np.where(np.abs(HCArray[:,VmaxIndex]*HCArray[:,sigmaIndex] - aVmaxSigma) < 1e-09)]
        sigmaList = np.unique(vosigmaConstArray[:,sigmaIndex])
        print (sigmaList)
        
        if len(sigmaList) > 3:
            correlationVsSigmaArray = np.empty((0,7))
            max = 0.0
            for aSigma in sigmaList:
                sigmaData = vosigmaConstArray[vosigmaConstArray[:,sigmaIndex] == aSigma]
                print (sigmaData.shape)
                meanRev = np.nanmean(sigmaData[:,evMaxIndex])
                #meancorrMax = np.mean(np.maximum(sigmaData[:,twodcorrelationx],sigmaData[:twodcorrelationy]))
               # print (meancorrMax)
                meanrvv = np.nanmean(sigmaData[:,vvMaxIndex]) * 0.5
                varRev = np.sqrt(np.nanvar(sigmaData[:,evMaxIndex],dtype=np.float64))
                varRvv = np.sqrt(np.nanvar(sigmaData[:,vvMaxIndex], dtype=np.float64))
                meantwod = np.nanmean(sigmaData[:,twodmaxIndex])
                vartwod = np.nanvar(sigmaData[:,twodmaxIndex])
                if np.isnan(varRev):
                    print ("NAN in VAR")
                    print  (sigmaData[:,evMaxIndex])
                    print ("ENDE")
                print ("variance = ", varRev, varRvv, vartwod,aSigma)
                print ("mean = " ,meanRev, meanrvv, meantwod,aSigma)
                # einpacken in ein array
                resultArray = np.array([aSigma,meanRev,varRev,meanrvv,varRvv,meantwod,vartwod])
                correlationVsSigmaArray = np.vstack((correlationVsSigmaArray, resultArray)) 
            print (correlationVsSigmaArray.shape)
            if max < np.amax(correlationVsSigmaArray[:,1])*1.2:
                max = np.amax(correlationVsSigmaArray[:,1])*1.2
            #axs[2,0].plot(vosigmaConstArray[:,sigmaIndex]*9.871, vosigmaConstArray[:,evMaxIndex], 'ro')
            axs[0,0].errorbar(correlationVsSigmaArray[:,0]*9.871,correlationVsSigmaArray[:,1],yerr=correlationVsSigmaArray[:,2], marker='s',
          ms=20, mew=4,linestyle='None',label =str(aVmaxSigma))
            axs[0,0].set_title("$r_{ev}$")
            axs[0,0].set_xlabel('sigma')
            axs[0,0].set_ylabel('$r_{ev}$')
            axs[0,0].legend(numpoints=1)
            axs[0,0].set_ylim(bottom=0)
            axs[0,0].set_ylim(top=max)
           # axs[2,1].plot(vosigmaConstArray[:,sigmaIndex]*9.871, vosigmaConstArray[:,vvMaxIndex], 'bo')  
            axs[0,1].errorbar(correlationVsSigmaArray[:,0]*9.871,correlationVsSigmaArray[:,3],yerr=correlationVsSigmaArray[:,4], marker='s',
          ms=20, mew=4,linestyle='None',label=str(aVmaxSigma))
            axs[0,1].set_title("$r_{vv}$")
            axs[0,1].set_ylim(bottom=0)
            axs[0,1].set_ylim(top = max)
            axs[0,1].legend(numpoints=1)
            axs[0,1].set_ylabel('$r_{vv}$')
            axs[1,0].errorbar(correlationVsSigmaArray[:,5]*9.871/100, correlationVsSigmaArray[:,1],xerr=correlationVsSigmaArray[:,6], yerr=correlationVsSigmaArray[:,2],  marker='s',
          ms=20, mew=4,linestyle='None',label=str(aVmaxSigma))
            axs[1,0].set_ylim(bottom=0)
            axs[1,0].set_ylim(top = max)
            axs[1,0].set_xlim(left=0)
            axs[1,0].legend(numpoints=1)
            axs[1,0].set_title("$r_{ev}(l_{corr}^{2d})$")
        else:
            continue  
        labelString = str(aVmaxSigma)
      #  axs[1,0].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,evMaxIndex], '*', label = labelString)
        axs[1,1].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,vvMaxIndex], '+', label = labelString)
    plt.legend()
   
    plt.show()
    #sys.exit()
    
# gnuplot indices 
NeIndex = 0  #1
NmIndex = 1 #2
interactionIndex = 3 #4 
impCountIndex = 4  #5
VmaxIndex = 5 #6
VminIndex = 6  #7
sigmaIndex = 7 #8
lcorrIndex = 8  #9
twodcorrelationx = 9 #10
twodcorrelationy = 10 #11
evMaxIndex = 11 # 12
vvMaxIndex = 12 # 13
gsIndex = 13  #14
varIndex = 13
gapStateIndex = 16  #17
twodmaxIndex = 19
splittingIndex = 20

fileName = "CompleteData_5_6_imps_multipleCosigmas.dat"
#fileName = "testResults_new.dat"
#fileName = "testResults_valid_5-15.dat"

#fileName = "testREsults_6_18_hc_vosigma66e-5.dat"
allData = np.genfromtxt(fileName, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    
print  (allData.shape)
extraColumn = addMaxtwoColumn(allData)

splitColumn = calcSplittingCoefficent(allData)
print ("All shapes")
print (extraColumn.shape)
print (splitColumn.shape)
allData = np.append(allData,extraColumn,axis=1)
allData=np.append(allData, splitColumn, axis=1)
print (allData.shape)
print ("Erste Zeila nach dem hinzufuegen des splittings")
print (allData[0,:])
print (allData[0,20])
print (allData[0,twodcorrelationx], allData[0,twodcorrelationy], allData[0,twodmaxIndex])
nelist = np.unique(allData[:,0])
print (nelist)
ialist = [1,0]
#showNonsenseValues(allData)
#sys.exit()
#print (vmaxsigmalist)
ialist = np.unique(allData[:,interactionIndex])
data = getVarianceData(allData)
HCArray = allData[np.where(allData[:,interactionIndex] == 1)]
HCArray = HCArray[np.where(HCArray[:,impCountIndex] == 4800)]
print (HCArray.shape)
vosigma = HCArray[:,VmaxIndex]*HCArray[:,sigmaIndex]
vmaxsigmalist = np.unique(vosigma.round(decimals = 6))
print (vmaxsigmalist)
searcehdArray = HCArray[np.where(np.abs(HCArray[:,VmaxIndex]*HCArray[:,sigmaIndex] - 4e-05) < 1e-08)]
sigmaList = np.unique(searcehdArray[:,sigmaIndex])
print (sigmaList)

#plotrevVsgap(data,6,1)
plotrevOvervariancefixLcorr(allData, 6, 1, [0.2,0.5,1.0,1.5,2.0],0.2)
plotrevOverLcorrRaw(allData, 6, 1, [1.3e-05, 5e-05], True, 'E')
gnuplotComparisonPlot(allData,6,1)
plotrevOverLcorr(allData, 6, 1, True, 'E')



sys.exit()

plotrevOverLcorr(allData,6,1,True,'E')
plotrevOverLcorr(allData,6,0,True,'E')
revVsLcorrIacomparison(allData,6, (1.3e-05,3e-05))
plotrevOverLcorr(allData,6,1,True)
plotrevOverLcorr(allData, 6, 0,True)
plotrevOverLcorr(allData, 6, 0,False)
plotrevVsVariance(data,6,1,True)
plotrevVsVariance(data,6,1,False)

#plotrevVsVariance(data,6,0,True)
#plotrevVsVariance(data,6,0,False)
#gnuplotComparisonPlot(allData,5,1)
#plotlCorrOnetwoD(allData,6)


sys.exit()
gnuplotComparisonPlot(allData,6,0,False)

gnuplotComparisonPlot(allData,5,0)
gnuplotComparisonPlot(allData,5,1)
plotrevVsLcorr(allData,6,1)
plotrevVsLcorr(allData,6,0)
sys.exit()
plotLcorrOverGap(allData,6,0)
plotLcorrOverGap(allData,6,1)
gnuplotComparisonPlot(allData,6,0)


sys.exit()
for ne in nelist:
    print(ne)
    electronNumberDataArray =allData[allData[:,0]==ne]
    for ia in ialist:
        print (ia)
        subDataArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==ia]
        plt.plot(subDataArray[:,8],subDataArray[:,evMaxIndex], 'o')
        plt.show()
        vosigma = subDataArray[:,VmaxIndex]*subDataArray[:,sigmaIndex]
        vmaxsigmalist = np.unique(vosigma.round(decimals = 6))  
        print (vmaxsigmalist)
        print (subDataArray[0,:])
        #print (subDataArray[1223,:])
        
        plotrevvvOverlcorrAllVoSigma(subDataArray, vmaxsigmalist)
        continue
        #calculateMeanFormVmaxSigmaList(vmaxsigmalist, subDataArray, (evMaxIndex,vvMaxIndex), True)
        #print (subDataArray)
        #print ("====================================================")
        plt.plot(subDataArray[:,VmaxIndex]*subDataArray[:,sigmaIndex],subDataArray[:,lcorrIndex], "ro")
        plt.plot(subDataArray[:,VmaxIndex]*subDataArray[:,sigmaIndex], subDataArray[:,twodcorrelationx],"bo")
        plt.title(str(int(ne))+" electrons, interaction = "+ str(int(ia)))
        plt.show()
        plt.plot(subDataArray[:,gsIndex]-subDataArray[:,gapStateIndex],subDataArray[:,lcorrIndex], "ro")
        plt.title(str(ne)+" electrons")
        plt.show()
