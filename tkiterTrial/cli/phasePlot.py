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

from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from scipy.optimize import curve_fit
from array import array
from random import randrange
from docutils.languages import da

# from cli.clusterEvaluation import allData

rc('font', **{'family':'sans-serif', 'sans-serif':['Helvetica']})
# # for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# berechne den mittelwwet und die Varianz der Spalte index in data
# gib den Mittelwert und die Varianz zurueck
def calculateMeanAndVarianceSingleEntity(data, index):
    tempArray = []
    if data.size == 0:
        return
    print (data.shape)
    for aDatum in data:
        tempVal = aDatum[index]
        if np.isnan(tempVal):
            continue
        tempArray = np.append(tempArray, tempVal)
    
    tempArray = np.asarray(tempArray)
    mean = np.mean(tempArray)
    var = np.var(tempArray)
    if np.isnan(mean) or np.isnan(var):
        return
    return (mean, var)


# Berechne varainz und mittelwert in den Spalten in indexlist von data  
# Gibt ein Array der Form [[mean1, var1],[mean2,var2],...] zurueck
#
def calculateMeanAndVarianceListOfEntities(data, indexlist):
    retArray = []
    for index in indexlist:
        (mean, var) = calculateMeanAndVarianceSingleEntity(data, index)
        retArray = np.hstack((retArray, [mean, var]))
    return retArray


def calculateMeanFormVmaxSigmaList(vosigmaList, data, indexList, plot=True):
    # indices to be meaned and aard given in indexList,
    # 
    indexSize = int(len(indexList)) * 2 + 1
    resultArray = np.empty([0, indexSize])
    exVoSigmaList = []
    print (resultArray.shape)
    for vosigma in vosigmaList:
        vosigmaConstArray = data[np.where(np.abs(data[:, VmaxIndex] * data[:, sigmaIndex] - vosigma) < 1e-07)]
        if vosigmaConstArray.size == 0:
         
            continue
        exVoSigmaList.append(vosigma)
        retVal = calculateMeanAndVarianceListOfEntities(vosigmaConstArray, indexList)
        retVal = np.append(retVal, vosigma)
        print (retVal.shape)
        print (resultArray.shape)
        resultArray = np.vstack((resultArray, retVal))
    if plot:
        plt.plot(exVoSigmaList, resultArray[:, 0], "ro")
        plt.xlabel("$V_{max}\sigma$")
        plt.ylabel("$r_{ev}$")
        plt.show()


def plotrevvvOverlcorrAllVoSigma(data, vosigmaList):
    
    print (data.shape)
    for vosigma in vosigmaList:
        vosigmaConstArray = data[np.where(np.abs(data[:, VmaxIndex] * data[:, sigmaIndex] - vosigma) < 1e-09)]
        print (vosigmaConstArray.shape)
        if vosigmaConstArray.size == 0:
         
            continue
        titleString = "V  sigma = " + str(vosigma)
        plt.plot(vosigmaConstArray[:, lcorrIndex], vosigmaConstArray[:, evMaxIndex], "ro", label=titleString)
        
        plt.xlabel("l_{corr}")
        plt.ylabel("r_{ev}")
        plt.legend(titleString)
    plt.show()


def addMaxtwoColumn(data):
    maxArray = []
    for aData in data:
        
        if aData[twodcorrelationx] > aData[twodcorrelationy]:
            maxArray = np.append(maxArray, aData[twodcorrelationx])
        else:
            maxArray = np.append(maxArray, aData[twodcorrelationy])
    maxArray = np.expand_dims(maxArray, -1)
    print (maxArray.shape)
    return (maxArray)


def plotLcorrOverGap(data, Ne, ia):
    
    # find max and min value of gap
    lcorrArray = np.array([])
    gapMin = np.min(data[:, gapStateIndex] - data[:, gsIndex])
    gapMax = np.max(data[:, gapStateIndex] - data[:, gsIndex])
    deltaGap = (gapMax - gapMin) / 10.0
    print (gapMin, gapMax, deltaGap)
    figs, axs = plt.subplots(2, 1)
    titleString = " Electrons"
    titleString = str(Ne) + titleString
    if ia == 0:
        titleString = titleString + " Coulomb interaction"
    else:
        titleString = titleString + " SRI"
    plt.suptitle(titleString)
    axs[0, 0].plot(data[:, gapStateIndex] - data[:, gsIndex], data[:, evMaxIndex], 'ro')
    for aGap in np.linspace(gapMin, gapMax, 10):
        print (aGap)
        # get subarray data
        lines = data[np.where((np.abs(data[:, gapStateIndex] - data[:, gsIndex]) - aGap) < deltaGap)]
        lcorravg = np.nanmean(lines[:, evMaxIndex])
        lcorrVar = np.nanvar(lines[:, evMaxIndex])
        lcorrArray = np.append(lcorrArray, lcorravg)
    axs[0, 0].plot(np.linspace(gapMin, gapMax, 10), lcorrArray, 'g+', ms=20)
    plt.show()     


def getScaleFactor(Ne):
    retval = np.sqrt(Ne * 3 * 2 * np.pi) / 1000.0 
    return retval


def getUnitcellSizeinLo(Ne):
    return np.sqrt(Ne * 3 * 2 * np.pi)


def plotrevVsLcorr(data, Ne, ia):
    
    if len(ia) > 1:    
        # beide ww typen
        print (ia)
        if len(Ne) > 1:
            # wenn auch alle elec Typen: 2x2
            figs, axs = plt.subplots(2, 2)
        else:
        # sonst 2x1
            figs, axs = plt.subplots(2, 1)
    else:
        if len(Ne) > 1:
            figs, axs = plt.subplots(2, 1)
        else:
            figs, axs = plt.plot()
    
    titleString = " Electrons"
    titleString = str(Ne) + titleString
    if ia == 0:
        titleString = titleString + " Coulomb interaction"
    else:
        titleString = titleString + " SRI"
    plt.suptitle(titleString)
    plt.plot(data[:, lcorrIndex] / getUnitcellSizeinLo(Ne), data[:, evMaxIndex] * getScaleFactor(Ne), 'ro')
    plt.show()

    
def expfunc(x, a, c, d):
    return a * np.exp(-c * x) + d   

def createLegend(popt,pcov):
    a = round(popt[0],2)
    c = round(popt[1],3)
    d = round(popt[2],2)
    retVal = "$"+str(a)+"e^{"+str(c)+" \pm " + str(round(pcov[1,1],4))+"l_{corr}} + " +str(d)+"$"
    return retVal
def sqrtfunc(x,a,b,c):
    return a*np.sqrt(b*x)+c




def get_color():
        
    for item in ['r +', 'g +', 'b +', 'c +', 'm +', 'y +', 'k +', '#eeefff +','r s', 'g s', 'b s', 'c s', 'm s', 'y s', 'k s', '#eeefff s','r s', 'g s', 'b s', 'c s', 'm s', 'y s', 'k s', '#eeefff s', 'last']:
        
        yield (item)

        
def get_color_and_symbol():
    all = get_color()
    Atring = next(all)
#    Atring = all.join()
    print (Atring)
    return (all.split())

    
def findBestFitParameters(data):

    sigma = data[:, sigmaIndex]
    sigmaList = np.unique(sigma.round(decimals=7))
    lcorrArray = data[:,lcorrIndex]
    lcorrMin = np.amin(lcorrArray)
    lcorrMax = np.amax(lcorrArray)
    print ("FindBestFit")
    print (lcorrMin, lcorrMax) 
    resArray=np.empty((0,3))
    for binCount in range(20,150,10):
        lcorrSpace = np.linspace(lcorrMin,lcorrMax,binCount)
        #print (lcorrSpace)
        meanArray = np.empty((0,2))
        for index in range(0,binCount-1):
            min = lcorrSpace[index]
            max =lcorrSpace[index+1]
            #print ("Min,Max")
            #print (min,max)
            partArray = data[np.where(data[:,lcorrIndex] > min)]
            partArray = partArray[np.where(partArray[:,lcorrIndex] < max)]

            #print (len(partArray))
            lcorrMean = np.nanmean(partArray[:,lcorrIndex])
            evMaxMean = np.nanmean(partArray[:,evMaxIndex])
            if np.isnan(lcorrMean) or np.isnan(evMaxMean):
                continue
            tArray = [lcorrMean, evMaxMean]
            #print (tArray)
            meanArray = np.vstack((meanArray,tArray))
            #print (meanArray)
        popt, pcov = curve_fit(expfunc, meanArray[:,0], meanArray[:, 1], p0=(1, 1e-6, 1), maxfev=3000)
        print ("first fit successfull, popt")
        print (binCount, popt[1], pcov[1][1])
        aTArray = [binCount, popt[1],pcov[1][1]]
        resArray = np.vstack((resArray, aTArray))
    print ("FitBestEnde")
    print(resArray)
    minIndex = np.argmin(resArray[:,2])
    print (minIndex)
    print (resArray[minIndex,0])
    return (int(resArray[minIndex,0]))
def threedPlot(data,Ne,ia):
    if Ne == 6: 
        theData = allData[allData[:, impCountIndex] == 4800]
    unitCellSize = np.sqrt(2 * np.pi * 3 * Ne)
    
    electronNumberDataArray = theData[theData[:, 0] == Ne]  
    
    HCArray = electronNumberDataArray[electronNumberDataArray[:, interactionIndex] == ia]
    HCArray = HCArray[np.where(np.abs(HCArray[:,13])<0.00001)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(HCArray[:,lcorrIndex], HCArray[:,13], HCArray[:,evMaxIndex])
    plt.xlabel("Lcorr")
    plt.ylabel("variance")
    plt.show()
    plt.scatter(HCArray[:,lcorrIndex]*unitCellSize/200.0, HCArray[:,13], c= HCArray[:,evMaxIndex]*unitCellSize/1000.0, s =20)
    plt.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
    plt.xlabel("$l_{corr} [l_{0}]$", fontsize=18.0)
    plt.ylabel("$var V(x,y)$", fontsize=18.0)
    plt.ylim(bottom=0)
    plt.ylim(top=1e-05)
    plt.xlim(left=0.0)
    cbar = plt.colorbar()
    cbar.set_label("$r_{ev}$", fontsize=14.0)
    plt.legend()
    plt.show()
                   

def plotrevOverLcorrRaw(data,Ne,ia):
    if Ne == 6: 
        theData = allData[allData[:, impCountIndex] == 4800]
    unitCellSize = np.sqrt(2 * np.pi * 3 * Ne)
    
    electronNumberDataArray = theData[theData[:, 0] == Ne]  
    HCArray = electronNumberDataArray[electronNumberDataArray[:, interactionIndex] == ia]
    if ia == 0:
        HCArray[:,evMaxIndex] = HCArray[:,evMaxIndex]-evMaxIndex_Coulomb
    binCount = findBestFitParameters(HCArray)
    sigma = HCArray[:, sigmaIndex]
    sigmaList = np.unique(sigma.round(decimals=7))
    lcorrArray = HCArray[:,lcorrIndex]
    lcorrMin = np.amin(lcorrArray)
    lcorrMax = np.amax(lcorrArray)
   
    print (lcorrMin, lcorrMax) 
    
    lcorrSpace = np.linspace(lcorrMin,lcorrMax,binCount)
    print (lcorrSpace)
    meanArray = np.empty((0,2))
    for index in range(0,binCount-1):
        min = lcorrSpace[index]
        max =lcorrSpace[index+1]
        print ("Min,Max")
        print (min,max)
        partArray = HCArray[np.where(HCArray[:,lcorrIndex] > min)]
        partArray = partArray[np.where(partArray[:,lcorrIndex] < max)]

        print (len(partArray))
        lcorrMean = np.nanmean(partArray[:,lcorrIndex])
        evMaxMean = np.nanmean(partArray[:,evMaxIndex])
        if np.isnan(lcorrMean) or np.isnan(evMaxMean):
            continue
        tArray = [lcorrMean, evMaxMean]
        print (tArray)
        meanArray = np.vstack((meanArray,tArray))
    print (meanArray)
    # fit to exponential
    popt, pcov = curve_fit(expfunc, meanArray[:,0], meanArray[:, 1], p0=(1, 1e-6, 1), maxfev=3000)
    print ("first fit successfull, popt")
    print (popt)
    print(pcov)
    popt2,pcov2 = curve_fit(expfunc,HCArray[:,lcorrIndex], HCArray[:,evMaxIndex], p0 = (1,0.1,1))
    print(popt2)
    fitX = np.linspace(0, lcorrMax,50)
    fitY = expfunc(fitX,popt[0],popt[1],popt[2])
   
    popt1, pcov1 = curve_fit(expfunc, meanArray[:,0]*unitCellSize/200.0, meanArray[:, 1]*unitCellSize/1000.0, p0=(1, 1e-6, 1), maxfev=3000)
    print ("Second fit, popt1, pcov1")
    print (popt1)
    print (pcov1)
    fitX1 = np.linspace(0, lcorrMax*unitCellSize/200.0,50)
    fitY1 = expfunc(fitX1, popt1[0], popt1[1], popt1[2])
    fitY2= expfunc(fitX, popt2[0],popt2[1],popt2[2])
    popt3,pcov3 = curve_fit(expfunc, fitX*unitCellSize/200.0, fitY*unitCellSize/1000, p0=(1, 1e-6, 1), maxfev=3000)
    color_symbol= get_color()
    for sigma in sigmaList:
       

        sigmaConstArray = HCArray[np.where(np.abs(HCArray[:, sigmaIndex]  - sigma) < 1e-10)]
        labelString = str(sigma)
        aString1 =next(color_symbol) 
        print (aString1)
        symbol = aString1.split()[1]
        print (symbol)
        plt.plot(sigmaConstArray[:,lcorrIndex], sigmaConstArray[:,evMaxIndex], symbol,label=labelString)
    plt.plot(meanArray[:,0], meanArray[:,1], "+", label ="mEAN",color="red", ms = 20.0, mew=10.0 )
    plt.plot(fitX,fitY,label="Fit1")
    plt.plot(fitX,fitY2,label="Fit2")
    plt.legend()
    plt.show()
    labelString = createLegend(popt3,pcov3)
    print (labelString)
    plt.plot(HCArray[:,lcorrIndex]*unitCellSize/200.0, HCArray[:,evMaxIndex]*unitCellSize/1000.0, 'o',ms=10.0, label="Data")
    plt.plot(meanArray[:,0]*unitCellSize/200, meanArray[:,1]*unitCellSize/1000.0, "+", label ="Mean",color="red", ms = 20.0, mew=10.0)
    plt.plot(fitX*unitCellSize/200.0,fitY*unitCellSize/1000.0,label=labelString, lw = 4.0)
    plt.xlabel('$l_{corr} [l_{0}]$', fontsize=18.0)
    plt.ylabel('$r_{ev}[l_{0}]$', fontsize=18.0)
    plt.legend()
    plt.savefig("test.eps", format="eps")

    plt.show()
    # fit again to find "real data"
   
    print (popt3, popt)
def plotrevOverLcorr(data, Ne, ia, maximum=False):
    # plot Ne = 5 , Interaction = HC, all points

    # only fitting impurities
    offset = 0.0
    if Ne == 5:
        theData = allData[allData[:, impCountIndex] == 4000]
        
    if Ne == 6: 
        theData = allData[allData[:, impCountIndex] == 4800]
    unitCellSize = np.sqrt(2 * np.pi * 3 * Ne)
    # theData.vstack
    electronNumberDataArray = theData[theData[:, 0] == Ne]  
    HCArray = electronNumberDataArray[electronNumberDataArray[:, interactionIndex] == ia]
    vosigma = electronNumberDataArray[:, sigmaIndex] * electronNumberDataArray[:, VmaxIndex]
    vosigmalList = np.unique(vosigma.round(decimals=7))
    print (vosigmalList)
   # figs,axs = plt.subplots(2,1)
    titleString = " Electrons"
    titleString = str(Ne) + titleString
    if ia == 0:
        titleString = titleString + " Coulomb interaction"
        offset = 0.6
    else:
        titleString = titleString + " SRI"
    plt.suptitle(titleString)   
    (color,symbol) = get_color_and_symbol() 
   # axs[0,0].plot(HCArray[:,lcorrIndex], HCArray[:,evMaxIndex], 'o')
   # axs[0,1].plot(HCArray[:,lcorrIndex], HCArray[:,vvMaxIndex]*0.5,'+')
    for aVmaxSigma in vosigmalList:
        vosigmaConstArray = HCArray[np.where(np.abs(HCArray[:, VmaxIndex] * HCArray[:, sigmaIndex] - aVmaxSigma) < 1e-10)]
        sigmaList = np.unique(vosigmaConstArray[:, sigmaIndex])
        print (sigmaList)
       
        if len(sigmaList) > 3:
            correlationVsSigmaArray = np.empty((0, 9))
            max = 0.0
            for aSigma in sigmaList:
                sigmaData = vosigmaConstArray[vosigmaConstArray[:, sigmaIndex] == aSigma]
                print (sigmaData.shape)
                meanRev = np.nanmean(sigmaData[:, evMaxIndex]) * unitCellSize / 1000 - offset
                # meancorrMax = np.mean(np.maximum(sigmaData[:,twodcorrelationx],sigmaData[:twodcorrelationy]))
               # print (meancorrMax)
                meanrvv = np.nanmean(sigmaData[:, vvMaxIndex]) * 0.5
                varRev = np.sqrt(np.nanvar(sigmaData[:, evMaxIndex] * unitCellSize / 1000, dtype=np.float64))
                varRvv = np.sqrt(np.nanvar(sigmaData[:, vvMaxIndex] * unitCellSize / 100, dtype=np.float64))
                meantwod = np.nanmean(sigmaData[:, twodmaxIndex]) * unitCellSize / 200
                vartwod = np.nanvar(sigmaData[:, twodmaxIndex]) * unitCellSize / 200
                meanLcorr = np.nanmean(sigmaData[:, lcorrIndex] * unitCellSize / 200)
               
                # meanLcorr2D = np.nanmean(np.amax(sigmaData[0:[]]))
                varLcorr = np.nanvar(sigmaData[:, lcorrIndex] * unitCellSize / 200)
                print ("LCorr 1D vs 2D", meanLcorr, varLcorr, meantwod, vartwod)
                if np.isnan(varRev):
                    print ("NAN in VAR")
                    print  (sigmaData[:, evMaxIndex])
                    print ("ENDE")
                print ("variance = ", varRev, varRvv, vartwod, aSigma)
                print ("mean = " , meanRev, meanrvv, meantwod, aSigma)
                # einpacken in ein array
                resultArray = np.array([aSigma, meanRev, varRev, meanrvv, varRvv, meantwod, vartwod, meanLcorr, varLcorr])
                correlationVsSigmaArray = np.vstack((correlationVsSigmaArray, resultArray)) 
            z = np.polyfit(correlationVsSigmaArray[:, 7], correlationVsSigmaArray[:, 1], 3)
            p = np.poly1d(z)
            fitX = np.linspace(0, np.max(correlationVsSigmaArray[:, 7]) * 1.2, 20)
            fitData = p(fitX)
            print (z)
            acolor = next(color)
            plt.plot(fitX, fitData, lw=3, color=acolor)
            print (correlationVsSigmaArray.shape)
            if max < np.amax(correlationVsSigmaArray[:, 1]) * 1.2:
                max = np.amax(correlationVsSigmaArray[:, 1]) * 1.2
            # axs[2,0].plot(vosigmaConstArray[:,sigmaIndex]*9.871, vosigmaConstArray[:,evMaxIndex], 'ro')

            if maximum == False:
                plt.errorbar(correlationVsSigmaArray[:, 7], correlationVsSigmaArray[:, 1], xerr=np.sqrt(correlationVsSigmaArray[:, 8]), yerr=np.sqrt(correlationVsSigmaArray[:, 2]), marker='s',
          ms=20, mew=4, color=acolor, linestyle='None', label=str(aVmaxSigma))
            else:
                plt.errorbar(correlationVsSigmaArray[:, 5], correlationVsSigmaArray[:, 1], xerr=np.sqrt(correlationVsSigmaArray[:, 6]), yerr=np.sqrt(correlationVsSigmaArray[:, 2]), marker='s',
          ms=20, mew=4, color=acolor, linestyle='None', label=str(aVmaxSigma))
                
           # plt.plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray[:,evMaxIndex], )
            
            plt.xlabel('$l_{corr} [l_{0}]$', fontsize=16)
            if ia == 0:
                plt.ylabel('$\Delta r_{ev}r_{ev}^{homo}[l_{0}]$', fontsize=16)
            else:
                plt.ylabel('$r_{ev}[l_{0}]$', fontsize=16)
            plt.legend(numpoints=1)
            plt.ylim(bottom=0)
            plt.ylim(top=max)
            plt.title(titleString)
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

def plotDataAndExponentialFit(data, Ne, ia, maximum=False):
    # plot Ne = 5 , Interaction = HC, all points

    # only fitting impurities
    offset = 0.0
    if Ne == 5:
        theData = allData[allData[:, impCountIndex] == 4000]
        
    if Ne == 6: 
        theData = allData[allData[:, impCountIndex] == 4800]
    unitCellSize = np.sqrt(2 * np.pi * 3 * Ne)
    # theData.vstack
    electronNumberDataArray = theData[theData[:, 0] == Ne]  
    HCArray = electronNumberDataArray[electronNumberDataArray[:, interactionIndex] == ia]
    if ia == 0:
        HCArray = HCArray[np.where(HCArray[:,evMaxIndex]>10)]
    fitArray = HCArray[~np.isnan(HCArray[:, lcorrIndex])]
    fitArray = fitArray[~np.isnan(fitArray[:, evMaxIndex])]
    # fitArray = fitArray[np.where(fitArray[:,evMaxIndex]>2)]
    firstFitSuccess = True
    try:
        popt, pcov = curve_fit(expfunc, fitArray[:, lcorrIndex]* unitCellSize / 200, fitArray[:, evMaxIndex]* unitCellSize / 1000, p0=(1, 1e-6, 1), maxfev=3000)
        print ("first fit successfull, popt")
        print (popt)
    except RuntimeError as e:
        print ("First fit failed")
        print (e)
        print (popt)
        firstFitSuccess = False
    popt1, pcov1 = curve_fit(expfunc, fitArray[:, lcorrIndex], fitArray[:, evMaxIndex], p0=(1, 1e-6, 1))
    print("Result of second fit vor umskalierung")
    print(popt1)
    print("Result of second fit nach umskalierung")
    popt1[0] =popt1[0]* unitCellSize / 200
    popt1[1] = popt1[1]* unitCellSize / 1000
    print (popt1)
    legendString="EMPTY"
    if firstFitSuccess == True:
        print (popt)
        print (pcov)
        legendString = createLegend(popt,pcov)
    else:
        legendString = createLegend(popt1,pcov1)
        popt = popt1
    print ("POPT =============")
    print (pcov1)
    print("PCOV ==============")
    #legendString = createLegend(popt,pcov)
    print (legendString)
    expxData = np.linspace(0.0, np.max(fitArray[:, lcorrIndex])*unitCellSize/200, 100)
    expyData = expfunc(expxData, *popt)
    #print (popt[:])
    newSigma = popt[1] * unitCellSize / 200.0
    newSigma = round(newSigma, 3)
    # plot(expxData,expyData,label="exp. Fit", lw =4)
    
      #  axs[1,0].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,evMaxIndex], '*', label = labelString)
        # axs[1,1].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,vvMaxIndex], '+', label = labelString)
        
    plot(fitArray[:, lcorrIndex] * unitCellSize / 200, fitArray[:, evMaxIndex] * unitCellSize / 1000, 'ro' )
    plot(expxData , expyData , label=legendString, lw=4)
    plt.xlabel('$l_{corr} [l_{0}]$', fontsize=16)
    if ia == 0:
        plt.ylabel('$\Delta r_{ev}r_{ev}^{homo}[l_{0}]$', fontsize=16)
        titleString ="Coulomb"
    else:
        plt.ylabel('$r_{ev}[l_{0}]$', fontsize=16)
        titleString = "SRI"
    
      #  axs[1,0].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,evMaxIndex], '*', label = labelString)
        # axs[1,1].plot(vosigmaConstArray[:,lcorrIndex], vosigmaConstArray [:,vvMaxIndex], '+', label = labelString)
    plt.title(titleString)
    plt.legend()
    plt.show()
    # sys.exit()



    



def revOverLcorrConstGap(data, Ne, ia):
    
    return


def plotlCorrOnetwoD(data, Ne):
    if Ne == 5:
        theData = allData[allData[:, impCountIndex] == 4000]
        
    if Ne == 6: 
        theData = allData[allData[:, impCountIndex] == 4800]
    
    print (np.amax(theData[:, twodcorrelationx]), np.amax(theData[:, twodcorrelationy]))
    maxYindex = np.argmax(theData[:, twodcorrelationy])
    print (theData[maxYindex, :])
    plt.plot(theData[:, lcorrIndex], theData[:, twodcorrelationx], label='2Dx')
    plt.plot(theData[:, lcorrIndex], theData[:, twodcorrelationy], label="2dy")
    plt.legend()
    plt.show()
    return


def showNonsenseValues(data):
    # lcorr too small or too large
    print ("One line")
    print (data[1, :])
    strippedData = np.delete(data, 2, axis=1)
    print (strippedData[1, :])
    bogusData = strippedData[np.where(strippedData[:, lcorrIndex] < 1.0)]
    print (bogusData)
    
    # everything with Nan in it
    print(strippedData[np.isnan(strippedData[:[NeIndex]]).any(axis=1)])
    print (len(strippedData))
    print (len(strippedData[np.isnan(strippedData).any(axis=1)]))
    
    return


def getVarianceData(data):
    retVal = data[np.abs(data[:, varIndex]) < 0.001 ]
    return retVal







def calculateAverage(data, index):
    
    length = len(data)
    value = 0.0
    count = 0
    print ("calcualte average, anzahl der Daten", length)
    for row in data:
        if row[index] > 1:
            value = value + row[index]
        
            count = count + 1
    return value / count




    
def checkData(data):
    badDataCount = 0
    for row in data:
        print (row.size)
        if (row[NeIndex] > 5):
            continue
        if row[interactionIndex] == 0:
            homoGap = gap_homo_Coulomb_6_18
        else:
            homoGap = gap_homo_SRI_6_18_sri
        gap = row[gsIndex] - row[gapStateIndex]
        if (np.abs(gap) > 10 * homoGap):
            badDataCount = badDataCount + 1
    print (badDataCount)


# gnuplot indices
def checkFile(file):
    f = open(file, "r")
    linecount = 0
    line19count = 0
    line20count = 0
    line21count = 0
    otherlinecount = 0
    for line in f:
        array = line.split()
        elems = len(array)
        linecount = linecount + 1
        if  (elems == 19):
            line19count = line19count + 1
            continue
        if (elems == 20):
            line20count = line20count + 1
            continue
        if (elems == 21):
            line21count = line21count + 1
            continue
        otherlinecount = otherlinecount + 1
    

    return 


def createPhaseDiagram(data, Ne, ia, index = 8):
    if Ne == 5:
        theData = allData[allData[:, impCountIndex] == 4000]
        
    if Ne == 6: 
        theData = allData[allData[:, impCountIndex] == 4800]
    
    unitCellSize = np.sqrt(2 * np.pi * 3 * Ne)
    results = np.empty((0,4))
    electronNumberDataArray = theData[theData[:, 0] == Ne]  
    iaArray = electronNumberDataArray[electronNumberDataArray[:, interactionIndex] == ia]

    maxLcorr = np.amax(iaArray[:,index])
    minLcorr = np.amin(iaArray[:,index])
    maxVar = 5e-05
    minVar = 0.0
    binCount=50
    print ("Lcorr", maxLcorr, minLcorr)
    print ("Variance", maxVar, minVar)
    
    varArray, varInterval = np.linspace(minVar,maxVar,binCount,retstep=True)
    lcorrArray,  lcorrInterval = np.linspace(minLcorr,maxLcorr,binCount,retstep=True)
    lastLCorr = -1
    lastVar = -1
    f = open("neededData.dat","w+")
    f.write("# variance   lcorr  varInterval, lcorrInterval, last sigma last Vmax \n")
    for lcorr in lcorrArray:
        for aVariance in varArray:
        
            print (aVariance, lcorr)
            # find data within the array
            theData = iaArray[np.where(np.logical_and(iaArray[:,index]>lcorr, iaArray[:,index]<lcorr+lcorrInterval))]
            print ("Thedata", theData.shape)
            boxedInData = theData[np.where(np.logical_and(theData[:,varIndex]>aVariance, theData[:,varIndex]<aVariance+varInterval))]
            num_rows,num_cols = boxedInData.shape
            if num_rows == 0:
                partResult = (aVariance,lcorr,-10,0)
                print ("No data for ", aVariance, lcorr)
                print ("Last data from", lastVar, lastLCorr)
                if lastLCorr > -1.0:
                    #print (lastData)
                    f.write(str(aVariance))
                    f.write(" ")
                    f.write(str(varInterval))
                    f.write(" ")
                    f.write(str(lcorr))
                    f.write(" ")
                    f.write(str(lcorrInterval))
                    f.write(" ")
                    f.write (str(lastData[sigmaIndex]))
                    f.write( "  ")
                    f.write(str(lastData[VmaxIndex]))
                    f.write("\n")
                print ("===========================")
#               continue
            else:
                
                revAverage = np.nanmean(boxedInData[:,evMaxIndex])
                points,cols = boxedInData.shape
                print ("Points ",points)
                lastData = boxedInData[0,:]
                lastVar = aVariance
                lastLCorr = lcorr
                partResult = (aVariance,lcorr,revAverage,points)
            results = np.vstack((results,partResult))
        print (results)
    f.close()
    print (np.amax(results[:,0]))
    nzPoints = 0
    sum =0
    for aRow in results:
        if aRow[3]>0.0:
            nzPoints = nzPoints+1
            sum = sum +  aRow[3]

    print ("Maximal Zahl von Points",nzPoints)
    
    print ("Average",)
    print (sum/nzPoints)
    print ("Number of nonzero entries",nzPoints)

    cmap, norm = mpl.colors.from_levels_and_colors([-10, 0, 30, 50, 150], ['lightgrey', 'green', 'yellow','red'])
    print (norm)
    print ("No data available")
    print (results[np.where(results[:,2] < 0)])
    plt.scatter(results[:,1]* unitCellSize / 200, results[:,0], c=results[:,2], cmap=cmap, norm=norm,marker= 's', s=80.0)
#    plt.scatter(results[:,1], results[:,0], c=results[:,2], cmap=cmap, norm=norm,marker= 's', s=80.0)
    anot = plt.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    anot.set_visible(True)
    plt.ylim(bottom=minVar-0.000001, top=maxVar)
    plt.xlim(left=0.0)
    plt.xlabel('$l_{corr} [l_{0}]$', fontsize=16)
    plt.ylabel('$ var (V(r)$', fontsize=16)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(['no Data','FQH-state','transition regime','insulating state'])
    plt.draw()
    plt.savefig("phasePlot.eps", format='eps')
    plt.show()

    #plt.scatter(results[:,0],results[:,1]* unitCellSize / 200, c=results[:,3])
    plt.hist(results[:,3])
    #plt.xlim(left=minVar-0.000005, right=maxVar)
    #plt.colorbar()
    plt.show()
    intermediateLow = 30
    intermediateTop = 60
    
    mdata = theData[np.where(np.logical_and(theData[:,index]>intermediateLow, theData[:,index]<intermediateTop))]
    plt.plot(mdata[:,varIndex], mdata[:,lcorrIndex], 'o')
    plt.show()
    print ("Missing data")
    dataSet1 = iaArray[np.where(np.abs(iaArray[:,index] -4) < 2)]
    dataSet1 = dataSet1[np.where(np.abs(dataSet1[:,varIndex]-1e-06)< 2e-06)]
    print ("Sigma, vmax, lcorr, variance")
    print (dataSet1[:,[sigmaIndex,VmaxIndex, index,varIndex]])
    print (np.unique(dataSet1[:,sigmaIndex]))

def plot_rev_over_bandwith(data, Ne, ia):
    print ("Start rev_over_bandwidth")
    print (allData.shape)
    print (allData[0,4])
    
    if Ne == 5:
        theData = allData[allData[:, impCountIndex] == 4000]
        
    if Ne == 6: 
        print("6")
        theData = allData[allData[:, impCountIndex] == 4800]
    
    print (theData.shape)
    
    
    electronNumberDataArray =theData[theData[:,0]==Ne] 
    HCArray = electronNumberDataArray[electronNumberDataArray[:,interactionIndex]==1]
    unitCellSize = np.sqrt(2 * np.pi * 3 * Ne)
    # Filter data were rev < 2, print bandwidth
    smallrevArray = HCArray[np.where(HCArray[:,evMaxIndex] < 2)]
    number_of_rows = smallrevArray.shape[0]
    random_indices = np.random.choice(number_of_rows, size=5, replace=False)
     
    printDataReadable(smallrevArray[random_indices,:])
    #print (smallrevArray[0,:])
    #print ("Selected values, evMax, vvMax, lcorr, 2 states")
    #print (smallrevArray[:,[evMaxIndex, vvMaxIndex,lcorrIndex,14,16]])
    #print ("---------------------------------------------------")
    #print (smallrevArray[0,[evMaxIndex, vvMaxIndex,lcorrIndex,14,16]])
    #print(smallrevArray[:,evMaxIndex])
    print ("End of small rev")
    largerevArray = HCArray[np.where(HCArray[:,evMaxIndex]> 100)]
    number_of_rows = largerevArray.shape[0]
    random_indices = np.random.choice(number_of_rows, size=5, replace=False)
    printDataReadable(largerevArray[random_indices,:])
    print ("End of large rev")
    revMax = np.max(HCArray[:,evMaxIndex])
    revmaxPos = np.argmax(HCArray[:,evMaxIndex])
    print ("max of rev", revMax, revmaxPos)
    
    bandwidth = HCArray[:,16]-HCArray[:,14] # teilen durch wurzel aus varianz (oder so) LandauLevelbreite (1 Teilchen)
    print(HCArray[revmaxPos,:])
    print ("Max bandwidth " + str(np.max(bandwidth)))
    maxIndex = np.argmax(bandwidth)
    print (maxIndex)
    print (HCArray[maxIndex,:])
    print ("Min bandwidth " + str(np.min(bandwidth)))
    print (HCArray[np.argmin(bandwidth)])
    print ("====================================")
    plt.scatter(bandwidth, HCArray[:,evMaxIndex])
    plt.xlabel('$\Delta E_{0,2}$', fontsize=16)
    plt.ylabel('$ r_{ev} []$', fontsize=16)
    plt.xlim(0,5e-04)
    plt.ylim(0,200)
    plt.show()
    plt.scatter(bandwidth, HCArray[:,13])
    plt.xlabel('$\Delta E_{0,2}$', fontsize=16)
    plt.ylabel('var V(r)')
    
    plt.show()



def createScatterbandwithvarlcorr(data, Ne, ia):
    print ("Start bandwidthscatter")
    print (allData.shape)
    print (allData[0,4])
    
    if Ne == 5:
        theData = allData[allData[:, impCountIndex] == 4000]
        
    if Ne == 6: 
        print("6")
        theData = allData[allData[:, impCountIndex] == 4800]
    
    print (theData.shape)
    unitCellSize = np.sqrt(2 * np.pi * 3 * Ne)
    results = np.empty((0,4))
    electronNumberDataArray = theData[theData[:, 0] == Ne]  
    iaArray = electronNumberDataArray[electronNumberDataArray[:, interactionIndex] == ia]
    # sinnvolle Werte sind welche mit einer varianxz > 0 und < 1 
    viatArray = iaArray[iaArray[:,13] < 1]
    viaArray = viatArray[np.where(np.abs(viatArray[:,16]-viatArray[:,14]) < 0.1)]
    print ("New array with reasonable bandwidth data")
    print(viaArray.shape)
    varArray = viaArray[:,13]
    print (iaArray.shape)
    maxLcorr = np.amax(viaArray[:,8])
    minLcorr = np.amin(viaArray[:,8])
    print (viaArray[0,10:18])  
    print ("Variance = " + str(iaArray[12,13]))
    print (viaArray[0,13])
    print (viaArray[0,14])
    print (viaArray[0,15])
    print (viaArray[0,16])
    print (viaArray[0,16] - viaArray[0,14]) 
    for i in range(10):
        
        index = randrange(0,1234)
        print ("Element " + str(index))
        print (viaArray[index,13:17])
        print ("Bandwidth")
        print (viaArray[index,16] - viaArray[index,14])
        print("variance")
        print(viaArray[index,13])
    print ("==== END DUMP ====")
    bandwidth = viaArray[:,16]-viaArray[:,14] # teilen durch wurzel aus varianz (oder so) LandauLevelbreite (1 Teilchen)
    
    print ("extrema cder bandwidth")
    print (np.max(bandwidth))
    print (np.min(bandwidth))
    rescaleBandwidth = bandwidth / np.sqrt(viaArray[:,13])
    #plt.scatter(iaArray[:,8]* unitCellSize / 200, iaArray[:,12], c=bandwidth,  norm=norm,marker= 's', s=80.0)
    plt.scatter(viaArray[:,8]* unitCellSize / 200, viaArray[:,13], c=rescaleBandwidth,marker= 's', s=10.0, vmin=0.0, vmax=0.01)   
    plt.ylim(bottom=0.0)
    plt.ylim(top=1e-5)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.xlabel('$l_{corr} [l_{0}]$', fontsize=16)
    plt.ylabel('$ var (V(r)$', fontsize=16)
    
    clb = plt.colorbar(label='$\Delta E_{0,2}$ over $ \sqrt{var(V(r))}$')
    plt.clim(0.0,5.0)
    plt.savefig("Scatterbandwithvarlcorr_6_18-sri.png")
    plt.show()         
    #plt.scatter(viaArray[:,8]* unitCellSize / 200, viaArray[:,13], c=rescaleBandwidth,marker= 's',s=10.0, vmin=0.0, vmax=0.01)   
    plt.scatter(viaArray[:,8]* unitCellSize / 200, viaArray[:,13], c=bandwidth,marker= 's',s=10.0, vmin=0.0, vmax=0.01)
    plt.ylim(bottom=0.0)
    plt.ylim(top=1e-5)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.xlabel('$l_{corr} [l_{0}]$', fontsize=16)
    plt.ylabel('$ var (V(r)$', fontsize=16)
    
    clb = plt.colorbar(label='$\Delta E_{0,2}$ over $\sqrt{var(V(r))}$')
    plt.clim(0.0,1)
    plt.savefig("Scatterrescalebandwithvarlcorr_6_18-sri_fine.png")
    plt.show()   
    
def printDataReadable(data):
    print ("Daten lesbar")
    print (data.shape)
    for dataset in data:
        print (dataset)
        print ('---------------------------------------')
        print ("Ne, Nm", dataset[[0,1]])
        print ("interaction,sigma, Vmax", dataset[[interactionIndex,sigmaIndex,VmaxIndex]])
        print ("lcorr, variance", dataset[[8,9,13]])
        print ("Vortexgroessen", dataset[[evMaxIndex, vvMaxIndex] ])
        print ("Spectrum und was davor= ", dataset[[12,13,14,15,16,17]])
        print ("Bandwidth = ", dataset[16]-dataset[14])
        print ("gap =", dataset[17]-dataset[14])
        
        print ('---------------------------------------')
              
NeIndex = 0  # 1
NmIndex = 1  # 2
interactionIndex = 3  # 4 
impCountIndex = 4  # 5
VmaxIndex = 5  # 6
VminIndex = 6  # 7
sigmaIndex = 7  # 8
lcorrIndex = 8  # 9
twodcorrelationx = 9  # 10
twodcorrelationy = 10  # 11
evMaxIndex = 11  # 12
vvMaxIndex = 12  # 13

gsIndex = 14  # 14
varIndex = 13
gapStateIndex = 17  # 17
twodmaxIndex = 19
gap_homo_SRI_6_18_sri = 0.195254103
gap_homo_Coulomb_6_18 = 0.063009445
evMaxIndex_Coulomb = 80
# fileName = "testREsults_6_18_coul.dat"
fileName = "CompleteData_5_6_imps_multipleCosigmas.dat"
#fileName = "temp.dat"
#fileName = "testResults_valid_5-15.dat"
#fileName ="testResults_valid_5-15.dat"
checkFile(fileName)

# fileName  = "results_all_cluster.dat"
# fileName = "testREsults_6_18_hc_vosigma66e-5.dat"
allData = np.genfromtxt(fileName, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
print (allData[0, :])
print  (allData.shape)

checkData(allData)
extraColumn = addMaxtwoColumn(allData)
print (extraColumn.shape)

allData = np.append(allData, extraColumn, axis=1)

print (allData.shape)
plot_rev_over_bandwith(allData, 6,1)
createScatterbandwithvarlcorr(allData, 6, 1)
sys.exit()
createPhaseDiagram(allData, 5,1)
createPhaseDiagram(allData, 6,1)
createPhaseDiagram(allData, 6, 1, 19)
plotDataAndExponentialFit(allData,6,1)
plotDataAndExponentialFit(allData,6,0)
createPhaseDiagram(allData, 6,0)
dataset1 = allData
