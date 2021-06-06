import tkinter
from tkinter import *

#from matplotlib.backends.backend_tkagg import *

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg)
#from matplotlib.backends.backend_tkagg import(NavigationToolbar2Tk)
#(  FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure

import numpy as np
# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/    



import matplotlib 
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
import Gaussian

LARGE_FONT= ("Verdana", 12)

 
class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        #tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Sea of BTC client")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, GraphPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="Graph Page",
                            command=lambda: controller.show_frame(GraphPage))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        self.sigma=0.1
        self.nscatt=10
        self.vmax=0.1
        reps = 10
        sigmaText = StringVar()
        trialText = StringVar()
        scattText = StringVar()
        self.lcorrSmallText = StringVar()
        self.lcorrLargeText =StringVar()
        vmaxText=StringVar()
        vmaxText.set(self.vmax)
        scattText.set(self.nscatt)
        trialText.set(reps)
        sigmaText.set(self.sigma)
        
        tk.Frame.__init__(self, parent)
        
        label = tk.Label(self, text="Largest and smallest correlation length", font=LARGE_FONT)
        label.grid(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid()

        button2 = ttk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.grid()
        self.sigmaLabel = tk.Label(self,text="Sigma")
        self.vmaxlabel = tk.Label(self,text="Vmax")
        self.nscattlabel = tk.Label(self,text="Number of imps")
        self.sigmaEntry = tk.Entry(self,textvariable = sigmaText)
        
        
        self.vmaxEntry = tk.Entry(self,textvariable=vmaxText)
        self.nscattEntry = tk.Entry(self,textvariable = scattText)
        self.repLabel = tk.Label(self,text="Trials")
        self.repEntry = tk.Entry(self, textvariable = trialText)
        self.sigmaEntry.grid(row=4,column=0)
        self.vmaxEntry.grid(row=4,column=1)
        self.nscattEntry.grid(row=4,column=2)
        self.sigmaLabel.grid(row=5,column=0)
        self.vmaxlabel.grid(row=5,column=1)
        self.nscattlabel.grid(row=5,column=2)
        self.nscattEntry.grid(row=4,column=3)
        self.repEntry.grid(row=4,column=4)
        self.repLabel.grid(row=5,column=4)
        startButton3 = tk.Button(self,text= "Start run", command=lambda: self.calculate(self.sigmaEntry.get(), self.vmaxEntry.get(), self.nscattEntry.get(), self.repEntry.get()))
        startButton3.grid()
        
        # Data
        
        self.lfigure_potential = Figure(figsize=(4,4), dpi=100)
        self.lfigure_correlation = Figure(figsize=(4,4), dpi=100)
        self.plot_left_potential = self.lfigure_potential.add_subplot(111)
        self.plot_left_potential.set_title("Potential - Largest correlation length")
        
        self.rfigure_potential= Figure(figsize=(4,4), dpi=100)
        self.rfigure_correlation= Figure(figsize=(4,4), dpi=100)
        self.plot_right_potential = self.rfigure_potential.add_subplot(111)
        self.plot_right_potential.set_title("Potential - Smallest correlation length")
        self.plot_right_correlation = self.rfigure_correlation.add_subplot(111)
        self.plot_right_correlation.set_title("ACF - 2D")
        
        self.plot_left_correlation = self.lfigure_correlation.add_subplot(111)
        self.plot_left_correlation.set_title("ACF - 2D")
        self.lcanvastop = FigureCanvasTkAgg(self.lfigure_potential, self)
        self.rcanvastop = FigureCanvasTkAgg(self.rfigure_potential, self)
        self.lcanvastop.get_tk_widget().grid(row=6,column=1)
        self.rcanvastop.get_tk_widget().grid(row=6,column=2)
        self.lcanvasbottom = FigureCanvasTkAgg(self.lfigure_correlation, self)
        self.rcanvasbottom = FigureCanvasTkAgg(self.rfigure_correlation, self)
        self.lcanvasbottom.get_tk_widget().grid(row=7,column=1)
        self.rcanvasbottom.get_tk_widget().grid(row=7,column=2)
        self.lcanvastop.show()
        self.rcanvastop.show()
        self.rcanvasbottom.show()
        self.lcanvasbottom.show()
        self.lcorrlargeLabel = tk.Label(self,textvariable = self.lcorrLargeText)
        self.lcorrsmallLabel = tk.Label(self,textvariable = self.lcorrSmallText)
        self.lcorrsmallLabel.grid(row=8,column=2)
        self.lcorrlargeLabel.grid(row=8,column=1)

    def calculate(self,sigma,vmax,nscatt,reps):
        (potLargeLcorr, lcorrLarge,potSmallLcorr, lcorrSmall) = Gaussian.findSmallestAndLargestCorrelationLength(float(vmax), int(nscatt), float(sigma), int(reps))
        # get acfs
       
        potSize = int(np.sqrt(len(potLargeLcorr)))
        potLargeLcorr = np.reshape(potLargeLcorr, (potSize,potSize))
        potSmallLcorr = np.reshape(potSmallLcorr, (potSize,potSize))
        corrSmall, corrSmallCut, lcorrLarge, fittedData,error  = Gaussian.calculateAutoCorrelationData(potSmallLcorr)
        corrLarge, corrLargeCut, lcorrSmall, fittedData,error = Gaussian.calculateAutoCorrelationData(potLargeLcorr)
        smallSigmaX, smallSigmaY,dataSmall =  (Gaussian.calculateAutoCorrelation2dFromPotentialData(potSmallLcorr))
        largeSigmaX, largeSigmaY,dataLarge =  (Gaussian.calculateAutoCorrelation2dFromPotentialData(potLargeLcorr))
#         print (smallSigmaX, smallSigmaY)
#         print (largeSigmaX, largeSigmaY)
#         print (lcorrLarge)
#         print (type(smallSigmaX))
#         print (largeSigmaX.shape)
        stringLarge = "lcorr = (%s) x = (%s), y = (%s)"%(str(lcorrLarge), str(largeSigmaX), str(largeSigmaY))
        stringSmall = "lcorr = (%s) x = (%s), y = (%s)"%(str(lcorrSmall), str(smallSigmaX), str(smallSigmaY))
#         print (stringLarge)
#         print (stringSmall)
        fitSize = int(np.sqrt(len(dataLarge)))
        dataLarge = np.reshape(dataLarge,(fitSize,fitSize))
        dataSmall = np.reshape(dataSmall,(fitSize, fitSize))
        self.lcorrLargeText.set(stringLarge )
        self.lcorrSmallText.set(stringSmall )
        self.plot_left_potential.contourf(potLargeLcorr)
        self.plot_right_potential.contourf(potSmallLcorr)
        self.plot_left_correlation.contourf(corrLarge)
        self.plot_left_correlation.contour(dataLarge, 8, colors='w')
        self.plot_right_correlation.contourf(corrSmall)
        self.plot_right_correlation.contour(dataSmall, 8, colors='w')
        self.lcanvastop.draw_idle()
        self.lcanvastop.flush_events()
        self.rcanvastop.draw_idle()
        self.rcanvastop.flush_events()
        self.lcanvasbottom.draw_idle()
        self.lcanvasbottom.flush_events()
        self.rcanvasbottom.draw_idle()
        self.rcanvasbottom.flush_events()
        return  
class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()


class GraphPage(tk.Frame):


    def calculateAndShowStatistics(self, pot):
        if  len(pot)  == 0:
            self.mean = "Unknown"
            self.variance = "Unknown"
        else:
            self.mean, self.variance = self.calculateStatistics(pot)
        self.meanText.set("Mean = "+str(self.mean))
        self.varianceText.set("Variance = "+ str(self.variance))
        

    def __init__(self, parent, controller):
        self.varianceText=StringVar()
        self.meanText=StringVar()
        
        
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.grid(row=0,column=0)
        self.sigma = 0.1
        self.nscatt = 10
        self.vmax = 0.15
        self.sigmaText = StringVar()
        self.nscattText = StringVar()
        self.vmaxtext= StringVar()
        self.sigmaText.set(self.sigma)
        self.nscattText.set(self.nscatt)
        self.vmaxtext.set(self.vmax)
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row=1,column=0)
       
        button2 = ttk.Button(self,text="Calculate",command = lambda: self.changeImage(float(self.sigmaEntry.get()), int(self.nscattEntry.get()), float(self.vmaxEntry.get()), x, y))
        button2.grid(row=1,column=1)
        plt.ion()
        x = np.linspace(0,1,101)
        y = np.linspace(0,1,101)
        (x,y) = np.meshgrid(x,y)
        pot =self.potentialData(0.1,1000,0.15,x,y)
        pot = self.potentialDataFromGaussian(1, 0.1, 1000, 0.15, x, y)
        data  = self.calculateAutoCorrelation(pot)
        autocorr2d = data[0]
        lcorr = data[1]
        xsize = len(autocorr2d)
        
        self.lfigure_potential = Figure(figsize=(5,5), dpi=100)
        
        self.plot_a = self.lfigure_potential.add_subplot(111)
        self.plot_a.set_title("Potential")
        self.plot_a.contourf(x,y,pot)
        
        self.rfigure_potential = Figure(figsize=(5,5), dpi=100)
        self.plot_autocorr2d = self.rfigure_potential.add_subplot(111)
        self.plot_autocorr2d.set_title("Autocorrelation l_{corr} =" +str(lcorr))
        self.plot_autocorr2d.contourf(autocorr2d)
        self.lcanvastop = FigureCanvasTkAgg(self.lfigure_potential, self)
        self.lcanvastop.show()
        self.rcanvastop = FigureCanvasTkAgg(self.rfigure_potential,self)
        self.rcanvastop.show()
        self.lcanvastop.get_tk_widget().grid(row=2,column=1)
        self.rcanvastop.get_tk_widget().grid(row=2,column=2)
        #self.lcanvastop.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

       # toolbar = NavigationToolbar2TkAgg(self.lcanvastop, self)
       # toolbar.update()
        self.lcanvastop._tkcanvas.grid(row=2,column=0)
        self.rcanvastop._tkcanvas.grid(row=2,column=1)
        #side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.varLabel = tk.Label(self,  textvariable=self.varianceText )
        self.meanLabel = tk.Label(self, textvariable = self.meanText )
        self.varLabel.grid(row=3,column=0)
        self.meanLabel.grid(row=3, column=1)
        self.calculateAndShowStatistics(pot)
        #===========Controls ==================
        self.sigmaLabel = tk.Label(self,text="Sigma")
        self.vmaxlabel = tk.Label(self,text="Vmax")
        self.nscattlabel = tk.Label(self,text="Number of imps")
        self.sigmaEntry = tk.Entry(self,textvariable = self.sigmaText)
        self.vmaxEntry = tk.Entry(self,textvariable=self.vmaxtext)
        self.nscattEntry = tk.Entry(self,textvariable = self.nscattText)
        self.sigmaEntry.grid(row=4,column=0)
        self.vmaxEntry.grid(row=4,column=1)
        self.nscattEntry.grid(row=4,column=2)
        self.sigmaLabel.grid(row=5,column=0)
        self.vmaxlabel.grid(row=5,column=1)
        self.nscattlabel.grid(row=5,column=2)
        
    def changeImage(self, sigma,nscatt,vmax,x,y):
       
        print("Chanegimage")
        #pot = self.potentialData(sigma, nscatt, vmax, x, y)
        pot = self.potentialDataFromGaussian(1, sigma, nscatt, vmax, x, y)
        self.plot_a.contourf(x,y,pot)
        self.lcanvastop.draw_idle()
        self.lcanvastop.flush_events()
        
        
    def potentialData(self, sigma,nscatt,vmax,x,y):
        print ("pot Data")
        print (sigma,nscatt,vmax)
        data = Gaussian.generateRandomPotential(nscatt,vmax,sigma)
        pot = data[2]
        self.calculateAndShowStatistics(pot)
        potSize = len(pot)
        
            #print (potSize)
        pot = np.reshape(pot,(potSize,potSize))
        
        
        return pot
        #, autocorr2d[0])
    def potentialDataFromGaussian(self,reps, sigma,nscatt,vmax,x,y):
        for i in np.arange(0,reps,1):
            pot = Gaussian.procedure(i,nscatt,vmax,sigma,True)[0]
        # resize pot 
        potSize = len(pot)
        potSize = int(np.sqrt(potSize))
        
        pot = np.reshape(pot,(potSize,potSize))
        self.calculateAndShowStatistics(pot)
        return pot
    
    
    def calculateAutoCorrelation(self,pot):
        autocorr2d = Gaussian.calculateAutoCorrelationData(pot)
        
        return autocorr2d
    
    def calculateStatistics(self,pot):
        variance = np.var(pot)
        mean = np.mean(pot)
        return (mean,variance)

app = SeaofBTCapp()
app.mainloop()
        
