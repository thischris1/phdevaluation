'''
Created on Jan 1, 2020

@author: chris
'''
from sympy.physics.continuum_mechanics.beam import matplotlib
from matplotlib.widgets import Slider

if __name__ == '__main__':
    pass
#!/usr/bin/python3
import tkinter
from tkinter import *

#from matplotlib.backends.backend_tkagg import *

from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg)
#from matplotlib.backends.backend_tkagg import(NavigationToolbar2Tk)
#(  FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure

import numpy as np


class App:
    def __init__(self, master):
        frame = Frame(master)
        #frame.pack()
        Grid.rowconfigure(root, 0, weight=1)
        Grid.columnconfigure(root, 0, weight=1)
        frame.grid(row=0, column=0, sticky=N+S+E+W)
        self.button = Button(frame, 
                             text="QUIT", fg="red",
                             command=frame.quit)
        
        self.button.pack(side=LEFT)
        self.sigmaSlider = Scale(frame,from_=0, to=100,length = 600,tickinterval=10,orient=HORIZONTAL)
        self.sigmaSlider.pack(side=BOTTOM)
        self.nmaxSlider =Scale(frame, from_=1, to =10000, length = 600, tickinterval=1000, orient=HORIZONTAL)
        self.nmaxSlider.pack(side=BOTTOM)
        self.vmaxSlider=Scale(frame,from_ =1, to_=10, tickinterval=1,length = 600, orient=HORIZONTAL)
        self.vmaxSlider.pack(side=BOTTOM)
        self.slogan = Button(frame,
                             text="Calculate",
                             command=self.evaluate_andwrite)
        self.slogan.pack(side=LEFT)
    def evaluate_andwrite(self):
        print (self.sigmaSlider.get())
        print (self.nmaxSlider.get())
        print ("Tkinter is easy to use!")
        # start calculation
        data = np.random.rand(self.nmaxSlider.get())
        print ("Variance " +str(np.var(data)))
        print ("mean " + str(np.mean(data)))
               
    def setValues(self,n_sigma,n_vmax,n_nmax):
        self.sigmaSlider.set(n_sigma)
        self.nmaxSlider.set(n_nmax)
        self.vmaxSlider.set(n_vmax)
          

root = Tk()
app1 = App(root)
app2 = App(root)
root.mainloop()



# root = tkinter.Tk()
# root.wm_title("Embedding in Tk")
# 
# 
# fig = Figure(figsize=(5, 4), dpi=100)
# t = np.arange(0, 3, .01)
# fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
# 
# canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
# canvas.draw()
# canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
# 
# #toolbar = NavigationToolbar2Tk(canvas, root)
# #toolbar.update()
# canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
# 
# tkinter.mainloop()
