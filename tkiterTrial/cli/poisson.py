#!/usr/bin/env python


import pylab
import threading

class MyThread(threading.Thread):
    def __init__(self,y_lower,y_upper):
        threading.Thread.__init__(self)
        self.y_lower= y_lower
        self.y_upper = y_upper

    def run(self):
        for xi in range(len(x)):
            for yi in range(self.y_lower,self.y_upper):
                P[yi,xi] = pylab.sum(
                    pylab.select(
                        [(x[xi]-X)*(x[yi]-Y)==0,],
                        [0,],
                        Q/pylab.sqrt((x[xi]-X)**2+(x[yi]-Y)**2)
                    )
                )

dx = .05
x = pylab.arange(-5,5,dx)
X,Y = pylab.meshgrid(x,x)

# Homogeneous charge density
Q = pylab.zeros_like(X)
#Q = pylab.select([abs(X+2)<4*dx,],[pylab.select([abs(Y)<4*dx,],[-1,],0),],Q)
Q= pylab.exp(-(X**2+Y**2)/2) -  pylab.exp(-((X-2)**2+(Y+2)**2)/2) 
#Q += pylab.select([abs(X+.5)<.3,],[pylab.select([abs(Y)<.33],[-1,],0),],Q)

# Sharp rectangular charge density
#Q = pylab.zeros_like(X)
#Q[40,31:70] = 1
#Q[41:45,30] = 1
#Q[41:45,70] = 1
#Q[45,31:70] = 1

#Q[60,46:54] = -1
#Q[61:62,45] = -1
#Q[61:62,54] = -1
#Q[62,46:54] = -1

# Sharp circular charge density
#Q = pylab.zeros_like(X)
#Q = pylab.select([abs((X-2)**2+(Y+2)**2-(1.5)**2)<pylab.sqrt(2*dx)],[1],Q)

P = pylab.zeros_like(X)

threads = []
dy = 4
for i in range(dy):
    y_lower = int(len(x)/dy*float(i))
    y_upper = int(len(x)/dy*float(i+1))
    t = MyThread(y_lower,y_upper)
    threads.append(t)
    t.start()

if y_upper < len(x):
    t = MyThread(y_upper,len(x))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

#E = pylab.gradient(P)
pylab.gray()
pylab.pcolormesh(X,Y,Q)
pylab.spectral()
pylab.contourf(X,Y,P,levels=pylab.linspace(P.min(),P.max(),125),alpha=0.8)
pylab.colorbar()
#pylab.quiver(X,Y,E[1],E[0],pivot='middle',alpha=.5,color='gray')
pylab.show()
