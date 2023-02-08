import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from collections import Counter
import scipy.stats as ss

x1=np.loadtxt('sub1.txt',dtype=int)
x2=np.loadtxt('sub2.txt',dtype=int)
x3=np.loadtxt('sub3.txt',dtype=int)
x_normal=np.loadtxt('x_normal.txt',dtype=int)

x=[]
x.extend(x1)
x.extend(x2)
x.extend(x3)

xw=np.linspace(5,17,100)
xw1=np.linspace(5,13,100)
xw2=np.linspace(8,15,100)
xw3=np.linspace(9,17,100)
xw4=np.linspace(5,17,100)


y1=0.3*ss.norm(9,1).pdf(xw1)
y2=0.4*ss.norm(11,0.8).pdf(xw2)
y3=0.3*ss.norm(13,1).pdf(xw3)
y=ss.norm(11,1.6).pdf(xw4)

y4=ss.norm(10,2).pdf(xw1)
y5=ss.norm(16,2).pdf(xw2)
y6=ss.norm(21,2).pdf(xw3)

y_true=0.3*ss.norm(9,1).pdf(xw)+0.4*ss.norm(11,0.8).pdf(xw)+0.3*ss.norm(13,1).pdf(xw)

plt.plot(xw1, y1, "r--",alpha=1,linewidth=2,label='Sub-distribution1',marker='+')
plt.plot(xw2, y2, "b--",alpha=1,linewidth=2,label='Sub-distribution2')
plt.plot(xw3, y3, "orange",alpha=1,linewidth=2,label='Sub-distribution3',marker='|')
plt.plot(xw, y_true, "grey",alpha=1,linewidth=2,label = 'Convolution distribution',marker='*')
plt.hist1(x1,30,density = True,facecolor = 'red',alpha=0.4,label = 'Clone1',edgecolor="black")
plt.hist2(x2,30,density = True,facecolor = 'green',alpha=0.4,label = 'Clone2',edgecolor="black")
plt.hist3(x3,30,density = True,facecolor = 'orange',alpha=0.4,label = 'Clone3',edgecolor="black")
#plt.hist(x,30,density = True,facecolor = 'cyan',alpha=0.4,label = 'Mixed sample')
plt.hist(x_normal,30,density = True,facecolor = 'skyblue',alpha=0.3,label = 'Normal sample',histtype='bar',edgecolor="black")
plt.plot(xw4, y, "black",alpha=0.6,linewidth=2,label = 'Normal sample distribution')
#plt.plot(xw1, y4, "r--",alpha=0.2,linewidth=2)
#plt.plot(xw2, y5, "b--",alpha=0.2,linewidth=2)
#plt.plot(xw3, y6, "orange",alpha=0.2,linewidth=2)
#plt.hist(x1,50,density = True,color = 'orange',label = 'Random Data')
#powderblue
plt.xlabel("Microsatellite sequence length(bp)")
plt.ylabel("Probability")
plt.legend()
plt.show()




