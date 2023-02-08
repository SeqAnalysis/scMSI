import numpy as np
import statsmodels.stats.weightstats as sw

X1=np.loadtxt('normal_0.5.txt')
X2=np.loadtxt('tumor_0.5.txt',dtype=int)
print(np.mean(X1))
print(np.std(X1))
print("//////")
a=np.mean(X2)
b=np.std(X2)
print(np.mean(X2))
print(np.std(X2))
print('///')
a=round(a,2)
b=round(b,2)
print(a,b)
print("///")
p=sw.ztest(X1,value=13.69)
print(p)
if(p[1]>=0.001):
    print("Microsatellite status:mss")
else:
    print("Microsatellite status:msi")