import numpy as np
import math
from collections import Counter
from decimal import *
def approx_gamma(Z):
    RECIP_E=0.36787944117144232159552377016147
    TWOPI=6.283185307179586476925286766559
    D=1.0 / (10.0 * Z)
    D = 1.0 / ((12 * Z) - D)
    D = (D + Z) * RECIP_E
    D = pow(D, Z)
    D *= math.sqrt(TWOPI / Z)
    return D

def igf(S,Z):
    if(Z<0.0):
        return 0.0
    Sc=(1.0/S)
    Sc *= pow(Z, S)
    Sc *= math.exp(-Z)
    Sum = Nom = Denom = float(1.0)
    for i in range(150):
        Nom*=Z
        print("Nom:",Nom)
        S+=1
        Denom*=S
        Sum+=(Nom/Denom)

    return Sum*Sc

def chisqr(Dof,Cv):
    if(Cv<0 | Dof<1):
        return 0.0
    #K=float(Dof)*0.5
    K = Dof* 0.5
    X=Cv*0.5
    if(Dof==2):
        return math.exp(-1.0*X)
    PValue=igf(K,X)
    if ((np.isnan(PValue)) | (np.isinf(PValue)) | (PValue <= 1e-8)):
        return 1e-14
    PValue /=approx_gamma(K)
    return (1.0-PValue)

def X2BetweenTwo(FirstOriginal,SecondOriginal,dispots):
    SumFirst = 0.0
    SumSecond = 0.0
    SumTotal = 0.0
    ExpFirst=np.zeros(dispots)
    ExpSecond=np.zeros(dispots)
    SumBoth=np.zeros(dispots)

    for i in range(dispots):
        SumBoth[i] = FirstOriginal[i] + SecondOriginal[i]
        SumFirst += FirstOriginal[i]
        SumSecond += SecondOriginal[i]
    SumTotal=SumFirst+SumSecond

    for i in range(dispots):
        ExpFirst[i] = SumBoth[i] * SumFirst / SumTotal
        ExpSecond[i] = SumBoth[i] * SumSecond / SumTotal

    print("ExpFirst:")
    print(ExpFirst)
    print(ExpSecond)


    result=0.0
    Degree=0
    for i in range(dispots):
        if(FirstOriginal[i] + SecondOriginal[i] > 0.0):
            Degree+=1
            if(ExpFirst[i]):
                result += (FirstOriginal[i] - ExpFirst[i]) * (FirstOriginal[i] - ExpFirst[i]) / ExpFirst[i]
            if(ExpSecond[i]):
                result += (SecondOriginal[i] - ExpSecond[i]) * (SecondOriginal[i] - ExpSecond[i]) / ExpSecond[i]
    print("Degree:")
    print(Degree)
    print("result:")
    print(result)
    PValue=0.0
    if((Degree == 1) | (result == 0)):
        PValue=1.0
    else:
        Degree-=1
        PValue = chisqr(Degree, result)

    if(PValue<0):
        PValue = PValue * (-1)

    return PValue,Degree

def transformation(X):
    count = Counter(X)
    print(count)
    dictlist = []
    for keys, value in count.items():
        temp = [keys, value]
        dictlist.append(temp)
    print(dictlist)
    max = dictlist[0][0]
    for t in range(len(dictlist)):
        if (max < dictlist[t][0]):
            max = dictlist[t][0]
    print(max)
    d = np.zeros(max)
    for i in range(max):
        for j in range(len(dictlist)):
            if (dictlist[j][0] == i + 1):
                d[i] = dictlist[j][1]
    #print(d)
    return d


X1=np.loadtxt('normal_0.5.txt',dtype=int)
X1=transformation(X1)
print(X1)
X2=np.loadtxt('tumor_0.5.txt',dtype=int)
X2=transformation(X2)
print("X2:")
print(X2)
X3=[]
X3=X1.tolist()
print("X3:")
print(X3)
X4=[]
X4=X2.tolist()
print("X4：")
print(X4)
print(len(X4))
print(len(X3))
x1=len(X3)
x2=len(X4)
if(x1<x2):
    len=x2
    for i in range(x2-x1):
        X3.append(0)
else:
    len = x1
    for i in range(x1-x2):
        X4.append(0)
print("newx3:")
print(X3)
print("newx4:")
print(X4)

D=0
P,D=X2BetweenTwo(X3,X4,len)
print("len:")
print(len)
print("P值:")
print(P)
if(P>=0.001):
    print("Microsatellite status：mss")
else:
    print("Microsatellite status：msi")




















