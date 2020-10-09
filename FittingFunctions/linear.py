import numpy as np
import math
import matplotlib.pyplot as plt



def Linearregression(xlist, ylist):
    """ Takes two inputs, in list, tuple or arrays and computes a linear regression with method of least squares.
    Returns k * (x) + m where """ #Add the return of std-error and r^2 value
    if not isinstance((xlist, ylist), (np.generic, np.ndarray)):
        if isinstance((xlist, ylist), (list, tuple)):
            xlist, ylist = np.array(xlist), np.array(ylist)
        else:
            raise TypeError("[LinearRegression] Can't make linear fit with given input")
    if len(xlist) < 2:
        raise TypeError("[LinearRegression] Can't make linear fit with given input, add more terms")
    else:
        Line = lambda k,x,m: k * x + m
        try:
            bline = np.ones(len(xlist))
            A = np.array([xlist, bline]).T
            ATA = A.T.dot(A)
            ATY = A.T.dot(ylist)
            ATAInv = np.linalg.inv(ATA)
            KM = ATAInv.dot(ATY)
            return KM
        except Exception as E:
            raise E

"""
xlist1 = np.array([1,2,3,4,5,6,7,8,9,10])
ylist1 = np.array([4,6,9,10,12,14,16,18,20,21])
Regression = Linearregression(xlist1, ylist1)

plt.plot(xlist1,ylist1, '.')
plt.plot(xlist1, Regression[0] * xlist1 + Regression[1])
plt.show()
"""
