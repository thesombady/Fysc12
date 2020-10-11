import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class Gaussian:
    """Parametric representation of the Gaussian fit."""
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return f'{[arg for arg in self.args]}'

    def __str__(self):
        return f'{[arg for arg in self.args]}'

    def __getitem__(self, index):
        return self.args[index]

def Gaussianfit(xlist, ylist):
    """Computes a gaussian fit in accordance to method of least square in terms of two list that are given."""
    if not isinstance((xlist, ylist), (np.generic, np.ndarray)):
        if isinstance((xlist, ylist), (list, tuple)):
            xlist = np.array(xlist)
            ylist = np.array([[arg] for arg in ylist])
        else:
            raise TypeError("[GaussianFit] Can't make Gaussianfit with given input")
    if len(xlist) < 4 and len(xlist) == len(ylist):
        raise KeyError("[GaussianFit] Can't make Gaussianfit due to too few values given")
    else:
        Error = lambda y, A, x, mu, sigma: np.ln(y) - (np.ln(A) - mu**2/(2*sigma**2) + 2 * mu * sigma/(2*sigma**2) - x**2/(2*sigma**2))

        Constant_a = np.ones(len(xlist))
        line1 = np.ones(len(xlist))
        MatrixA = np.array([np.ones(len(xlist)), xlist, xlist ** 2]).T
        MatrixAT = MatrixA.T
        try:
            InverseA = np.linalg.inv(MatrixAT.dot(MatrixA))
            ylist2 = np.log(ylist)
            ylist3 = MatrixAT.dot(ylist2)
            Constants = InverseA.dot(ylist3)
        except Exception as E:
            raise E
        sigma = np.log(- 1 / (2 * Constants[2]))
        mu = Constants[1] * sigma ** 2
        A = np.exp(Constants[0] + mu ** 2 / (2 * sigma ** 2))
        print(A, mu, sigma)

        return np.array([A, mu, sigma])


xlist1 = np.array([1,2,3,4,5,6,8,9,10])
ylist1 = np.array([2,3,4,7,8,6,5,4,3])
Fitted = Gaussianfit(xlist1, ylist1)
plt.plot(xlist1, ylist1, '.', label = "tested values")
#plt.show()
