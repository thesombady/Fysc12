import numpy as np
import math
from scipy.optimize import curve_fit

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
        gaussian2 = lambda A, x, mu, sigma: A * math.exp(- (x - mu) **2/(2*sigma ** 2) ** 2)
        #mu => center of peak
        #sigma => width
        #A => amplitude
        Converty = lambda y: np.ln(y)
        ConvertRhs = lambda A, x, mu, sigma: np.ln(A) - mu**2/(2*sigma**2) + 2 * mu * sigma/(2*sigma**2) - x**2/(2*sigma**2)
        #ln(y) = a + bx + cx^2
        a = lambda A, mu, sigma: np.ln(A) - mu ** 2/(2 * sigma**2)
        b = lambda mu, sigma: mu / (sigma**2)
        c = lambda sigma: - 1 /(2 * sigma**2)

        Error = lambda y, A, x, mu, sigma: np.ln(y) - (np.ln(A) - mu**2/(2*sigma**2) + 2 * mu * sigma/(2*sigma**2) - x**2/(2*sigma**2))

        Constant_a = np.ones(len(xlist))
        line1 = np.ones(len(xlist))
        MatrixA = np.array([np.ones(len(xlist)), xlist, xlist ** 2])
        MatrixAT = MatrixA.T
        print(MatrixAT.dot(MatrixA))
        print(np.linalg.det(MatrixAT.dot(MatrixA)))

        try:
            InverseA = np.linalg.inv(MatrixAT.dot(MatrixA))
            print(InverseA)
        except Exception as E:
            raise E


        """
        line1 = np.ones(len(xlist))
        A = np.array([xlist, line1]).T
        ATA = A.T.dot(A)
        ATY = A.T.dot(ylist)
        ATAInv = np.linalg.inv(ATA)
        KX = ATAInv.dot(ATY)
        if KX[0] == 0 or KX[1] == 0:
            raise KeyError("[Gaussianfit] Can't make an accurate fit")
        print(KX)
        def Gaussian3(modx, mody):
            if not isinstance((modx, mody), (np.generic, np.ndarray)):
                raise TypeError("[GaussianFit] Can't make Gaussianfit.")
            else:
        """

        #return np.array([gaussian2(1,1,1,1)])

xlist1 = np.array([1,2,3,4,5,6,8,9,10])
ylist1 = np.array([2,3,4,7,7,6,5,5,3])
print(Gaussianfit(xlist1, ylist1))
