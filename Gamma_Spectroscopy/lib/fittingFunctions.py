#!/usr/bin/env python3
import csv                               ## for reading in our data files
import logging                           ## for orderly print output
import numpy as np                       ## for handling of data
import sys                               ## useful system calls (used to exit cleanly)
import os                                ## path manipulations
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ipywidgets import interact, interactive, fixed, widgets


def GaussFunc(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def LineFunc(x, k, m):
    return k*x+m

def ExpFunc(t, a, b, c):
    """
    Normal exponential function
    where b is the decay constant
    """
    return a*np.exp(-b*t)+c

def Exp2Func(t, a, b, c, d, e):
    """
    Double exponential function
    where b and d are the two decay constants
    """
    return a*np.exp(-b*t)+c*np.exp(-d*t)+e


class Gauss:
    """A class to hold coefficients for Gaussian distributions"""
    def __init__(self, A, mu, sigma, covar_matrix):
        self.A = A
        self.mu = mu
        self.sigma = sigma
        self.covar_matrix = covar_matrix
    def value(self, x):
        return gaussfcn(x, self.A, self.mu, self.sigma)
    def area(self):
        return np.sqrt(2*np.pi)*self.A*np.abs(self.sigma)
    def as_string(self, ndigits=4):
        return str("A: {}, mu: {}, sigma: {}".format(round(self.A, ndigits),
                                                     round(self.mu, ndigits),
                                                     round(self.sigma, ndigits)))
    def print_full_info(self):
        print("Estimated parameters:\n A = {:.5f}, mu = {:.5f},  sigma = {:.5f} \n".format(self.A, self.mu, self.sigma))
        print("Uncertainties in the estimated parameters: \n \u03C3\u00b2(A) = {:.5f}, \u03C3\u00b2(mu) = {:.5f}, \u03C3\u00b2(sigma) = {:.5f} \n".format(self.covar_matrix[0][0], self.covar_matrix[1][1], self.covar_matrix[2][2]))
        print("Covariance matrix: \n {}".format(self.covar_matrix))
            
def fit_Gaussian(x, y, region_start, region_stop, mu_guess, A_guess=0, sigma_guess=1):
    """ a simple function that tries to fit a Gaussian and return a Gauss object if fit was successful """
    # these define a region of interest in our data binning:
    region = (region_start < x) & (x < region_stop)
    # now limit our data by 'slicing' it, limiting it to the region of interest:
    peak_region_bc     = x[region]
    peak_region_counts = y[region]
    # if A_guess was not given
    if (A_guess == 0): 
        A_guess = y.max()
     
    guess = [A_guess, mu_guess, sigma_guess] # our initial guess of parameters for a Gaussian fit 
    ## scypi gives a warning if the fit does not work; we want to know about those, so we set them up to be caught here:
    import warnings
    from scipy.optimize import OptimizeWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        ## perform the gaussian fit to the data:
        try:
            ## use the scipy curve_fit routine (uses non-linear least squares to perform the fit)
            ## see http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.optimize.curve_fit.html
            estimates, covar_matrix = curve_fit(GaussFunc,
                                    peak_region_bc,
                                    peak_region_counts,
                                    p0=guess)
            ## create a Gauss object with the fitted coefficients for better code readability
            g_final = Gauss(estimates[0], estimates[1], abs(estimates[2]), covar_matrix)
            return g_final
        except (RuntimeError, OptimizeWarning, TypeError):
            print("Gaussian fit failed! Try specifying another mu_guess.")
            return 0

def perform_Gaussian_fit(x, y, region_start, region_stop, mu_guess, A_guess=0, sigma_guess=1, left_selection=None, right_selection=None, plotting = 1, printing = 1):
    
    # these define a region of interest in our data binning:
    region = (region_start < x) & (x < region_stop)
    # now limit our data by 'slicing' it, limiting it to the region of interest:
    peak_region_bc     = x[region]
    peak_region_counts = y[region]
  
    if (left_selection and right_selection):
        ############ Selecting points to fit linear function 
        lin_region = ((x>left_selection[0]) & (x<left_selection[1])) | ((x>right_selection[0]) & (x<right_selection[1]))
        lin_bc = x[lin_region]
        lin_counts = y[lin_region]

        ############ Fitting linear function to selected points 
        guess = [2, 1] # guess parameters for linear fit 
        estimates_lin, covar_matrix = curve_fit(LineFunc,
                                            lin_bc,
                                            lin_counts,
                                            p0 = guess)

        #print("Linear fit coefficients (k m) = (", estimates_lin[0], estimates_lin[1], ")\n")
        ############ Subtract area under the linear fit
        line = LineFunc(peak_region_bc, estimates_lin[0], estimates_lin[1])
        peak_region_counts_subs = peak_region_counts - line
        peak_region_counts = peak_region_counts_subs.copy()    
        
    ############ Fit a Gaussian to the peak without background    
    g_final = fit_Gaussian(peak_region_bc, peak_region_counts, region_start, region_stop, mu_guess, A_guess, sigma_guess)   
    # Check if fit worked 
    if(g_final==0):
           return
       
    if (plotting):
        plt.figure()
        #Choose colour of the Gaussian depending on if fit was okay
        if (g_final.covar_matrix[2][2]<g_final.sigma):
            color = 'forestgreen'
        else:
            color = 'r'
        
        
        if (left_selection and right_selection):
            plt.step(peak_region_bc, peak_region_counts + line, where='mid', color='cornflowerblue', label='data') #plotting data
            plt.vlines(left_selection+right_selection, ymin=0, ymax=g_final.A)  #plot support lines around selected points 
            plt.plot(peak_region_bc, line + GaussFunc(peak_region_bc, g_final.A, g_final.mu, g_final.sigma), color=color, label = 'Gaussian fit')  # plot Gaussian 
            plt.plot(lin_bc, LineFunc(lin_bc, estimates_lin[0], estimates_lin[1]), color='r', label = 'linear fit', alpha=0.6)  # plot linear fit
        else:
            plt.step(peak_region_bc, peak_region_counts, where='mid', color='cornflowerblue', label='data') #plotting data
            plt.plot(peak_region_bc, GaussFunc(peak_region_bc, g_final.A, g_final.mu, g_final.sigma), color=color, label = 'Gaussian fit')
        plt.legend(loc='upper right', frameon=False)
        plt.show()
        
    if (printing):
        g_final.print_full_info()
        
    return g_final
