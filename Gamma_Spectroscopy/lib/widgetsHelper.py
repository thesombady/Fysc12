#!/usr/bin/env python3
import csv                               ## for reading in our data files
import logging                           ## for orderly print output
import numpy as np                       ## for handling of data
import sys                               ## useful system calls (used to exit cleanly)
import os                                ## path manipulations
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def GaussFunc(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def plot_manual_gaussian(x, y, region_start, region_stop, A, mu, sigma):
    region = (region_start < x) & (x < region_stop)
    peak_region_bc     = x[region]
    ### Plot your Gaussian fit
    plt.figure()
    plt.step(x, y, where='mid', label='Data')
    plt.plot(peak_region_bc, GaussFunc(peak_region_bc, A, mu, sigma), label = 'Gaussian fit') # plotting the Gaussian
    plt.legend() # adds a legend with all 'label'ed plots!
    print("Obtained parameters:\n A = {:.2f}, mu = {:.2f},  sigma = {:.2f} \n".format(A, mu, sigma))