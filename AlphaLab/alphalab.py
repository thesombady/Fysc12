from PhysicsNum import GaussianFit, Linearreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv                               ## for reading in our data files
import logging                           ## for orderly print output
import sys                               ## useful system calls (used to exit cleanly)
import os                                ## path manipulations
from scipy import optimize
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy

class Spectrum:
    """A class to hold our spectrum measurement data and meta data (such as duration)"""
    def __init__(self, filename):
        self.filename = filename
        self.bin_edges = np.array(np.zeros(1))    ## creates a new empty array; will store edges of our bins
        self.bin_centers = np.array(np.zeros(1))
        self.energy = np.array(np.zeros(1))
        self.counts = np.array(np.zeros(1))    ## creates a new empty array; will later store our y values
        self.name = os.path.splitext(     ## a more descriptive name, can be used e.g. in legends
                os.path.basename(filename))[0] ## init with file name without extension
        self.duration = 0
    def subtract(self, m):
        self.counts = self.counts - m.counts
        ## these are spectra: cannot have counts below 0, so remove these here and set them to 0 instead:
        self.counts[self.counts < 0] = 0;
    def scale(self, scale):
        self.counts *= scale
    def calibrate(self, slope, intercept):
        self.energy = self.bin_centers*slope + intercept
    def calculate_bin_centers(self):
        self.bin_centers = 0.5*(self.bin_edges[1:] + self.bin_edges[:-1])


def load_spectrum(filename):
    """Reads in a data file (csv format) stored by the Maestro MCA software and returns a 'Spectrum' object. Tested with Maestro Version 6.05 """
    log = logging.getLogger('load_spectrum') ## set up logging
    m = Spectrum(filename) ## create a new Spectrum measurement object; this is what we return in the end
    log.info("Reading data from file '" + filename + "'")
    xValues = []
    yValues = []
    try:
        with open(filename, newline='') as f:
            reader = csv.reader(f) ## use the python csv module to parse the file
            interval = []          ## start/stop channel numbers used to assign correct x values to the data points
            ## first parse the "header" of the data file (until the '$DATA:' line) containing all the meta data
            for row in reader:
                if row[0] == '$MEAS_TIM:':
                    ## this item gives the duration of the measurement
                    log.debug("Parsing MEAS_TIM header info")
                    row = next(reader)
                    duration = [int(s) for s in row[0].split(' ')]
                    m.duration = duration[1] ## two parts: real time/live time; take the second
                if row[0] == '$DATA:':
                    ## this is the last part of the header and contains the start/stop channel numbers
                    log.debug("Parsing DATA header info")
                    row = next(reader)
                    interval = [int(s) for s in row[0].split(' ')]
                    ## "DATA" is the last item: stop with the header processing
                    break
            ## TODO: make sure that the file does not end before we have parsed the header!
            log.debug("Done with header parsing")
            nchannel = int(interval[1]-interval[0])+1
            m.counts = np.array(np.zeros(nchannel))

            ## continue, now reading data
            for idx, row in enumerate(reader):
                if idx >= nchannel:
                    break
                m.counts[idx] = int(row[0])
            m.bin_edges = np.arange(interval[0], interval[1]+2,1)
            m.calculate_bin_centers()
            log.debug("Loaded all data from file")
    except IOError:
        log.error("Could not find the file '"+str(filename)+"'")
        return None
    return m.bin_centers, m.counts
"""
Amaricum80V = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/Amaricium80V.Spe')
Amercium90V = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium90V.Spe')
Amercium100V = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium100V.Spe')
#80V
Amercium80V1 = GaussianFit(Amaricum80V[0], Amaricum80V[1])
AmerciumValues80V = Amercium80V1.ComputeGaussian(2373, 2404)
ErrorAmercium80V = AmerciumValues80V[-1]
#90V
Amercium90V1 = GaussianFit(Amercium90V[0], Amercium90V[1])
AmerciumValues90V = Amercium90V1.ComputeGaussian(2375, 2407)
ErrorAmercium90V = AmerciumValues90V[-1]
#100V
Amercium100V1 = GaussianFit(Amercium100V[0], Amercium100V[1])
AmerciumValues100V = Amercium100V1.ComputeGaussian(2375, 2400)
ErrorAmercium100V = AmerciumValues100V[-1]
"""
#2388.66114752
#Every intersection goes towards the same point.
#Being the one listen above.
CalibrationConstant = 5485.56 #keV
Center = 2388.66114752
EnergyConvertion = lambda x: CalibrationConstant / Center * x
k = CalibrationConstant / Center
"""
Calibration done
"""
Amaricium100V1cm = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium100V1cm.Spe')
Amaricium100V2cm = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium100V2cm.Spe')
Amaricium100V2_5cm = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium100V2_5cm.Spe')
Amaricium100V3cm = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium100V3cm.Spe')
#Import done
Americum100V1cm1 = GaussianFit(Amaricium100V1cm[0], Amaricium100V1cm[1])
Americum100V1cm1.Calibration(EnergyConvertion, k)
Error1 = Americum100V1cm1.ComputeGaussian(1696, 1861)
Alphapeak1 = Americum100V1cm1.CalibratedPeaks()
#Americum100V1cm1.PlotData()
Amaricium100V2cm1 = GaussianFit(Amaricium100V2cm[0], Amaricium100V2cm[1])
Amaricium100V2cm1.Calibration(EnergyConvertion, k)
Error2 = Amaricium100V2cm1.ComputeGaussian(1214, 1380)
Alphapeak2 = Amaricium100V2cm1.CalibratedPeaks()
#Amaricium100V2cm1.PlotData()
Amaricium100V2_5cm1 = GaussianFit(Amaricium100V2_5cm[0], Amaricium100V2_5cm[1])
Amaricium100V2_5cm1.Calibration(EnergyConvertion, k)
Error3 = Amaricium100V2_5cm1.ComputeGaussian(930, 1100)
Alphapeak3 = Amaricium100V2_5cm1.CalibratedPeaks()
#Amaricium100V2_5cm1.PlotData()
Amaricium100V3cm1 = GaussianFit(Amaricium100V3cm[0], Amaricium100V3cm[1])
Amaricium100V3cm1.Calibration(EnergyConvertion, k)
Error4 = Amaricium100V3cm1.ComputeGaussian(625, 795)
Alphapeak4 = Amaricium100V3cm1.CalibratedPeaks()
#Amaricium100V3cm1.PlotData()
"""
Calibration curve
"""
#plt.plot(Amaricium100V3cm[0], 'b')
plt.plot(Amaricium100V3cm[0], np.array(Amaricium100V3cm[0]) *k,'g', label = "Calibration curve")
plt.xlabel("Count")
plt.ylabel("Energy [keV]")
plt.title("Calibration of instruments")
plt.legend()
plt.show()

"""
Bethe-Bloch task
"""
R = [1.05, 1.97, 2.45, 2.93] #cm
Ei = 5.48556 #MeV



def Bethe(E_k):
    Electronpart = 0.307075 #MeVcm^2/g
    z = 2
    electronmass = 0.511 #MeVcm
    Z = 14.46
    A = 28.96 #g/mol
    rho = 1.225*10**(-3)# g/cm3
    I = 8.6 * 10 ** (-5)#MeV
    alphamass = 3727 #MeV
    Beta2 = 1 - 1/(1 + E_k/alphamass)**2
    Na = 6.022 * 10 ** (23)
    return (Electronpart/Beta2 * Z * z ** 2 * rho / A  * (np.log((2*electronmass*Beta2)/(I*(1-Beta2)))-Beta2)) ** (-1)

def func1(E,r):
    solved = quad(Bethe, E, Ei)
    return solved[0] - r

vfunc1 = scipy.vectorize(func1)
EnergyValues = np.array([fsolve(vfunc1, 0.5, args=(r)) for r in R]) * 10 ** (3) #Convert from Mev to keV
MeasuredValues = np.array([Alphapeak1[0], Alphapeak2[0], Alphapeak3[0], Alphapeak4[0]])
print(EnergyValues)
print(MeasuredValues)
ErrorValues = np.array([Error1[1], Error2[1], Error3[1], Error4[1]])
print(ErrorValues)

plt.plot(R, EnergyValues, '.', label = "Theoretical values")
plt.plot(R, MeasuredValues, '+')
plt.errorbar(R, MeasuredValues, yerr = ErrorValues, xerr = 0.05, linestyle = "None", label = 'Measured values')
plt.title("Bethe & Bloch")
plt.xlabel('Distance [cm]')
plt.ylabel("Energy [keV]")
plt.legend()
plt.show()

"""
NewMeasuredValues = np.array([Americum100V1cm1.Counts[0]/200.18, Amaricium100V2cm1.Counts[0]/199.88,
    Amaricium100V2_5cm1.Counts[0]/199.66,
    Amaricium100V3cm1.Counts[0]/200.08])

Xlist = [Americum100V1cm1.Calibratedp()[0], Amaricium100V2cm1.Calibratedp()[0], Amaricium100V2_5cm1.Calibratedp()[0], Amaricium100V3cm1.Calibratedp()[0]]#Energy


plt.plot(Xlist, NewMeasuredValues, '.', label = "Data aquired")
plt.title("Counts per seconds versus energy")
plt.ylabel("Counts per seconds")
plt.xlabel("Energy [keV]")
plt.legend()
plt.show()
"""
newAmericum100V1cm = np.array(Amaricium100V1cm[1]) / 200.18
xaxis1 = np.array(Amaricium100V1cm[0]) * k
newAmericum100V2cm = np.array(Amaricium100V2cm[1]) / 199.88
xaxis2 = np.array(Amaricium100V2cm[0]) * k
newAmericum100V2_5cm = np.array(Amaricium100V2_5cm[1]) / 199.66
xaxis3 = np.array(Amaricium100V2_5cm[0]) * k
newAmericum100V3cm = np.array(Amaricium100V3cm[1]) / 200.08
xaxis4 = np.array(Amaricium100V3cm[0]) * k

plt.plot(xaxis1, newAmericum100V1cm, '.', color = 'b', label = "1cm Spectrum")
plt.plot(xaxis2, newAmericum100V2cm, '.', color = 'r', label = "2cm Spectrum")
plt.plot(xaxis3, newAmericum100V2_5cm, '.', color = 'g', label = "2.5cm Spectrum")
plt.plot(xaxis4, newAmericum100V3cm, '.', color = 'black', label = "3cm Spectrum")
plt.title("Varying distance spectrum")
plt.xlabel("Energy [keV]")
plt.ylabel("Counts per second")
plt.legend()
plt.show()


#Calibration done for the newly imported files
#Now we tend to fit
Thorium1 = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Thorium.Spe')
Thorium = GaussianFit(Thorium1[0], Thorium1[1])

#Thorium.Calibration(EnergyConvertion)
#Thorium.PlotData('Thorium spectrum', 'Energy [keV]', 'Counts', 'Aquired Data')

Thoriumpeak1 = Thorium.ComputeGaussian(2308, 2338)
Thoriumpeak2 = Thorium.ComputeGaussian(2345, 2379)
Thoriumpeak3 = Thorium.ComputeGaussian(2457, 2495)
Thoriumpeak4 = Thorium.ComputeGaussian(2605, 2675)
Thoriumpeak5 = Thorium.ComputeGaussian(2720, 2760)
Thoriumpeak6 = Thorium.ComputeGaussian(2938, 2975)
Thoriumpeak7 = Thorium.ComputeGaussian(3818, 3850)
Thorium.Calibration(EnergyConvertion, CalibrationConstant / Center)
print(Thorium.CalibratedPeaks())
Thorium.PlotData("Thorium Spectrum", "Energy [keV]", "Counts")
