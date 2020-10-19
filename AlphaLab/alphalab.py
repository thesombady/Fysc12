from PhysicsNum import GaussianFit, Linearreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv                               ## for reading in our data files
import logging                           ## for orderly print output
import sys                               ## useful system calls (used to exit cleanly)
import os                                ## path manipulations

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

Amaricum80V = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/Amaricium80V.Spe')
Amercium90V = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium90V.Spe')
Amercium100V = load_spectrum('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/20201007/Amaricium100V.Spe')
#80V

Amercium80V1 = GaussianFit(Amaricum80V[0], Amaricum80V[1])
AmerciumValues80V = Amercium80V1.ComputeGaussian(2373, 2404)

#90V

Amercium90V1 = GaussianFit(Amercium90V[0], Amercium90V[1])
AmerciumValues90V = Amercium90V1.ComputeGaussian(2375, 2407)

#100V

Amercium100V1 = GaussianFit(Amercium100V[0], Amercium100V[1])
AmerciumValues100V = Amercium100V1.ComputeGaussian(2375, 2400)

#2388.66114752
#Every intersection goes towards the same point.
#Being the one listen above.
CalibrationConstant = 5485.56 #keV
EnergyConvertion = lambda x: CalibrationConstant / x
