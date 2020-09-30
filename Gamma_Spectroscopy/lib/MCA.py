#!/usr/bin/env python3
import csv                               ## for reading in our data files
import logging                           ## for orderly print output
import numpy as np                       ## for handling of data
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
    return m

#This function reads the calibrated background spectrum that is to be analysed in the gamma lab.
def load_calibrated_spectrum(filename):
    log = logging.getLogger('gammalab_analysis') ## set up logging
    m = Spectrum(filename) ## create a new Spectrum measurement object; this is what we return in the end
    m.energy = np.zeros(8192)
    m.counts = np.zeros(8192)
    log.info("Reading calibrated data from file '" + filename + "'")
    try:
        with open(filename) as f: #the with keyword handles the opening (__enter__ method) and closing (__exit__ method) of the file automatically
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                counts, energy = row[0].split() 
                m.counts[idx] = int(counts)
                m.energy[idx] = float(energy)
    except IOError:
        log.error("Could not find the file '"+str(filename)+"'")
        sys.exit(-1)
    return m

def load_MCS_spectrum(filename):
    try:
        counts = np.arange(0)
        with open(filename, newline='') as f:
                reader = csv.reader(f) ## use the python csv module to parse the file
                for i, row  in enumerate(reader):
                    if i>0:
                        counts = np.append(counts, float(row[0].split(" ")[1]))
                    if i == int(28*60/10): #break after reaching 28 min of data with 10s binwidths
                        break
    except IOError:
        log.error("Could not find the file '"+str(filename)+"'")

    return counts
