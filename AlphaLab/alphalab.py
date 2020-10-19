from PhysicsNum import GaussianFit, Linearreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv                               ## for reading in our data files
import logging                           ## for orderly print output
import sys                               ## useful system calls (used to exit cleanly)
import os                                ## path manipulations

def ImportFunction(Path1, Path2):
    try:
        DataFrame1 = pd.read_csv(Path1)
        #DataFrame1 = DataFrame1[0:-1]
        DataFrame2 = pd.read_csv(Path2)
    except Exception as E:
        raise E
    """
    for index, row, in DataFrame2.iterrows():
        print(index)
    """
    print(DataFrame2)
    """
    for index, row in DataFrame1.iterrows():
        print(index)
    """
    return DataFrame1, DataFrame2


Value = ImportFunction('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/Amaricium80V.Spe', '/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/Amaricium80V.csv')
