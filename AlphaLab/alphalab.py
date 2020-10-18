from PhysicsNum import GaussianFit, Linearreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv                               ## for reading in our data files
import logging                           ## for orderly print output
import sys                               ## useful system calls (used to exit cleanly)
import os                                ## path manipulations

def ImportFunction(Path):
    try:
        DataFrame = pd.read_csv(Path)
        DataFrame = DataFrame[0:-1]
    except Exception as E:
        raise E
    for index, row in DataFrame.iterrows():
        print(str(row))

        print(index)

    return DataFrame


Value = ImportFunction('/Users/andreasevensen/Documents/GitHub/Fysc12/AlphaLab/Amaricium80V.Spe')
