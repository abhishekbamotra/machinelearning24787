from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import InputLayer, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
# DF = pd.read_csv("C:/Users/mohan/keystroke.csv",header=None)
DF = pd.read_csv("keystroke.csv",header=None)

DF.columns = DF.iloc[0]
DF=DF.drop([0], axis=0)
DF=DF.drop(['sessionIndex', 'rep'], axis=1)

