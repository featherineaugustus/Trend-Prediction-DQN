# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:35:52 2022

@author: Featherine
"""


#%%############################################################################
# Import libraries
###############################################################################
# import os
# import math
# import random
# import numpy as np
import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
# from keras.models import load_model

# from tqdm import tqdm_notebook, tqdm
# from collections import deque


#%%############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Train
    df = pd.read_csv('Results/Train/Results.csv')

    df = pd.DataFrame(df.groupby(['Epoch'], as_index=False).mean()
                                 .groupby('Epoch')['BAC', 'Total Reward'].median())
    df = df.sort_values(['BAC'], ascending=False)    
    print(df)    
    df.to_csv('Results/Best Model Train.csv')
    
    # Test
    df = pd.read_csv('Results/Test/Results.csv')

    df = pd.DataFrame(df.groupby(['Epoch'], as_index=False).mean()
                                 .groupby('Epoch')['BAC', 'Total Reward'].median())
    df = df.sort_values(['BAC'], ascending=False)    
    print(df)    
    df.to_csv('Results/Best Model Test.csv')
    
    