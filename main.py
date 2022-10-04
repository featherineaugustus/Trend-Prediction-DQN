# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:44:20 2022

Deep Reinforcement Learning for Trading with TensorFlow 2.0

https://www.mlq.ai/deep-reinforcement-learning-for-trading-with-tensorflow-2-0/

@author: WeiYanPEH
"""

#%%############################################################################
# Import libraries
###############################################################################
import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model

from tqdm import tqdm_notebook, tqdm
from collections import deque

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf.debugging.set_log_device_placement(True)

# Suppress TPU messages which start with "Executing op"

#%%############################################################################
# Import User Defined Class and Functions
###############################################################################
from models import AI_Trader
from function import sigmoid
from function import stock_price_format
from function import dataset_loader
from function import state_creator  
from function import features_extraction
from function import plot_up_down
from function import reward_format 
from trading_decision import trading_decision


#%%############################################################################
# Create Folders
###############################################################################
if not os.path.exists('Data'):
    os.makedirs('Data/Train')
    os.makedirs('Data/Test')
    
if not os.path.exists('Results'):
    os.makedirs('Results/Train')
    os.makedirs('Results/Test')
    
if not os.path.exists('Results'):
    os.makedirs('Results/Train/Image')
    os.makedirs('Results/Test/Image')
    
if not os.path.exists('Log'):
    os.makedirs('Log/Train')
    os.makedirs('Log/Test')
    
if not os.path.exists('Checkpoints'):
    os.makedirs('Checkpoints/Train')
    os.makedirs('Checkpoints/Test')


#%%############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Initialize parameters
    window_size = 30
    episodes = 50
    batch_size = 32
    additional_feature = 0
    # additional_feature = window_size*5
    
    # Variables
    holding_period = 5
    trade_cost = 0.1

    # Training
    train_test = 'Train'
 
    # Load model, prepare to train
    trader = AI_Trader(state_size = window_size + additional_feature,
                       batch_size = batch_size)
    trader.model.summary()
    
    # Select dataset to train
    # data.to_csv('Data/' + company + '.csv',index=False)
    
    results = []
    
    ###########################################################################
    # Training
    ###########################################################################
    # Select dataset to test
    # company_list = ['AAPL', 'ADSK', 'AMD',
    #                 'AMZN', 'BLK', 'FB',
    #                 'GOOG', 'HPQ', 'IBM',
    #                 'JPM', 'META', 'MSFT',
    #                 'MU', 'NVDA', 'PYPL',
    #                 'TSLA', 'TWTR', 'USB',
    #                 'V'
    #                 ]
    company_list = ['AAPL', 'AMZN', 
                    'GOOG', 'META', 
                    'MSFT',
                    ]
    
    for company in company_list:
        dataset = dataset_loader(company, holding_period, train_test)
    
    
    results = []
    ###########################################################################
    # Training
    ###########################################################################
    for episode in range(1, episodes + 1):
        for company in company_list:
            dataset = pd.read_csv('Data/' + train_test + '/' + 
                                  company + '.csv')
            print('\nEpisode: {}/{}'.format(episode, episodes) + ' - ' + company)
            trader, results = trading_decision(dataset, company, 
                                               trader, 
                                               window_size, batch_size, trade_cost,
                                               results, train_test,
                                               episode)