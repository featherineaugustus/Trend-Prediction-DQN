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
# import os
# import math
# import random
# import numpy as np
import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
from keras.models import load_model

# from tqdm import tqdm_notebook, tqdm
# from collections import deque

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf.debugging.set_log_device_placement(True)


#%%############################################################################
# Import User Defined Class and Functions
###############################################################################
from models import AI_Trader
# from function import sigmoid
# from function import stock_price_format
from function import dataset_loader
# from function import state_creator  
# from function import features_extraction
# from function import plot_up_down
from function import reward_format 
from trading_decision import trading_decision


#%%############################################################################
# Main
###############################################################################
if __name__ == "__main__":   
    # Initialize parameters
    window_size = 10
    episodes = 50
    batch_size = 32
    additional_feature = 0
    # additional_feature = window_size*5
    
    # Variables
    holding_period = 5
    trade_cost = 0.1
    
    # Testing, no training
    train_test = 'Test'
    
    # Find best model
    # best_model = pd.read_csv('Results/Best Model Train.csv')
    # best_index = int(best_model['Epoch'][0])
    # best_index = 30
    

    # Select dataset to test
    company_list = ['AAPL', 'ADSK', 'AMD',
                    'AMZN', 'BLK', 'FB',
                    'GOOG', 'HPQ', 'IBM',
                    'JPM', 'META', 'MSFT',
                    'MU', 'NVDA', 'PYPL',
                    'TSLA', 'TWTR', 'USB',
                    'V'
                    ]
    # company_list = ['AAPL', 'AMZN', 
    #                 'GOOG', 'META', 
    #                 'MSFT',
    #                 ]
    
    for company in company_list:
        dataset = dataset_loader(company, holding_period, train_test)
    
    results = []
    ###########################################################################
    # Testing
    ###########################################################################
    for episode in range(1, episodes + 1):
    # if True:
        for company in company_list:
            # print('\nComapny: ' + company)
            print('\nEpisode: {}/{}'.format(episode, episodes) + ' - ' + company)
            
            
            # Load data and model, no training
            trader = AI_Trader(state_size = window_size + additional_feature,
                               batch_size = batch_size)
            trader.model = load_model('Checkpoints/Train' + 
                                      '/ai_trader_{}.h5'.format(episode))
            
          
            dataset = pd.read_csv('Data/' + train_test + '/' + 
                                  company + '.csv')
            
            
            trader, results = trading_decision(dataset, company, 
                                               trader, 
                                               window_size, batch_size, trade_cost,
                                               results, train_test,
                                               episode)