# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:35:11 2022

@author: Featherine
"""


#%%############################################################################
# Import libraries
###############################################################################
# import os
# import math
# import random
import numpy as np
import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
# from keras.models import load_model

# from tqdm import tqdm_notebook, tqdm
# from collections import deque


#%%############################################################################
# Import User Defined Class and Functions
###############################################################################
# from models import AI_Trader
from function import sigmoid
from function import stock_price_format
# from function import dataset_loader
from function import state_creator  
from function import features_extraction
from function import plot_up_down
from function import reward_format 


#%%############################################################################
# Functions
###############################################################################
def trading_decision(dataset, company, 
                     trader, window_size, batch_size, trade_cost,
                     results, train_test,
                     episode=1):
    '''
    Parameters
    ----------
    dataset : DataFrame
        A dataframe containing all the information of the company.
    company : str
        The company name.
    trader : Class
        The class of the AITrader
    window_size : int
        The window length to backtrack in order to compute the state.
    batch_size : int
        The batch size used for training.
    results : list
        The result list.
    train_test : str
        Either 'Train' or 'Test', to indicate what the system is used for.
    episode : TYPE, optional
        The index used. The default is 1.

    Returns
    -------
    trader : Class
        The updated class of the AITrader.
    results : list
        The result list appended with the latest results
    '''
    
    action_dict = {0: 'UP',
                   1: 'DW'}
    
    
    # Counters
    total_up, total_dw = 0, 0           # Total count of up/dw
    counter_up, counter_dw = 0, 0       # Consecutive count of up/dw
    total_correct, total_wrong = 0, 0   # Total count of correct/wrong
    total_reward = 0                    # Total reward
    tp, tn, fp, fn = 0, 0, 0, 0         # Total count of tp, tn, fp, fn
    
    # Initial Values
    trader.inventory = []               # Inventory (UNUSED)  
    
    # Log -> Save outputs
    log = []        # Log files 
    states_up = []  # All the up state
    states_dw = []  # All the down state
    

    
    # Looping through the time series
    # for t in range(0, len(dataset)-1):
        
    # Looping through the time series
    # We start from window_size, as we need that amount of data
    # to get the state
    # for t in range(window_size-1, len(dataset)-1):
    for t in range(len(dataset)-1):
        
        # Compute state
        state = state_creator(dataset, t, window_size+1)
        
        action = trader.trade(state)   # Get current action
        reward = 0                     # Initialize reward to 0
        prediction = None              # Initalize prediction to None
        
        # Get price and rates
        current_price = dataset['Close'][t]           # Current price of the unit
        forward_return = dataset['Forward Return'][t] # Forward return
        # true_future = dataset['Up/Dw'][t]
        rate = dataset['Rate'][t]                     # % Gain
        rate -= trade_cost                            # % Gain after substracting trade cost
    
        # Check if it is a up or down
        if rate > 0:
            true_future = 'UP'
        else:
            true_future = 'DW'
        
        # Check if prediction is correct or wrong
        if action_dict[action] == true_future:
            prediction = 'CORRECT'
            total_correct += 1
        else:
            prediction = 'WRONG  '
            total_wrong += 1
            
        #######################################################################
        # Compute reward for UP action
        #######################################################################
        if action == 0:
            
            states_up.append(t)                   # Keep track of buy state
            total_up += 1
            counter_up += 1
            counter_dw = 0

            if action_dict[action] == true_future:
                tp += 1
                reward = rate
            else:
                fp +=1
                reward = rate

            print('Day ' + str(t+1) + ': Rate ' + reward_format(rate) + 
                  ': True ' + true_future + ' Pred UP - ' + prediction + 
                  ' - Reward: ' + reward_format(reward))
            
        #######################################################################
        # Compute reward for DW action
        #######################################################################
        elif action == 1:
            
            states_dw.append(t)               # Keep track of sell state
            total_dw += 1
            counter_dw += 1
            counter_up = 0
            
            if action_dict[action] == true_future:
                tn +=1
                reward = -rate
            else:
                fn +=1
                reward = -rate
            
            print('Day ' + str(t+1) + ': Rate ' + reward_format(rate) + 
                  ': True ' + true_future + ' Pred DW - ' + prediction + 
                  ' - Reward: ' + reward_format(reward))

        #######################################################################
        # Check if the forecasting ends
        #######################################################################
        done = True if (t == len(dataset) - 1) else False
          
        
        #######################################################################
        # Compute next state
        #######################################################################
        state_next = state_creator(dataset, t+1, window_size+1)

        
        #######################################################################
        # Upload data into trader memory
        #######################################################################
        # reward = 2*(sigmoid(reward) - 0.5)
        trader.memory.append((state, action, reward, state_next, done))
        # state = state_next # Update state
        

        #######################################################################
        # Train model
        #######################################################################
        if ((len(trader.memory) > batch_size) and # If have enough memory to train
            # (t%10 == 0) and                       # Train once every 10 step
            (train_test == 'Train')):             # Only during training mode
            trader.batch_train()
        

        #######################################################################
        # Compute performance statistics
        #######################################################################
        total_reward += reward
        total_all = total_up + total_dw
        
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn)>0 else 0
        sen = tp / (tp + fn) if (tp + fn)>0 else 0
        spe = tn / (tn + fp) if (tn + fp)>0 else 0
        bac = 0.5*(sen+spe)
        cm = [[tn,fp], [fn,tp]]
        
        
        #######################################################################
        # Upload log
        #######################################################################  
        log.append([company, t+1, current_price, forward_return, rate,
                    action, action_dict[action], true_future,
                    reward, total_reward,
                    counter_up, counter_dw, 
                    total_all,
                    total_up, total_dw,
                    total_up/total_all,
                    total_dw/total_all,
                    total_correct, total_wrong,
                    total_correct/total_all, 
                    total_wrong/total_all,
                    tp, tn, fp, fn,
                    acc, sen, spe, bac, str(cm)
                    ])
        
    ###########################################################################
    # Print Results
    ###########################################################################
    print('\n############################################')
    print('# ' + company)
    print('############################################')
    print('# TOTAL REWARD  : ' + reward_format(total_reward))
    print('# TOTAL CORRECT :  ' + str(total_correct))
    print('# TOTAL WRONG   :  ' + str(total_wrong))
    print('############################################')
    print('# ACC : ' + str(round(acc,3)))
    print('# BAC : ' + str(round(bac,3)))
    print('# SEN : ' + str(round(sen,3)))
    print('# SPE : ' + str(round(spe,3)))
    print('# CM  : ' , cm)
    print('############################################')
    
    
    ###########################################################################
    # Save Results
    ###########################################################################
    results.append([company, episode, total_reward,       
                    total_all,
                    total_up, total_dw, 
                    total_up/total_all,
                    total_dw/total_all,
                    total_correct, total_wrong,
                    total_correct/total_all, 
                    total_wrong/total_all,
                    tp, tn, fp, fn,
                    acc, sen, spe, bac, str(cm)
                    ])
    
    columns = ['Company', 'Epoch', 'Total Reward',
               'ALL',
               'UP', 'DOWN',
               'UP R', 'DOWN R',
               'CORRECT', 'WRONG',
               'CORRECT R', 'WRONG R',
               'TP', 'TN', 'FP', 'FN',
               'ACC', 'SEN', 'SPE', 'BAC', 'CM',
               ]
    
    results_save = pd.DataFrame(results, columns = columns)
    results_save.to_csv('Results/' + train_test + 
                        '/Results.csv', index=False)
  
    ###########################################################################
    # Save Log
    ###########################################################################
    columns = ['Company', 'Day', 'Current Price', 'Forward Return', 'Rate',
               'Action', 'Action', 'True Action',
               'Reward', 'Total Reward',
               'Counter UP', 'Counter DOWN',
               'ALL',
               'UP', 'DOWN', 
               'UP R', 'DOWN R',
               'CORRECT', 'WRONG',
               'CORRECT R', 'WRONG R',
               'TP', 'TN', 'FP', 'FN',
               'ACC', 'SEN', 'SPE', 'BAC', 'CM',
               ]
    
    log_save = pd.DataFrame(log, columns = columns)
    log_save.to_csv('Log/' + train_test + 
                    '/Log ' + str(episode) + '.csv', index=False)
        
    ###########################################################################
    # Plot results
    ###########################################################################   
    plot_up_down(company, dataset, episode,
                 states_up, states_dw, 
                 total_reward, 
                 train_test)
    
    ###########################################################################
    # Save Model
    ###########################################################################
    if train_test == 'Train':
        trader.model.save('Checkpoints/' + train_test + 
                          '/ai_trader_{}.h5'.format(episode))
        
    return trader, results