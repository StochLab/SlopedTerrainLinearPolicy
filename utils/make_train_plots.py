import sys, os
sys.path.append(os.path.realpath('../'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
from utils.logger import DataLog

def make_train_plots(log = None,
                     log_path = None,
                     keys = None,
                     save_loc = None):
    """
    Plots and saves images of all logged data
    :param log : A dictionary containing lists of data we want
    :param log_path : Path to a csv file that contains all the logged data
    :param keys : The keys of the dictionary, for the data you want to plot
    :param save_loc : Location where you want to save the images
    :returns : Nothing, saves the figures of the plot
    """
    if log is None and log_path is None:
        print("Need to provide either the log or path to a log file")
    if log is None:
        logger = DataLog()
        logger.read_log(log_path)
        log = logger.log
    # make plots for specified keys
    for key in keys:
        if key in log.keys():
            plt.figure(figsize=(10,6))
            plt.plot(log[key])
            plt.title(key)
            plt.savefig(save_loc+'/'+key+'.png', dpi=100)
            plt.close()


def make_train_plots_ars(log = None,
                     log_path = None,
                     keys = None,
                     save_loc = None):
    """
    Plots and saves images of all logged data
    :param log : A dictionary containing lists of data we want
    :param log_path : Path to a csv file that contains all the logged data
    :param keys : The keys of the dictionary, for the data you want to plot
    :param save_loc : Location where you want to save the images
    :returns : Nothing, saves the figures of the plot
    """
    if log is None and log_path is None:
        print("Need to provide either the log or path to a log file")
    if log is None:
        logger = DataLog()
        logger.read_log(log_path)
        log = logger.log

    plt.figure(figsize=(10,6))
    plt.plot(log[keys[0]], log[keys[1]])
    plt.title(keys[1])
    plt.savefig(save_loc+'/'+keys[1]+'.png', dpi=100)
    plt.close()

def plot_traj(logger, keys_x, keys_y, titles = None,save_loc = None):
    """
    Function meant to plot foot trajectories but can be used for any 2 dimensional plotting
    :param logger : Object of class DataLog present in logger.py
    :param keys_x : List of values that contains the keys to be used in x axis of plotting
    :param keys_y : List of values that contains the keys to be used in y axis of plotting, needs to be same size as keys_x
    :param titles : Titles desired for each plot, needs to be same size as keys_x
    :param save_loc : Path to the save the image, can be relative 
    :return : Saves a bunch of figures in required path
    """
    for i in range(len(keys_x)):
        plt.figure(figsize=(10,6))
        plt.plot(logger.log[keys_x[i]], logger.log[keys_y[i]])
        if(titles is None):
            titles = range(len(keys_x))
        plt.title(str(titles[i]))
        plt.savefig(save_loc+'/'+str(titles[i])+'.png', dpi=100)
        plt.close()