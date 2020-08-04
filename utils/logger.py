import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import os
import csv

class DataLog:
    """
    Logs data
    Most of the data is dictionaries and each dictionary consists of a list
    """
    def __init__(self):
        self.log = {}
        self.max_len = 0

    def log_kv(self, key, value):
        """
        Logs a particular piece of data
        :param key : Name of the data you want to log
        :param value : Value of the data you want to log
        :return : Doesn't return anything. The data is logged into the objects dictionary 
        """
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path):
        """
        Saves the log data as a oickle gile and a csv file
        :param save_path : This is location you want to save your data
        :return : nothing. Creates 2 files, a pickle file and a csv file.
        """
        pickle.dump(self.log, open(save_path+'/log.pickle', 'wb'))
        with open(save_path+'/log.csv', 'w') as csv_file:
            fieldnames = self.log.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        """
        Returns the latest piece of logged data
        :param :None
        :returns : A dictionary of containing the latest data for each logged variable
        """
        row_dict = {}
        for key in self.log.keys():
            row_dict[key] = self.log[key][-1]
        return row_dict

    def read_log(self, log_path):
        """
        Creates a dictionary out of a csv file (pickle is direct)
        :param log_path: Path of the CSV file
        :returns Nothing: Copies the logged dictionary onto the objects dictionary
        """
        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row in listr:
                for key in keys:
                    try:
                        data[key].append(eval(row[key]))
                    except:
                        None
        self.log = data

if(__name__ == "__main__"):
    logger = DataLog()
    logger.log_kv('speed', 10)
    logger.log_kv('age', 5)
    print(logger.log)
    logger.log_kv('speed', 20)
    print(logger.log)