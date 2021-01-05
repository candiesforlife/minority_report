'''Splits clean_data df into train and test data'''

import os
import pickle
import pandas as pd

from datetime import datetime


class Split:

    def __init__(self):

        self.data = None

        self.train_df = None
        self.test_df = None


    def load_data(self):
        '''Load clean dataframe for 75th precinct.'''
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean-75-precinct.pickle')

        # gcp_pickle_path = 'clean-75-precinct.pickle'

        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)

        self.data = df
        return self.data

    def train_test_df(self):
        '''Split dataframe into train and test dataframes.'''
        df = self.data.copy()

        # creating train df
        inf_train = df['period'] >= datetime(2007, 1, 1, 0, 0, 0)

        sup_train = df['period'] < datetime(2016, 1, 1, 0, 0, 0)

        self.train_df = df[inf_train & sup_train]

        # creating test df
        inf_test = df['period'] >= datetime(2016, 1, 1, 0, 0, 0)

        sup_test = df['period'] <= datetime(2019, 10, 28, 0, 0, 0)

        self.test_df = df[inf_test & sup_test]

        return self.train_df, self.test_df

    def save_data(self):
        '''Save each dataframe to pickle.'''

        root_dir = os.path.dirname(os.path.dirname(__file__))

        train_pickle_path = os.path.join(root_dir, 'raw_data', 'train_df.pickle')
        test_pickle_path = os.path.join(root_dir, 'raw_data', 'test_df.pickle')


        with open(train_pickle_path, 'wb') as f:
            pickle.dump(self.train_df, f)

        with open(test_pickle_path, 'wb') as f:
            pickle.dump(self.test_df, f)

if __name__ == '__main__':
    print('Initializing Split Class')
    split = Split()
    print('Loading clean dataframe')
    split.load_data()
    print('Splitting into train and test dataframes')
    split.train_test_df()
    print('Saving train_df and test_df pickles')
    split.save_data()
    print('New pickles ready to use! :)')
