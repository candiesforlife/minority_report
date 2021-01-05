'''Generates new dataframe out of clean_data df for Data Viz purposes.'''

import os
import pickle
import pandas as pd

class Viz:

  def __init__(self):

    self.data = None


  def load_data(self):
    '''Load clean NYC dataframe from pickle file.'''
    root_dir = os.path.dirname(os.path.dirname(__file__))
    pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')

    with open(pickle_path, 'rb') as f:
        df = pickle.load(f)

    # Assigns clean.pickle to class variable data
    self.data = df
    return self.data


  def time_columns(self):
    '''Add time category columns to dataframe.'''
    df = self.data.copy()

    # New column: year
    df['year'] = df['period'].apply(lambda x: x.year)

    # New column: month (1 - 12)
    df['month'] = df['period'].apply(lambda x: x.month)

    # New column: DOW where 1 is Monday and 7 is Sunday
    df['day_of_week'] = df['period'].apply(lambda x: x.isoweekday())

    # New column: hour (0 - 23)
    df['hour'] = df['period'].apply(lambda x: x.hour)

    self.data = df
    return self.data


  def to_pickle(self):
    '''Save new viz dataframe as viz.pickle.'''
    root_dir = os.path.dirname(os.path.dirname(__file__))
    pickle_path = os.path.join(root_dir, 'raw_data', 'viz.pickle')

    with open(pickle_path, 'wb') as f:
       pickle.dump(self.data, f)


if __name__ == '__main__':

  print('Loading clean_data dataframe')
  df = Viz()
  df.load_data()

  print('Adding time columns')
  df.time_columns()

  print('Saving viz dataframe as pickle')
  df.to_pickle()

  print('Viz Dataframe ready to use! :)')
