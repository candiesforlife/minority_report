
'''Generates new dataframe out of clean dataframe for viz purposes'''

import os
import pickle
import pandas as pd

class Viz:

    def __init__(self):
        self.data = None

    def load_data(self):
        '''
        Loads clean df from pickle file
        '''
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')

        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)

        self.data = df
        return self.data

    def new_columns(self):
        '''
        Generates new time-related columns
        '''
        df = self.data.copy()

        # new columns for group by
        df['year'] = df['period'].apply(lambda x: x.year)
        df['month'] = df['period'].apply(lambda x: x.month)
        # DOW where 1 is Monday and 7 is Sunday
        df['day_of_week'] = df['period'].apply(lambda x: x.isoweekday())
        df['hour'] = df['period'].apply(lambda x: x.hour)

        self.data = df
        return self.data


    def to_pickle(self):
      '''
      Saves new dataframe as viz.pickle
      '''
      root_dir = os.path.dirname(os.path.dirname(__file__))
      pickle_path = os.path.join(root_dir, 'raw_data', 'viz.pickle')

      with open(pickle_path, 'wb') as f:
         pickle.dump(self.data, f)

if __name__ == '__main__':
  print('Loading Clean Data')
  df = Viz()
  df.load_data()
  print('Adding Columns')
  df.new_columns()
  print('Saving viz dataframe as pickle')
  df.to_pickle()
  print('Finished! :)')
