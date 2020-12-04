'''Returns scaled dataframe for numerical and categorical values'''

import os
import itertools
import pickle
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder

class Scaling:

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

    #def label_encoding(self):
    #    '''
    #    Scaling column crime_completed with label_encoder technique. Returns the df updated.
    #    '''
    #    le = LabelEncoder()
    #    self.data['crime_completed'] = le.fit_transform(self.data['crime_completed'])
    #    return self.data

    def one_hot_encoding(self,features_list):
        '''
        Scaling categorical columns with one_hot_encoder technique. Returns the df updated.
        '''
        ohe = OneHotEncoder(sparse = False)
        print('first step')
        ohe.fit(self.data[features_list])
        print('second step')
        list_of_list_of_columns = [list(element) for element in ohe.categories_]
        print('third step')
        columns = list(itertools.chain(*list_of_list_of_columns))
        print('fourth step')
        et_oh = ohe.transform(self.data[features_list])
        print('fifth step')
        self.data[columns] = et_oh
        print('sixth step')
        return self.data

    def save_data(self, file_name):
      '''
      Saves encoded clean dataframe as a csv file.
      '''
      root_dir = os.path.dirname(os.path.dirname(__file__))
      csv_path = os.path.join(root_dir, 'raw_data', f'{file_name}.csv')
      return self.data.to_csv(csv_path, index = False)

    def to_pickle(self):

      root_dir = os.path.dirname(os.path.dirname(__file__))
      pickle_path = os.path.join(root_dir, 'raw_data', 'encoded.pickle')

      with open(pickle_path, 'wb') as f:
         pickle.dump(self.data, f)


if __name__ == '__main__':
  print('Initializing scaling object with clean data')
  df = Scaling()
  df.load_data()
  print('One hot encoding all the categories columns')
  features_list = ['crime_completed', 'offense_type','offense_level','borough','premise_desc','premise','suspect_age','suspect_race','suspect_sex','patrol_borough', 'metro','victim_age','victim_race','victim_sex','precinct_number']
  df.one_hot_encoding(features_list)
  print('Saving the encoding data as pickle')
  df.to_pickle()
  print('Finished!')
