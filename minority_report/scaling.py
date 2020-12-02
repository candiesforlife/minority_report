'''Returns scaled dataframe for numerical and categorical values'''

import os
import itertools
import pandas as pd

from minority_report.clean_data import CleanData
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class Scaling:

    def __init__(self):
        self.data = CleanData().total_clean()

    def label_encoding(self):
        '''
        Scaling column crime_completed with label_encoder technique. Returns the df updated.
        '''
        le = LabelEncoder()
        self.data['crime_completed'] = le.fit_transform(self.data['crime_completed'])
        return self.data

    def one_hot_encoding(self,features_list):
        '''
        Scaling categorical columns with one_hot_encoder technique. Returns the df updated.
        '''
        ohe = OneHotEncoder(sparse = False)
        ohe.fit(self.data[features_list])
        list_of_list_of_columns = [list(element) for element in ohe.categories_]
        columns = list(itertools.chain(*list_of_list_of_columns))
        et_oh = ohe.transform(self.data[features_list])
        self.data[columns] = et_oh
        return self.data

    def save_data(self, file_name):
      '''
      Saves encoded clean dataframe as a csv file.
      '''
      root_dir = os.path.dirname(os.path.dirname(__file__))
      csv_path = os.path.join(root_dir, 'raw_data', f'{file_name}.csv')
      return self.data.to_csv(csv_path, index = False)

if __name__ == '__main__':
  print('initializing sclaing object with clean data')
  df = Scaling()
  print('Label encoding')
  df.label_encoding()
  print('One hot encoding')
  features_list = ['offense_type','offense_level','borough','premise_desc','premise','suspect_age','suspect_race','suspect_sex','patrol_borough', 'metro','victim_age','victim_race','victim_sex','precinct_number']
  df.one_hot_encoding(features_list)
  print('Saving the encoding data as a csv file')
  df.save_data('data_encoded')
  print('finished!')


