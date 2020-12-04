
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from geopandas import GeoSeries
from datetime import datetime
from shapely.geometry import Point


class Geo:

    def __init__(self):
        self.data = None
        self.sample = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data



    def group_by_hour_list(df, year, month, day):#, sampling=True):
        '''
        get a sample of a month-time crimes grouped by hour
        inputs = start_date info
        '''
        sample = df.data[['period', 'latitude', 'longitude']]

        #if sampling:
        inf = sample['period'] > datetime(year, month, day, 0, 0, 0)
        next_month = month+1
        next_year = year
        if month == 12:
            next_month = 1
            next_year = year+1
        #print(next_year, next_month)
        sup = sample['period'] < datetime(next_year, next_month, day, 0, 0, 0)
        sample = sample[ inf & sup ]

        liste = np.sort(np.array(sample['period'].unique()))
        length = len(liste)
        lat_per_image = [[coord[1] for coord in np.array(sample[sample['period']== timestamp][['latitude', 'longitude']])]\
                 for index, timestamp in enumerate(liste)]
        lon_per_image = [[coord[0] for coord in np.array(sample[sample['period']== timestamp][['latitude', 'longitude']])]\
                 for index, timestamp in enumerate(liste)]

        return lat_per_image, lon_per_image


    def group_by_hour(df, year, month, day):#, sampling=True):
        '''
        get a sample of a month-time crimes grouped by hour
        inputs = start_date info
        '''
        sample = df.data[['period', 'latitude', 'longitude']]

        #if sampling:
        inf = sample['period'] > datetime(year, month, day, 0, 0, 0)
        next_month = month+1
        next_year = year
        if month == 12:
            next_month = 1
            next_year = year+1
        #print(next_year, next_month)
        sup = sample['period'] < datetime(next_year, next_month, day, 0, 0, 0)
        sample = sample[ inf & sup ]

        liste = np.sort(np.array(sample['period'].unique()))
        length = len(liste)
        lat_per_image = []
        lon_per_image = []
        for index, timestamp in enumerate(liste):
            if (index+1) % 100 ==0:
                print(f'Grouping timestamp {index+1}/{length}')
            by_hour = np.array(sample[sample['period']== timestamp][['latitude', 'longitude']])
            lat_per_image.append([coord[0] for coord in by_hour])
            lon_per_image.append([coord[1] for coord in by_hour])
        return lat_per_image, lon_per_image




    # 3 METHODEs TENSOR

    # def geoserie_to_tensor:
    #     pass


    # def list_geoseries_to_list_of_tensors:
    #     pass
    #     # return a list of tensors useful for model class


    # def save_list_tensors_to_pickle:
    #     pass



if __name__ == '__main__':
  print('initializing geo class')
  df = Geo()
  print("loading data")
  df.load_data()
  print('get a sample of a month-time crimes grouped by hour')
  lat, lon = df.group_by_hour(2016, 12, 17)
  print('Transforming into geoseries thanks to geopandas')
  df.get_geoseries(lat, lon)
  print('Finished!')

