
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



    def group_by_hour(self,year, month, day):
        '''
        get a sample of a month-time crimes grouped by hour
        inputs = start_date info
        '''
        sample = self.data[['period', 'latitude', 'longitude']]
        inf = sample['period'] > datetime(year, month, day, 0, 0, 0)
        next_month = month+1
        next_year = year
        if month == 12:
            next_month = 1
            next_year = year+1
        # print(next_year, next_month)
        sup = sample['period'] < datetime(next_year, next_month, day, 0, 0, 0)
        self.sample = sample[ inf & sup ]
        liste = np.sort(np.array(self.sample['period'].unique()))
        grouped = []
        length = len(liste)
        for index, timestamp in enumerate(liste):
            if (index+1) % 100 ==0:
                print(f'Grouping timestamp {index+1}/{length}')
            by_hour = np.array(self.sample[self.sample['period']== timestamp][['latitude', 'longitude']])
            grouped.append(by_hour)
        latitude_per_image = [[element[0] for element in crime] for crime in grouped]
        longitude_per_image = [[element[1] for element in crime] for crime in grouped]
        return latitude_per_image, longitude_per_image


    def get_geoseries(self, latitude_per_image, longitude_per_image):
        final_list_geoseries =  []
        sample = self.sample
        for lat, lon in zip(longitude_per_image, latitude_per_image):
            geometry = [Point(xy) for xy in zip(lon, lat)]
            df_geopandas = sample.drop(['longitude', 'latitude'], axis=1)
            geoseries_image = GeoSeries(geometry)
            final_list_geoseries.append(geoseries_image)
        return final_list_geoseries

    def visualization_from_geoseries_to_images(self,final_list_geoseries):
        for geoserie in final_list_geoseries:
            fig,ax = plt.subplots(figsize = (10,10))
            g = geoserie.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = 'NYC')
            plt.show()

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

