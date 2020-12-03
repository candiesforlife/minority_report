
import pandas as pd
import matplotlib.pyplot as plt

from geopandas import GeoSeries
from datetime import datetime
from shapely.geometry import Point


class Geo:

    def __init__(self):
        self.data = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data

    def group_by_hour(df, year, month, day):
        '''
        get a sample of a month-time crimes grouped by hour
        inputs = start_date info
        '''
        sample = df.data[['period', 'latitude', 'longitude']]
        inf = sample['period'] > datetime(year, month, day, 0, 0, 0)
        next_month = month+1
        next_year = year
        if month == 12:
            next_month = 1
            next_year = year+1
        print(next_year, next_month)
        sup = sample['period'] < datetime(next_year, next_month, day, 0, 0, 0)
        sample = sample[ inf & sup ]
        liste = np.sort(np.array(sample['period'].unique()))
        grouped = []
        length = len(liste)
        for index, timestamp in enumerate(liste):
            if (index+1) % 10 ==0:
                print(f'Grouping timestamp {index+1}/{length}')
            by_hour = np.array(sample[sample['period']== timestamp][['latitude', 'longitude']])
            grouped.append(by_hour)
        latitude_per_image = [[element[0] for element in crime] for crime in grouped]
        longitude_per_image = [[element[1] for element in crime] for crime in grouped]
        return latitude_per_image, longitude_per_image


    def geoseries_images(latitude_per_image, longitude_per_image):
        final_list_images =  []
        for lat, lon in zip(longitude_per_image, latitude_per_image):
            geometry = [Point(xy) for xy in zip(lon, lat)]
            df_geopandas = sample.drop(['longitude', 'latitude'], axis=1)
            geoseries_image = GeoSeries(geometry)
            final_list_images.append(geoseries_image)
        return final_list_images

    def visualization_images(final_list_images):
        for image in final_list_images:
            fig,ax = plt.subplots(figsize = (10,10))
            g = image.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = 'NYC')
            plt.show()

