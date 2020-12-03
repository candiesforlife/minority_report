
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from shapely.geometry import Point


class Geo:


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
        lat_per_image = [[element[0] for element in crime] for crime in grouped]
        lon_per_image = [[element[1] for element in crime] for crime in grouped]
        return lat_per_image, lon_per_image
