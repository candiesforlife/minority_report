'''Returns clean dataframe w/o NaN and with correct dtypes'''

import pandas as pd

from minority_report.data import NYPD
from ast import literal_eval

class CleanData:

    def __init__(self):
        # loads core dataframe
        self.data = NYPD().get_data()

    #  1.removing Nans & weird values
    def drop_nan(self):
        '''
        Drop precinct and time missing values, return a df without nan.
        '''

        df = self.data.copy()
        #drop precinct missing values
        not_unknown = df['precinct_number'] != -99.0
        not_nan = df['precinct_number'] == df['precinct_number']
        df = df[not_unknown & not_nan]
        #drop time and date missing values
        df = df[df['time'] == df['time']]
        df = df[df['date'] == df['date']]
        #drop offense_type and crime_completed
        df = df[df['offense_type'] == df['offense_type']]
        df = df[df['crime_completed'] == df['crime_completed']]
        return df

    #2.
    def miss_suspect (df):
        '''
            replace missing values by 'UNKNOWN' and returns a df.
        '''

        data = df.copy()
        age_liste = ['<18', '45-64', '18-24', '25-44', '65+']
        race_liste = ['BLACK', 'WHITE', 'WHITE HISPANIC', 'BLACK HISPANIC',
           'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE']
        sex_liste = ['M', 'F', 'U']
        data['suspect_age'] = [element if element in age_liste else 'UNKNOWN' for element in data['suspect_age']]
        data['suspect_race'] = [element if element in race_liste else 'UNKNOWN' for element in data['suspect_race']]
        data['suspect_sex'] = [element if element in sex_liste else 'UNKNOWN' for element in data['suspect_sex']]
        return data

    def miss_lon_lat(self):
        '''
        replace latitude and longitude missing values with median values by precinct
        '''
        df = self.data.copy()
        for precinct in data['precinct_number'].unique():
            #print(f'Modifying precinct{precinct}')
            geo = data[data['precinct_number'] == precinct][['latitude', 'longitude']]
            values = {'latitude': geo['latitude'].median(), 'longitude':geo['longitude'].median()} # get median lon and lat values as default for the precinct
            data[data['precinct_number']==precinct] = data[data['precinct_number']==precinct].fillna(value=values) # fill na with default values depending on precinct
        return data

    def miss_victim(self):
        '''
        replace missing values by unknown value
        '''
        df = self.data.copy()
        #values to keep
        age_liste = ['<18', '45-64', '18-24', '25-44', '65+']
        race_liste = ['BLACK', 'WHITE', 'UNKNOWN', 'WHITE HISPANIC', 'BLACK HISPANIC',
           'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE']
        sex_liste = ['M', 'F', 'D', 'E']
        #replace all others by unknown
        data['victim_age'] = [element if element in age_liste else 'UNKNOWN' for element in data['victim_age']]
        data['victim_race'] = [element if element in race_liste else 'UNKNOWN' for element in data['victim_race']]
        data['victim_sex'] = [element if element in sex_liste else 'UNKNOWN' for element in data['victim_sex']]
        return data

    #  Converting to correct datatypes
    def to_timestamp(self, column_name):
        '''
        Converts given column to datetime.time dtype, returns pd.series
        '''
        df = self.data.copy()
        df[column_name] = pd.to_datetime(df[column_name], format = '%H:%M:%S').dt.time
        return df[column_name]


    # might have to be modified tomorrow with some apply function recommended by Keurcien
    def to_date(self, column_name):
        '''
        Converts given column to datetime dtype, returns pd.series
        '''
        df = self.data.copy()
        df[column_name] = pd.to_datetime(df[column_name], format = '%m/%d/%Y')
        return df[column_name]
    ################################################################


    def round_int(series):
        """this functions rounds pd.series of int, up"""
        result = [round(x) for x in series]
        return result

    def complete_to_boolean(series):
        """turns complete/incomplete into boolean value"""
        result = series.replace({'COMPLETED': True, 'INCOMPLETE': False})
        return result

    def miss_borough (self):
        '''replace borough correct values depending on the precinct'''

        #replace borough depending on precinct_number
        df = self.data.copy()
        df['new_borough'] = [
            'MANHATTAN' if precinct <=34
            else 'BRONX' if precinct <=54
            else 'BROOKLYN' if precinct <= 94
            else 'QUEENS' if precinct <= 115
            else 'STATEN ISLAND'
            for precinct in df['precinct_number']
        ]

        #rename column
        df.drop(columns='borough', inplace=True)
        df.rename(columns={'new_borough':'borough'},inplace=True)
        return df


    def miss_patrol_borough (df):
        '''replace patrol_borough correct values depending on the precinct'''

        data = df.copy()
        # correct patrol borough
        bronx = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52]
        bklyn_south = [60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 76, 78]
        bklyn_north = [73, 75, 77, 79, 81, 83, 84, 88, 90, 94]
        man_south = [1, 5, 6, 7, 9, 10, 13, 14, 17, 18]
        man_north = [19, 20, 22, 23, 24, 25, 26, 28, 30, 32, 33, 34]
        queens_south = [100, 101, 102, 103, 105, 106, 107, 113]
        queens_north = [104, 108, 109, 110, 111, 112, 114, 115]
        staten = [120, 121, 122, 123]

        #replace with correct value
        data['new_patrol'] = [
            'PATROL BORO BRONX' if precinct in bronx
            else 'PATROL BORO BKLYN SOUTH' if precinct in bklyn_south
            else 'PATROL BORO BKLYN NORTH' if precinct in bklyn_north
            else 'PATROL BORO MAN SOUTH' if precinct in man_south
            else 'PATROL BORO MAN NORTH' if precinct in man_north
            else 'PATROL BORO QUEENS SOUTH' if precinct in queens_south
            else 'PATROL BORO QUEENS NORTH' if precinct in queens_north
            else 'PATROL BORO STATEN ISLAND'
            for precinct in data['precinct_number']
        ]

        data.drop(columns='patrol_borough', inplace=True)
        data.rename(columns={'new_patrol':'patrol_borough'},inplace=True)
        return data





