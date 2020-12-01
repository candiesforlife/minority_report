'''Returns clean dataframe w/o NaN and with correct dtypes'''

import pandas as pd

from minority_report.data import NYPD
from ast import literal_eval

class Clean_Data:

    def __init__(self):
        # loads core dataframe
        self.data = NYPD().get_data()

    # removing Nans & weird values
    def drop_nan(self):
        """
        Returns a DataFrame w/o NaN
        """
        df = self.data.copy()
        pass

    def drop_miss(df):
        '''
        Drop precinct and time missing values
        '''
        #drop precinct missing values
        not_unknown = df['precinct_number'] != -99.0
        not_nan = df['precinct_number'] == df['precinct_number']
        df = df[not_unknown & not_nan]
        #drop time and date missing values
        df = df[df['time'] == df['time']]
        df = df[df['date'] == df['date']]
        return df

    #might have to be removed tomorrow
    def removing_date_before_2007(df):
        '''
        Remove any dates before 2007 & values not corresponding to our US date format month/day/year, returns pd.series
        '''
        rg_expression = r'(1[0-2]|0?[1-9])\/(3[01]|[12][0-9]|0?[1-9])\/20([1-2][0-9]|[0-2][7-9])'
        boolean_values = df['date'].str.match(rg_expression)
        return df[boolean_values]
    ################################################################


    def miss_lon_lat(df):
        '''
        replace latitude and longitude missing values with median values by precinct
        '''
        data = df.copy()
        for precinct in data['precinct_number'].unique():
            print(f'Modifying precinct{precinct}')
            geo = data[data['precinct_number'] == precinct][['latitude', 'longitude']]
            values = {'latitude': geo['latitude'].median(), 'longitude':geo['longitude'].median()}
            data[data['precinct_number']==precinct] = data[data['precinct_number']==precinct].fillna(value=values)
        return data

    def miss_victim(df):
        '''
        replace missing values by unknown value
        '''
        data = df.copy()
        age_liste = ['<18', '45-64', '18-24', '25-44']
        race_liste = ['BLACK', 'WHITE', 'UNKNOWN', 'WHITE HISPANIC', 'BLACK HISPANIC',
           'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE']
        sex_liste = ['M', 'F', 'D', 'E']
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





