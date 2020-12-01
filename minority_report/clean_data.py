'''Returns clean dataframe w/o NaN and with correct dtypes'''

import pandas as pd

from minority_report.data import NYPD

class Clean_Data:

    def __init__(self):
        # loads core dataframe
        self.data = NYPD().get_data()

    def drop_nan(self):
        """
        Returns a DataFrame w/o NaN
        """
        df = self.data.copy()
        pass

    def to_timestamp(self, column_name):
        '''
        Converts given column to datetime.time dtype, returns pd.series
        '''
        df = self.data.copy()
        df[column_name] = pd.to_datetime(df[column_name], format = '%H:%M:%S').dt.time
        return df[column_name]

    def to_date(self, column_name):
        '''
        Converts given column to datetime dtype, returns pd.series
        '''
        df = self.data.copy()
        df[column_name] = pd.to_datetime(df[column_name], format = '%m/%d/%Y')
        return df[column_name]
