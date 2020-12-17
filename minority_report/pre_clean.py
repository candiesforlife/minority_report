'''Returns dataframe to clean with unnecessary columns dropped'''

import os
import pandas as pd

from minority_report.data import NYPD


class PreClean:

    def __init__(self):
      '''
      Loads core dataframe
      '''
      self.data = NYPD().get_data()

    def drop_cols(self):
        '''
        Drops unnecessary columns, such as IDs
        '''
        df = self.data.copy()

        df.drop(columns = ['CMPLNT_NUM',
                        'CMPLNT_TO_DT',
                        'CMPLNT_TO_TM',
                        'HADEVELOPT',
                        'HOUSING_PSA',
                        'JURISDICTION_CODE',
                        'KY_CD',
                        'PD_CD',
                        'PD_DESC',
                        'RPT_DT',
                        'TRANSIT_DISTRICT',
                        'X_COORD_CD',
                        'Y_COORD_CD',
                        'Lat_Lon',
                       ], inplace = True)
