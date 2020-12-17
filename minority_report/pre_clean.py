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

        self.data = df
        return self.data

    def rename_cols(self):
        '''
        Renames dataset columns to more readable names
        '''

        df = self.data.copy()

        # dictionary of new names for each column
        columns_dic = {

            'CMPLNT_FR_DT' : 'date',
            'CMPLNT_FR_TM' : 'time',

            'LAW_CAT_CD' : 'offense_level',
            'OFNS_DESC' : 'offense_type',
            'CRM_ATPT_CPTD_CD' : 'crime_completed',

            'LOC_OF_OCCUR_DESC' : 'premise_desc',
            'PREM_TYP_DESC' : 'premise',
            'Latitude' : 'latitude',
            'Longitude' : 'longitude',
            'BORO_NM' : 'borough',
            'PATROL_BORO' : 'patrol_borough',
            'ADDR_PCT_CD' : 'precinct_number',
            'JURIS_DESC' : 'jurisdiction',

            'SUSP_AGE_GROUP' : 'suspect_age',
            'SUSP_RACE' : 'suspect_race',
            'SUSP_SEX' : 'suspect_sex',
            'VIC_AGE_GROUP' : 'victim_age',
            'VIC_RACE' : 'victim_race',
            'VIC_SEX' : 'victim_sex',
            'PARKS_NM' : 'park_name',
            'STATION_NAME' : 'metro'
        }

        df.rename(columns = columns_dic, inplace = True)

        self.data = df
        return self.data


