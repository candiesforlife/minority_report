'''Returns clean dataframe w/o NaN and with correct dtypes'''

import os
import pandas as pd
from datetime import datetime
import pickle

from minority_report.data import NYPD
from minority_report.utils import round_six_hours
#from ast import literal_eval

class CleanData:

    def __init__(self):
      # loads core dataframe
      self.data = NYPD().get_data()
      self.precinct = None

    #  1.
    def drop_nan(self):
      '''
      Returns a dataframe without NaN
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
      self.data = df
      return self.data


    def to_date_format(self):
      '''
      Merges date and time into new column, drops old
      Filters dataframe to show only complaints dated 2007 onwards
      '''
      df = self.data.copy()
      # concat to new period column
      df['period'] = df['date'] + ' ' + df['time']
      # change to datetime
      df['period'] = df['period'].apply(lambda x: \
                             datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))
      # round to hour bucket
      df['period'] = df['period'].apply(lambda x: x.replace(minute = 0, second = 0))
      # filter to post 2007
      df = df[df['period'] > datetime(2006, 12, 31, 23, 59, 0)]
      # drop date and time
      df.drop(columns = ['date', 'time'], inplace = True)
      self.data = df
      return self.data


    #4.
    def miss_suspect(self):
      '''
      Returns dataframe where missing values are replaced by 'UNKNOWN'
      '''

      df = self.data.copy()
      age_liste = ['<18', '45-64', '18-24', '25-44', '65+']
      race_liste = ['BLACK', 'WHITE', 'WHITE HISPANIC', 'BLACK HISPANIC',
         'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE']
      sex_liste = ['M', 'F', 'U']
      df['suspect_age'] = [element if element in age_liste else 'UNKNOWN' for element in df['suspect_age']]
      df['suspect_race'] = [element if element in race_liste else 'UNKNOWN' for element in df['suspect_race']]
      df['suspect_sex'] = [element if element in sex_liste else 'UNKNOWN' for element in df['suspect_sex']]
      self.data = df
      return self.data

    #5.
    def miss_victim(self):
      '''
      Returns dataframe where missing values are replaced by 'UNKNOWN'
      '''
      df = self.data.copy()
      #values to keep
      age_liste = ['<18', '45-64', '18-24', '25-44', '65+']
      race_liste = ['BLACK', 'WHITE', 'UNKNOWN', 'WHITE HISPANIC', 'BLACK HISPANIC',
         'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE']
      sex_liste = ['M', 'F', 'D', 'E']
      #replace all others by unknown
      df['victim_age'] = [element if element in age_liste else 'UNKNOWN' for element in df['victim_age']]
      df['victim_race'] = [element if element in race_liste else 'UNKNOWN' for element in df['victim_race']]
      df['victim_sex'] = [element if element in sex_liste else 'UNKNOWN' for element in df['victim_sex']]
      self.data = df
      return self.data

    # 6.
    def miss_premise(self):
      '''
      Returns dataframe where missing values are replaced by 'UNKNOWN'
      '''
      df = self.data.copy()
      #values to keep
      premise = ['RESIDENCE - PUBLIC HOUSING', 'RESIDENCE-HOUSE', 'PUBLIC SCHOOL',
         'RESIDENCE - APT. HOUSE', 'DOCTOR/DENTIST OFFICE',
         'DEPARTMENT STORE', 'STREET', 'DRUG STORE', 'CHAIN STORE',
         'RESTAURANT/DINER', 'HOSPITAL', 'TRANSIT - NYC SUBWAY',
         'FOOD SUPERMARKET', 'BUS (NYC TRANSIT)', 'COMMERCIAL BUILDING',
         'GAS STATION', 'FACTORY/WAREHOUSE', 'CLOTHING/BOUTIQUE',
         'PRIVATE/PAROCHIAL SCHOOL', 'TELECOMM. STORE', 'GROCERY/BODEGA',
         'BEAUTY & NAIL SALON', 'OTHER', 'PUBLIC BUILDING',
         'SOCIAL CLUB/POLICY', 'TAXI (LIVERY LICENSED)',
         'PARKING LOT/GARAGE (PUBLIC)', 'FAST FOOD', 'BAR/NIGHT CLUB',
         'PARKING LOT/GARAGE (PRIVATE)','PARK/PLAYGROUND', 'BANK',
         'HOTEL/MOTEL', 'BUS TERMINAL', 'VARIETY STORE', 'SMALL MERCHANT',
         'HIGHWAY/PARKWAY', 'CHURCH', 'STORE UNCLASSIFIED',
         'DRY CLEANER/LAUNDRY', 'TUNNEL', 'MARINA/PIER', 'AIRPORT TERMINAL',
         'CANDY STORE', 'ATM', 'GYM/FITNESS FACILITY',
         'TAXI (YELLOW LICENSED)', 'JEWELRY', 'SYNAGOGUE',
         'OPEN AREAS (OPEN LOTS)', 'LIQUOR STORE', 'PHOTO/COPY',
         'BOOK/CARD', 'STORAGE FACILITY', 'VIDEO STORE',
         'FERRY/FERRY TERMINAL', 'SHOE', 'LOAN COMPANY', 'BUS (OTHER)',
         'CHECK CASHING BUSINESS', 'TRANSIT FACILITY (OTHER)',
         'MAILBOX OUTSIDE', 'MOSQUE', 'BUS STOP', 'CONSTRUCTION SITE',
         'BRIDGE', 'ABANDONED BUILDING', 'TAXI/LIVERY (UNLICENSED)',
         'OTHER HOUSE OF WORSHIP', 'CEMETERY', 'TRAMWAY']
      premise_desc = ['INSIDE', 'FRONT OF', 'REAR OF', 'OPPOSITE OF', 'OUTSIDE']
      #replace values
      df['premise'] = [element if element in premise else 'UNKNOWN' for element in df['premise']]
      df['premise_desc'] = [element if element in premise_desc else 'UNKNOWN' for element in df['premise_desc']]
      self.data = df
      return self.data

    #7.
    def miss_park_metro(self):
      '''
      Returns dataframe where missing values are replaced by:
      'NOT PARK' or 'NOT SUBWAY'
      '''
      df = self.data.copy()
      park_list = ['JACKIE ROBINSON PARK MANHATTAN', 'TOMPKINS SQUARE PARK',
         'FLUSHING MEADOWS CORONA PARK', 'BIG BUSH PARK',
         'LEIF ERICSON PARK', 'JOYCE KILMER PARK',
         'COLONEL DAVID MARCUS PLAYGROUND', 'BAISLEY POND PARK',
         'GRAVESEND PARK', 'WILLIAMSBRIDGE OVAL',
         'ROCKAWAY BEACH AND BOARDWALK', 'HAMILTON FISH PARK',
         'POWER PLAYGROUND', 'QUEENSBRIDGE PARK',
         'A PHILIP RANDOLPH SQUARE', 'CENTRAL PARK', 'SUNSET PARK',
         'FREDERICK DOUGLASS PLAYGROUND', 'CLAREMONT PARK', 'BRONX PARK',
         'ST. JAMES PARK', 'RIVERSIDE PARK', 'DYKER BEACH PARK',
         'PELHAM BAY PARK', 'UNION SQUARE PARK', 'WASHINGTON SQUARE PARK',
         'HIGHBRIDGE PARK MANHATTAN SIDE', 'HOFFMAN PARK',
         'BROOKLYN BRIDGE PARK', 'ST. NICHOLAS PARK', 'MACOMBS DAM PARK',
         "ROSEMARY'S PLAYGROUND", 'PARK OF THE AMERICAS', 'MCCARREN PARK',
         'ECHO TRIANGLE', 'THOMAS GREENE PLAYGROUND', 'ST. ALBANS PARK',
         'CONEY ISLAND BEACH & BOARDWALK', 'SEWARD PARK',
         'LINCOLN TERRACE / ARTHUR S. SOMERS PARK', 'MARCUS GARVEY PARK',
         'VAN CORTLANDT PARK', 'COFFEY PARK', 'MOORE HOMESTEAD PLAYGROUND',
         "RANDALL'S ISLAND PARK", 'CHELSEA PARK', 'BRIDGE PLAYGROUND',
         'ROCKAWAY COMMUNITY PARK', 'CRISPUS ATTUCKS PLAYGROUND',
         'FATHER DUFFY SQUARE', 'PLAYGROUND 52 LII',
         'DOWNING STREET PLAYGROUND', 'LINDEN PARK', 'MCKINLEY PARK',
         'CALVERT VAUX PARK', 'ROY WILKINS RECREATION CENTER',
         'GIVAN SQUARE', 'LUNA PARK', 'LYONS SQUARE PLAYGROUND',
         'MORNINGSIDE PARK', 'PROSPECT PARK', 'METRO TRIANGLE',
         'HIGHLAND PARK', 'JAMES WELDON JOHNSON PLAYGROUND',
         'SPRINGFIELD PARK', 'TOMPKINSVILLE PARK', 'MOSHOLU PARKWAY',
         'ST. VARTAN PARK', 'RUFUS KING PARK', 'MOUNT HOPE PLAYGROUND',
         'NOSTRAND PLAYGROUND', 'MERRIAM PLAYGROUND',
         'JACOB H. SCHIFF PLAYGROUND', 'POLICE OFFICER EDWARD BYRNE PARK',
         'JOHN HANCOCK PLAYGROUND', 'COLUMBUS PARK AT MANHATTAN',
         'PULASKI PLAYGROUND', 'CANARSIE PARK', 'BRYANT PARK',
         "ST. MARY'S PARK BRONX", 'BOOKER T. WASHINGTON PLAYGROUND',
         'ELMHURST PARK', 'AQUEDUCT WALK', 'SHEEPSHEAD PLAYGROUND',
         'MCCAFFREY PLAYGROUND', 'TRAVERS PARK', 'BARTLETT PLAYGROUND',
         'MARIA HERNANDEZ PARK', 'HIGHBRIDGE PARK BRONX SIDE',
         'LAWRENCE VIRGILIO PLAYGROUND', 'AMERICAN TRIANGLE',
         'FLYNN PLAYGROUND', 'CROTONA PARK', 'MADISON SQUARE PARK',
         'COMMODORE BARRY PARK', 'BROWNSVILLE PLAYGROUND']
      metro_list = ['BROOKLYN BRIDGE-CITY HALL', '103 ST.-CORONA PLAZA',
         '59 STREET', '34 ST.-HERALD SQ.', '33 STREET', 'BEDFORD AVENUE',
         '157 STREET', 'UTICA AVE.-CROWN HEIGHTS', '125 STREET',
         '170 STREET', '50 STREET', 'PARSONS/ARCHER-JAMAICA CENTER',
         'FULTON STREET', '8 AVENUE', 'FORDHAM ROAD', 'BURNSIDE AVENUE',
         'BROADWAY/NASSAU', '207 ST.-INWOOD', 'UNION SQUARE',
         'ALLERTON AVENUE', '111 STREET', '110 STREET', 'ROCKAWAY AVENUE',
         '145 STREET', 'BOWLING GREEN', 'FRANKLIN AVENUE', 'GRANT AVENUE',
         'BROOK AVENUE', '81 ST.-MUSEUM OF NATURAL HISTO', '135 STREET',
         '86 STREET', '7TH AVENUE', 'BAY PARKWAY', 'LEXINGTON AVENUE',
         'W. 4 STREET', 'GRAND STREET', 'BROADWAY-EAST NEW YORK',
         '42 ST.-GRAND CENTRAL', 'SIMPSON STREET', 'HOYT-SCHERMERHORN',
         'BROADWAY/LAFAYETTE', '5 AVENUE', 'SUTTER AVENUE-RUTLAND ROAD',
         '1 AVENUE', 'COURT SQUARE', '181 STREET', '241 ST.-WAKEFIELD',
         '167 STREET', 'TREMONT AVENUE', 'WOODLAWN', 'MT. EDEN AVENUE',
         'PELHAM PKWY.', '42 ST.-TIMES SQUARE', 'CHAMBERS STREET',
         'JACKSON AVENUE', '69 STREET', 'EAST BROADWAY',
         'JAY STREET-BOROUGH HALL', 'CANAL STREET', '34 ST.-PENN STATION',
         '6 AVENUE', 'FLATBUSH AVE.-BROOKLYN COLLEGE',
         '3 AVENUE-149 STREET', '49 STREET', '72 STREET', '14 STREET',
         'MYRTLE AVENUE', 'EAST 174 STREET', 'CYPRESS AVENUE', '96 STREET',
         '80 STREET', '116 STREET', '42 ST.-PORT AUTHORITY BUS TERM',
         'QUEENSBORO PLAZA', 'DITMARS BLVD.-ASTORIA', 'EAST 180 STREET',
         'LORIMER STREET', '57 STREET', '205 ST.-NORWOOD',
         '82 ST.-JACKSON HEIGHTS', 'KINGSTON-THROOP AVENUES',
         'NEW UTRECHT AVENUE', 'GUN HILL ROAD', 'EAST 149 STREET',
         'GRAHAM AVENUE', 'NEVINS STREET', '14 ST.-UNION SQUARE',
         'PELHAM BAY PARK', 'MYRTLE-WILLOUGHBY AVENUES', 'KINGSBRIDGE ROAD',
         'HOWARD BEACH-JFK AIRPORT', 'BEDFORD-NOSTRAND AVENUES',
         '137 ST.-CITY COLLEGE', 'NOSTRAND AVENUE', 'FLUSHING AVENUE',
         '74 ST.-BROADWAY', 'BRONX PARK EAST', '66 ST.-LINCOLN CENTER',
         'UTICA AVENUE', 'ATLANTIC AVENUE', '47-50 STS./ROCKEFELLER CTR.',
         '34 STREET', '3 AVENUE-138 STREET', '20 AVENUE', 'LIBERTY AVENUE',
         '61 ST.-WOODSIDE', 'AVENUE "J"', 'PROSPECT AVENUE', '9 AVENUE',
         'STILLWELL AVENUE-CONEY ISLAND', '176 STREET', 'JAMAICA-VAN WYCK',
         'LAFAYETTE AVENUE', '71 AVE.-FOREST HILLS',
         '110 ST.-CATHEDRAL PKWY.', '23 STREET', 'WOODHAVEN BLVD.',
         '148 ST.-HARLEM', '59 ST.-COLUMBUS CIRCLE', 'GRAND ARMY PLAZA',
         'SHEEPSHEAD BAY', '182-183 STREETS', 'WHITEHALL ST.-SOUTH FERRY',
         '161 ST.-YANKEE STADIUM', 'BERGEN STREET', '36 STREET',
         '103 STREET', '168 ST.-WASHINGTON HTS.', 'DISTRICT 34 OFFICE',
         "EAST 143 ST.-ST. MARY'S STREET", '55 STREET', 'BOWERY',
         'CLINTON-WASHINGTON AVENUES', '79 STREET', '7 AVENUE',
         'SPRING STREET', 'HOUSTON STREET', '110 ST.-CENTRAL PARK NORTH',
         '191 STREET', '4 AVENUE-9 STREET', 'ROCKAWAY PKWY-CANARSIE',
         'BRIGHTON BEACH', '28 STREET', 'EUCLID AVENUE', 'NEW LOTS AVENUE',
         'CORTELYOU ROAD', 'UNION TURNPIKE-KEW GARDENS', 'QUEENS PLAZA',
         'ROOSEVELT AVE.-JACKSON HEIGHTS', 'AVENUE "P"', '155 STREET',
         '231 STREET', 'NEWKIRK AVENUE', 'PACIFIC STREET',
         'EAST 177 ST.-PARKCHESTER', 'CITY HALL', 'MARCY AVENUE',
         'BUSHWICK AVE.-ABERDEEN ST.', 'FAR ROCKAWAY-MOTT AVE.',
         'SHEPHERD AVENUE', 'HALSEY STREET', 'WHITLOCK AVENUE',
         'MAIN ST.-FLUSHING', '9TH STREET', 'CHURCH AVENUE',
         'FOREST AVENUE', '225 ST.-MARBLE HILL', 'CYPRESS HILLS',
         'LEXINGTON AVE.', 'BLEECKER STREET', 'CARROLL STREET',
         '116 ST.-COLUMBIA UNIVERSITY', '75 AVENUE', '51 STREET',
         'EASTERN PKWY-BROOKLYN MUSEUM', 'YORK STREET', 'BAY RIDGE AVENUE',
         'ELDER AVENUE', 'BROAD STREET', 'RALPH AVENUE',
         'WEST 8 STREET-NY AQUARIUM', 'SARATOGA AVENUE', 'AVENUE "M"',
         'WALL STREET', 'DEKALB AVENUE', '183 STREET', 'DELANCEY STREET',
         'KINGS HIGHWAY', 'KINGSTON AVENUE', 'WILLETS POINT-SHEA STADIUM',
         'SENECA AVENUE', 'JUNIUS STREET', '86TH STREET', 'AVENUE "X"',
         '2 AVENUE', '3 AVENUE', '175 STREET', 'BAY 50 STREET',
         '149 ST.-GRAND CONCOURSE', 'DISTRICT 30 OFFICE', '174-175 STREETS',
         'ST. LAWRENCE AVENUE', 'PRESIDENT STREET', 'MOSHOLU PKWY.',
         'SUTPHIN BLVD.-ARCHER AVE.', 'HUNTS POINT AVENUE',
         'BROADWAY-EASTERN PKWY', 'PARKSIDE AVENUE', 'HOYT STREET',
         'AVENUE "U"', '53 STREET', 'CRESCENT STREET', '207 STREET',
         '25 AVENUE', 'FREEMAN STREET', '90 ST.-ELMHURST AVE.',
         'BEACH 67 STREET', 'MYRTLE/WYCKOFF AVENUES',
         '68 ST.-HUNTER COLLEGE', 'BEDFORD PK. BLVD.-LEHMAN COLLE',
         'BEDFORD PK. BLVD.', 'SMITH-9 STREETS',
         '242 ST.-VAN CORTLANDT PARK', 'LIVONIA AVENUE', 'CLEVELAND STREET',
         '21 ST.-QUEENSBRIDGE', 'HIGH STREET', 'NORTHERN BLVD.',
         'CHAMBERS ST.-WORLD TRADE CENTE', '67 AVENUE', '46 STREET',
         '45 STREET', '163 ST.-AMSTERDAM AVE.',
         'EAST TREMONT AVE.-WEST FARMS S', 'DYRE AVE.-EASTCHESTER',
         'GRAND AVE.-NEWTON', 'CHRISTOPHER ST.-SHERIDAN SQ.',
         'VAN SICLEN AVENUE', 'BROADWAY', 'BROAD CHANNEL', 'BOROUGH HALL',
         'VAN WYCK BLVD.-BRIARWOOD', 'WILSON AVENUE', 'PRINCE STREET',
         '77 STREET', 'SUTPHIN BLVD.', '200 ST.-DYCKMAN ST.', '18 AVENUE',
         'FORT HAMILTON PKWY', 'CASTLE HILL AVENUE', 'NECK ROAD',
         '190 STREET', 'ALABAMA AVENUE', 'METROPOLITAN AVENUE',
         '219 STREET', 'MONTROSE AVENUE', 'DISTRICT 12 OFFICE',
         'HEWES STREET', 'WYCKOFF AVENUE', '15 ST.-PROSPECT PARK',
         '75 ST.-ELDERTS LANE', '238 STREET', 'KOSCIUSKO STREET',
         'PROSPECT PARK', 'JUNCTION BLVD.', 'STEINWAY ST.', 'RECTOR STREET',
         'STERLING STREET', 'GREENPOINT AVENUE', '8 ST.-NYU',
         'CHAUNCEY STREET', '95 STREET-BAY RIDGE', 'GATES AVENUE',
         'JEFFERSON STREET', '52 STREET', 'BEACH 60 STREET',
         'BEACH 90 STREET', 'ASTORIA BLVD.', 'AVENUE "I"',
         'VERNON BLVD.-JACKSON AVE.', 'EAST TREMONT AV.-WESTCHESTER S',
         'BEVERLEY ROAD', 'ESSEX STREET', 'NORWOOD AVENUE', 'CLARK STREET',
         'SUTTER AVENUE', 'INTERVALE AVENUE', 'NASSAU AVENUE',
         'UNION STREET', '21 STREET', 'ELMHURST AVE.', 'ROCKAWAY BLVD.',
         'DISTRICT 4 OFFICE', 'CLASSON AVENUE', 'KNICKERBOCKER AVENUE',
         'BEVERLY ROAD', '169 STREET', '18 STREET', '42 STREET',
         'COURT STREET', 'BUHRE AVENUE', 'SOUTH FERRY', 'BEACH 25 STREET',
         'ROCKAWAY PARK-BEACH 116 ST.', 'PARSONS BLVD.',
         'PENNSYLVANIA AVENUE', '85 ST.-FOREST PKWY.', 'ASTOR PLACE',
         'BEACH 36 STREET', 'BOTANIC GARDEN', 'ZEREGA AVENUE', '39 AVENUE',
         '138 ST.-GRAND CONCOURSE', '121 STREET', '238 ST.-NEREID AVE.',
         'OCEAN PKWY', 'BEACH 98 STREET', '225 STREET', 'LAWRENCE STREET',
         'MORGAN AVENUE', 'DITMAS AVENUE', 'DISTRICT 1 OFFICE',
         'LONGWOOD AVENUE', '233 STREET', 'BEACH 44 STREET',
         '23 STREET-ELY AVENUE', 'BEACH 105 STREET', 'PARK PLACE',
         '179 ST.-JAMAICA', '62 STREET', '45 ROAD-COURT HOUSE SQUARE',
         'NEPTUNE AVENUE', 'DISTRICT 23 OFFICE', '71 STREET',
         'DISTRICT 33 OFFICE', 'SOUNDVIEW AVENUE', '36 AVENUE',
         '63 DRIVE-REGO PARK', 'BURKE AVENUE',
         'AQUEDUCT-NORTH CONDUIT AVE.', '104 STREET', 'BAYCHESTER AVENUE',
         'LEFFERTS BLVD.', 'ROOSEVELT ISLAND', 'MIDDLETOWN ROAD',
         'AVENUE "N"', '215 STREET', '88 STREET', 'MORRIS PARK',
         'EAST 105 STREET', '40 STREET', '25 STREET', 'WINTHROP STREET',
         '65 STREET', 'CENTRAL AVENUE', '30 AVENUE', 'HUNTERS POINT AVENUE',
         'DISTRICT 3 OFFICE', 'AVENUE "H"', 'CORTLANDT STREET',
         'FRESH POND ROAD', '102 STREET', 'DISTRICT 11 OFFICE',
         'FRANKLIN STREET', 'AQUEDUCT-RACETRACK', 'OFF-SYSTEM']
      df['park_name'] = [element if element in park_list else 'NOT PARK' for element in df['park_name']]
      df['metro'] = [element if element in metro_list else 'NOT SUBWAY ' for element in df['metro']]
      self.data = df
      return self.data


    #8.
    def miss_lon_lat(self):
      '''
      Returns dataframe with missing coordinates replaced by
      median precinct coordinates (coordinates with most crime)
      '''
      df = self.data.copy()
      for precinct in df['precinct_number'].unique():
          #print(f'Modifying precinct{precinct}')
          geo = df[df['precinct_number'] == precinct][['latitude', 'longitude']]
          values = {'latitude': geo['latitude'].median(), 'longitude':geo['longitude'].median()} # get median lon and lat values as default for the precinct
          df[df['precinct_number']==precinct] = df[df['precinct_number']==precinct].fillna(value=values) # fill na with default values depending on precinct
      self.data = df
      return self.data

    def miss_borough(self):
      '''
      Returns dataframe with corrected boroughs for precinct wrong values
      '''

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
      self.data = df
      return self.data


    def miss_patrol_borough(self):
      '''
      Returns dataframe with corrected patrol boroughs per precinct
      '''

      df = self.data.copy()
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
      df['new_patrol'] = [
          'PATROL BORO BRONX' if precinct in bronx
          else 'PATROL BORO BKLYN SOUTH' if precinct in bklyn_south
          else 'PATROL BORO BKLYN NORTH' if precinct in bklyn_north
          else 'PATROL BORO MAN SOUTH' if precinct in man_south
          else 'PATROL BORO MAN NORTH' if precinct in man_north
          else 'PATROL BORO QUEENS SOUTH' if precinct in queens_south
          else 'PATROL BORO QUEENS NORTH' if precinct in queens_north
          else 'PATROL BORO STATEN ISLAND'
          for precinct in df['precinct_number']
      ]

      df.drop(columns='patrol_borough', inplace=True)
      df.rename(columns={'new_patrol':'patrol_borough'},inplace=True)
      self.data = df
      return self.data

    #9. Run round_int sur df['precinct_number']
    def round_int_precinct(self):
      """
      Returns df with correct precinct numbers (floats rounded up)
      """
      df = self.data.copy()
      df['precinct_number'] = [round(x) for x in df['precinct_number']]
      self.data = df
      return self.data

    # 10. Run complete_to_boolean sur df['crime_completed']
    # def crime_completed_to_boolean(self):
    #   """
    #       turns complete/incomplete into boolean value
    #   """
    #   df = self.data.copy()
    #   df['crime_completed'] = df['crime_completed'].replace({'COMPLETED': True, 'INCOMPLETE': False})
    #   self.data = df
    #   return self.data

    def clean_up_df(self):
      '''
      Returns clean df with reordered cols and droped complaint ID
      '''
      df = self.data.copy()

      df.drop(columns = 'complaint_id', inplace = True)

      list_column = ['period',
                     'latitude',
                     'longitude',
                     'offense_level',
                     'offense_type',
                     'crime_completed',
                     'premise_desc',
                     'premise',
                     'borough',
                     'patrol_borough',
                     'precinct_number',
                     'jurisdiction',
                     'suspect_age',
                     'suspect_race',
                     'suspect_sex',
                     'victim_age',
                     'victim_race',
                     'victim_sex',
                     'park_name',
                     'metro']
      # column reindex to be reviewed: doesn't work - not vital!
      df.reindex(columns = list_column, copy=False)

      self.data = df
      return self.data


    def filter_with_NYC_boundaries(self):
      """
      get rid of crimes out of NYC boundaries
      """
      df = self.data.copy()
      lat_interv = df['latitude'].between(40.49611539518921, 40.91553277600008)
      lon_interv = df['longitude'].between(-74.25559136315213,-73.70000906387347)

      df = df[lat_interv & lon_interv]
      self.data = df
      return self.data

    def six_hour_period(self):
      '''
      Adds column with period rounded to 6h timeframes
      '''
      df = self.data.copy()

      df['six_hour_date'] = df['period'].apply(lambda x: round_six_hours(x))

      self.data = df
      return self.data

    def total_clean(self):
      '''
      Combines all cleaning functions and returns clean dataframe
      '''
      print('dropping NaNs')
      self.drop_nan()
      print('Changing date column')
      self.to_date_format()
      print('Changing suspect column')
      self.miss_suspect()
      print('Changing victim column')
      self.miss_victim()
      print('Changing premise column')
      self.miss_premise()
      print('Changing park & metro column')
      self.miss_park_metro()
      print('Changing coordinates columns')
      self.miss_lon_lat()
      print('Changing borough column')
      self.miss_borough()
      print('Changing patrol column')
      self.miss_patrol_borough()
      print('Changing precinct column')
      self.round_int_precinct()
      print('Filtering NYC boundaries')
      self.filter_with_NYC_boundaries()
      print('Reording dataframe and final clean')
      self.clean_up_df()
      return self.data

    def precinct_75(self):
      '''
      Creates df with 75th precinct only and cuts lat long outliers
      '''
      df = self.data.copy()

      df_precinct_75 = df[df['precinct_number'] == 75]

      # max and min lat long for 75th precinct
      lat_min, lat_max, lon_min, lon_max = (40.6218192717505,
         40.6951504231971,
         -73.90404639808888,
         -73.83559344190869)

      max_lat = df['latitude'] <= lat_max # is smaller or equal to max lat boundary
      min_lat = df['latitude'] >= lat_min # is greater or equal to min lat boundary

      max_lon = df['longitude'] <= lon_max # is smaller or equal to max lon boundary
      min_lon = df['longitude'] >= lon_min # is greater or equal to min lon boundary

      df = df[ max_lat & min_lat & max_lon & min_lon] # side note: excludes 125 wrong lat long

      self.precinct = df

      return self.precinct


    def save_data(self):
      '''
      Saves clean dataframe to clean data pickle
      '''
      root_dir = os.path.dirname(os.path.dirname(__file__))
      pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
      precinct_75_pickle_path = os.path.join(root_dir, 'raw_data', 'clean-75-precinct.pickle')

      with open(pickle_path, 'wb') as f:
         pickle.dump(self.data, f)

      with open(precinct_75_pickle_path, 'wb') as f:
         pickle.dump(self.precinct, f)




if __name__ == '__main__':
  print('Initializing CleanData')
  clean_data = CleanData()
  print('Creating clean dataframe')
  clean_data.total_clean()
  print('Saving clean dataframe')
  clean_data.save_data()
