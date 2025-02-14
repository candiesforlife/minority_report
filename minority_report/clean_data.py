'''Return clean dataframe w/o NaN and with correct dtypes.'''

import os
import pandas as pd
import pickle
from datetime import datetime

from minority_report.data import NYPD
from minority_report.utils import round_six_hours


class CleanData:

  def __init__(self):

    self.data = NYPD().get_data() # Loads raw NYPD dataframe
    self.precinct = None


  def drop_cols(self):
    '''Drop unnecessary columns, e.g. complaint ID.'''
    df = self.data.copy()

    df.drop(columns = [
                    'CMPLNT_NUM',
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
                    'Lat_Lon'], inplace = True)

    self.data = df
    return self.data


  def rename_cols(self):
    '''Rename dataset columns to snake_case format.'''
    df = self.data.copy()

    # Dictionary of new names for each column
    columns_dic = {
        'CMPLNT_FR_DT': 'date',
        'CMPLNT_FR_TM': 'time',
        'LAW_CAT_CD': 'offense_level',
        'OFNS_DESC': 'offense_type',
        'CRM_ATPT_CPTD_CD': 'crime_completed',
        'LOC_OF_OCCUR_DESC': 'premise_desc',
        'PREM_TYP_DESC': 'premise',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'BORO_NM': 'borough',
        'PATROL_BORO': 'patrol_borough',
        'ADDR_PCT_CD': 'precinct_number',
        'JURIS_DESC': 'jurisdiction',
        'SUSP_AGE_GROUP': 'suspect_age',
        'SUSP_RACE': 'suspect_race',
        'SUSP_SEX': 'suspect_sex',
        'VIC_AGE_GROUP': 'victim_age',
        'VIC_RACE': 'victim_race',
        'VIC_SEX': 'victim_sex',
        'PARKS_NM': 'park_name',
        'STATION_NAME': 'metro'}

    df.rename(columns = columns_dic, inplace = True)

    self.data = df
    return self.data


  def drop_nan(self):
    '''Return a dataframe without NaN.'''
    df = self.data.copy()

    # Drops missing values in 'precinct'
    not_unknown = df['precinct_number'] != -99.0
    not_nan = df['precinct_number'] == df['precinct_number']
    df = df[not_unknown & not_nan]

    # Drops missing values in 'time' and 'date'
    df = df[df['time'] == df['time']]
    df = df[df['date'] == df['date']]

    # Drops missing values in 'offense_type' and 'crime_completed'
    df = df[df['offense_type'] == df['offense_type']]
    df = df[df['crime_completed'] == df['crime_completed']]

    self.data = df
    return self.data


  def to_date_format(self):
    ''' Keep only complete years and create datetime column.

    Create new column 'period' out of columns 'date' and 'time'.
    Drop columns 'date' and 'time'.
    Filter dataframe to show only complaints dated >= 2007.
    '''
    df = self.data.copy()

    # New period column out of concatenated 'date' and 'time'
    df['period'] = df['date'] + ' ' + df['time']

    # Converts 'period' to datetime format
    df['period'] = df['period'].apply(lambda x: \
                          datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))

    # Trims 'period' to hour
    df['period'] = df['period'].apply(lambda x: \
                          x.replace(minute = 0, second = 0))

    # Filters dataframe to exclude crimes older than 2007
    df = df[df['period'] > datetime(2006, 12, 31, 23, 59, 0)]

    # Drops columns 'date' and 'time'
    df.drop(columns = ['date', 'time'], inplace = True)

    self.data = df
    return self.data


  def miss_suspect(self):
    '''Replace missing suspect values with UNKNOWN.'''
    df = self.data.copy()

    # Values to keep
    age_liste = ['<18', '45-64', '18-24', '25-44', '65+']
    race_liste = ['BLACK', 'WHITE', 'WHITE HISPANIC', 'BLACK HISPANIC',
       'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE']
    sex_liste = ['M', 'F', 'U']

    # Replaces all other values with 'UNKNOWN'
    df['suspect_age'] = [element if element in age_liste else 'UNKNOWN' for element in df['suspect_age']]
    df['suspect_race'] = [element if element in race_liste else 'UNKNOWN' for element in df['suspect_race']]
    df['suspect_sex'] = [element if element in sex_liste else 'UNKNOWN' for element in df['suspect_sex']]

    self.data = df
    return self.data


  def miss_victim(self):
    '''Replace missing victim values with UNKNOWN.'''
    df = self.data.copy()

    # Values to keep
    age_liste = ['<18', '45-64', '18-24', '25-44', '65+']
    race_liste = ['BLACK', 'WHITE', 'UNKNOWN', 'WHITE HISPANIC', 'BLACK HISPANIC',
       'ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE']
    sex_liste = ['M', 'F', 'D', 'E']

    # Replaces all other values with 'UNKNOWN'
    df['victim_age'] = [element if element in age_liste else 'UNKNOWN' for element in df['victim_age']]
    df['victim_race'] = [element if element in race_liste else 'UNKNOWN' for element in df['victim_race']]
    df['victim_sex'] = [element if element in sex_liste else 'UNKNOWN' for element in df['victim_sex']]

    self.data = df
    return self.data


  def miss_premise(self):
    '''Replace missing premise values with UNKNOWN.'''
    df = self.data.copy()

    # Values to keep
    premise = [
      'RESIDENCE - PUBLIC HOUSING', 'RESIDENCE-HOUSE', 'PUBLIC SCHOOL',
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

    # Replaces all other values with 'UNKNOWN'
    df['premise'] = [element if element in premise else 'UNKNOWN' for element in df['premise']]
    df['premise_desc'] = [element if element in premise_desc else 'UNKNOWN' for element in df['premise_desc']]
    self.data = df
    return self.data


  def miss_park_metro(self):
    '''Replace missing park and subway values with NOT PARK or NOT SUBWAY'''
    df = self.data.copy()

    park_list = [
      'JACKIE ROBINSON PARK MANHATTAN', 'TOMPKINS SQUARE PARK',
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

    metro_list = [
      'BROOKLYN BRIDGE-CITY HALL', '103 ST.-CORONA PLAZA',
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


  def miss_lon_lat(self):
    '''Replace missing latitude and longitude with median precinct coordinates.

    Median precinct coordinates are the coordinates with the highest
    crime occurence per precinct.
    '''

    df = self.data.copy()

    for precinct in df['precinct_number'].unique():
        geo = df[df['precinct_number'] == precinct][['latitude', 'longitude']]
        # Median lat & lon values per precinct
        values = {'latitude': geo['latitude'].median(), 'longitude':geo['longitude'].median()}
        # Replaces NaN with default precinct values
        df[df['precinct_number'] == precinct] = df[df['precinct_number'] == precinct].fillna(value=values)

    self.data = df
    return self.data


  def miss_borough(self):
    '''Replace wrong borough values.

    Assign correct borough for each precinct number.
    '''
    df = self.data.copy()

    # Fills in correct borough for given precinct numbers
    df['new_borough'] = [
        'MANHATTAN' if precinct <=34
        else 'BRONX' if precinct <=54
        else 'BROOKLYN' if precinct <= 94
        else 'QUEENS' if precinct <= 115
        else 'STATEN ISLAND'
        for precinct in df['precinct_number']]

    # Replaces original borough column
    df.drop(columns='borough', inplace=True)
    df.rename(columns={'new_borough': 'borough'}, inplace=True)

    self.data = df
    return self.data


  def miss_patrol_borough(self):
    '''Replace wrong patrol borough values.

    Assign each researched patrol borough number to associated NYC borough.
    '''
    df = self.data.copy()

    # Correct patrol borough
    bronx = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52]
    bklyn_south = [60, 61, 62, 63, 66, 67, 68, 69, 70, 71, 72, 76, 78]
    bklyn_north = [73, 75, 77, 79, 81, 83, 84, 88, 90, 94]
    man_south = [1, 5, 6, 7, 9, 10, 13, 14, 17, 18]
    man_north = [19, 20, 22, 23, 24, 25, 26, 28, 30, 32, 33, 34]
    queens_south = [100, 101, 102, 103, 105, 106, 107, 113]
    queens_north = [104, 108, 109, 110, 111, 112, 114, 115]
    staten = [120, 121, 122, 123]

    # Replaces with correct patrol borough
    df['new_patrol'] = [
        'PATROL BORO BRONX' if precinct in bronx
        else 'PATROL BORO BKLYN SOUTH' if precinct in bklyn_south
        else 'PATROL BORO BKLYN NORTH' if precinct in bklyn_north
        else 'PATROL BORO MAN SOUTH' if precinct in man_south
        else 'PATROL BORO MAN NORTH' if precinct in man_north
        else 'PATROL BORO QUEENS SOUTH' if precinct in queens_south
        else 'PATROL BORO QUEENS NORTH' if precinct in queens_north
        else 'PATROL BORO STATEN ISLAND'
        for precinct in df['precinct_number']]

    # Replaces original patrol borough column
    df.drop(columns='patrol_borough', inplace=True)
    df.rename(columns={'new_patrol': 'patrol_borough'}, inplace=True)

    self.data = df
    return self.data


  def round_int_precinct(self):
    '''Round inappropriate float precinct numbers.'''
    df = self.data.copy()

    df['precinct_number'] = [round(x) for x in df['precinct_number']]

    self.data = df
    return self.data


  def clean_up_df(self):
    '''Reorganise columns for dataframe legibility.'''
    df = self.data.copy()

    list_column = [
                  'period',
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

    # Reindex to be reviewed: doesn't work - not vital!
    df.reindex(columns = list_column, copy=False)

    self.data = df
    return self.data


  def filter_with_NYC_boundaries(self):
    '''Exclude complaints from outside of NYC coordinate boundaries.'''
    df = self.data.copy()

    lat_interv = df['latitude'].between(40.49611539518921, 40.91553277600008)
    lon_interv = df['longitude'].between(-74.25559136315213, -73.70000906387347)

    df = df[lat_interv & lon_interv]

    self.data = df
    return self.data


  def six_hour_period(self):
    '''Add column with period rounded to 6h timeframes'''

    df = self.data.copy()

    # Function round_six_hours from utils.py
    df['six_hour_date'] = df['period'].apply(lambda x: round_six_hours(x))

    self.data = df
    return self.data


  def precinct_75(self):
    '''Create dataframe containing only 75th precinct complaints.

    Filter to contain only precinct number 75.
    Exclude latitude and longitude outliers to fit 75th boundaries.
    '''
    df = self.data.copy()

    df = df[df['precinct_number'] == 75]

    # Lat/Long extremities for 75th precinct
    lat_min, lat_max, lon_min, lon_max = (
                                        40.6218192717505,
                                        40.6951504231971,
                                        -73.90404639808888,
                                        -73.83559344190869)

    max_lat = df['latitude'] <= lat_max
    min_lat = df['latitude'] >= lat_min

    max_lon = df['longitude'] <= lon_max
    min_lon = df['longitude'] >= lon_min

    # Excludes 125 wrong lat long values
    df = df[ max_lat & min_lat & max_lon & min_lon]

    self.precinct = df
    return self.precinct


  def save_data(self):
    '''Save clean dataframe to clean data pickle.'''

    root_dir = os.path.dirname(os.path.dirname(__file__))

    pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
    precinct_75_pickle_path = os.path.join(root_dir, 'raw_data', 'clean-75-precinct.pickle')

    # Saves entire NYC dataframe to pickle
    with open(pickle_path, 'wb') as f:
       pickle.dump(self.data, f)

    # Saves 75th precinct only (smaller scope)
    with open(precinct_75_pickle_path, 'wb') as f:
       pickle.dump(self.precinct, f)


if __name__ == '__main__':
  '''Take original NYPD data and return clean dataframe.'''
  print('Initializing CleanData')
  clean_data = CleanData()
  print('Creating clean dataframe (17 Steps):')
  print('1. Dropping Columns')
  clean_data.drop_cols()
  print('2. Renaming Columns')
  clean_data.rename_cols()
  print('3. Dropping NaNs')
  clean_data.drop_nan()
  print('4. Changing date column')
  clean_data.to_date_format()
  print('5. Changing suspect column')
  clean_data.miss_suspect()
  print('6. Changing victim column')
  clean_data.miss_victim()
  print('7. Changing premise column')
  clean_data.miss_premise()
  print('8. Changing park & metro column')
  clean_data.miss_park_metro()
  print('9. Changing coordinates columns')
  clean_data.miss_lon_lat()
  print('10. Changing borough column')
  clean_data.miss_borough()
  print('11. Changing patrol column')
  clean_data.miss_patrol_borough()
  print('12. Changing precinct column')
  clean_data.round_int_precinct()
  print('13. Filtering NYC boundaries')
  clean_data.filter_with_NYC_boundaries()
  print('14. Adding six hour period column')
  clean_data.six_hour_period()
  print('15. Creating 75th precinct df w/o outliers')
  clean_data.precinct_75()
  print('16. Reording dataframe')
  clean_data.clean_up_df()
  print('17. Saving clean dataframe')
  clean_data.save_data()
  print('New pickles ready to use! :)')
