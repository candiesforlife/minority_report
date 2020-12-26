'''
Passes clean_data into matrix, gaussian filter and stacking
Returns X & y train and test pickles
'''

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.ndimage import gaussian_filter

from minority_report.clean_data import CleanData
from minority_report.scaling import Scaling
from minority_report.utils import from_meters_to_steps


# Have clean_split.py to split into two df pickles
# Check where to call Split class from!
# Pass train and test in matrix.py
# (Matrix, Gaussian, Stacking)
# Get oversations


class Matrix:

    def __init__(self):
        self.data = None

        self.img3D_conv_train = None
        self.img3D_non_conv_train = None
        self.img3D_conv_test = None
        self.img3D_non_conv_test = None

        self.lat_meters = None
        self.lon_meters = None

        self.train_df = None
        self.test_df = None

        self.sigma_x = None
        self.sigma_y = None
        self.sigma_z = None

        self.raw_z = 12

        self.X_test = None
        self.y_test = None

        self.X_train = None
        self.y_train = None


    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean-75-precinct.pickle')

        # gcp_pickle_path = 'clean-75-precinct.pickle'

        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data

    # def train_test_df(self):
    #     '''
    #     Splitting train and test data into two dataframes
    #     '''
    #     df = self.data.copy()

    #     # creating train df
    #     inf_train = df['period'] >= datetime(2007, 1, 1, 0, 0, 0)

    #     sup_train = df['period'] < datetime(2016, 1, 1, 0, 0, 0)

    #     self.train_df = df[inf_train & sup_train]

    #     # creating test df
    #     inf_test = df['period'] >= datetime(2016, 1, 1, 0, 0, 0)

    #     sup_test = df['period'] <= datetime(2019, 10, 28, 0, 0, 0)

    #     self.test_df = df[inf_test & sup_test]

    #     return self.train_df, self.test_df

    # def from_meters_to_steps(self):
    #     """
    #     gives the latitude and longitude step to use for the grid buckets
    #     lat_meters, lon_meters = lat/lon step
    #     """
    #     #Position, decimal degrees
    #     lat = 40
    #     lon = -73

    #     #Earth’s radius, sphere
    #     R = 6378137

    #     #offsets in meters
    #     dn = self.lat_meters
    #     de = self.lon_meters

    #     #Coordinate offsets in radians
    #     dLat = dn/R
    #     dLon = de/(R*np.cos(np.pi*lat/180))

    #     #OffsetPosition, decimal degrees
    #     latO = dLat * 180/np.pi
    #     lonO = dLon * 180/np.pi

    #     return latO, lonO

    def getting_sigma_values(self, raw_x, raw_y, raw_z):
        '''
          Returns sigma values for the three dimensions, useful later for gaussian filtering.
        '''
        self.sigma_x = (raw_x / self.lat_meters) / 2
        self.sigma_y = (raw_y / self.lon_meters) / 2
        self.sigma_z = raw_z / 2

        return self.sigma_x, self.sigma_y, self.sigma_z




    # Train Matrix


    def from_coord_to_matrix_train(self, lat_meters, lon_meters):
        """
        outputs the 3D matrix of all coordinates for a given bucket height and width in meters
        """
        self.lat_meters = lat_meters
        self.lon_meters = lon_meters

        df = self.train_df.copy()
        # add 'time_index' column to df
        #ind = {time:index for index,time in enumerate(np.sort(df['period'].unique()))}
        #df['time_index'] = df['period'].map(ind)
        ind = {time: index for index, time in enumerate(np.sort(df['six_hour_date'].unique()))}
        df['time_index'] = df['six_hour_date'].map(ind)
        #print(df.groupby('time_index').count())
        #initiate matrix
        #40.49611539518921, 40.91553277600008, -74.25559136315213,-73.70000906387347) : NYC boundaries
        #([40.56952999448672, 40.73912795313436],[-74.04189660705046, -73.83355923946421]) : brooklyn boundaries
        #[40.6218192717505, 40.6951504231971],[-73.90404639808888, -73.83559344190869]) :precinct 75 boundaries
        grid_offset = np.array([ -df['latitude'].max() , df['longitude'].min(), 0 ]) # Where do you start
        #from meters to lat/lon step

        # lat_spacing, lon_spacing = self.from_meters_to_steps()
        # Added from_meters_to_steps in utils instead: CLEAN UP FOR TEST AS WELL IF IT WORKS
        lat_spacing, lon_spacing = from_meters_to_steps(lat_meters, lon_meters)

        grid_spacing = np.array([lat_spacing , lon_spacing, 1 ]) # What's the space you consider (euclidian here)
        #get points coordinates
        coords = np.array([( -lat, lon,t_ind) for lat, lon,t_ind \
                       in zip(df['latitude'],df['longitude'],df['time_index'])])

        # Convert point to index
        indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
        X = indexes[:,0]
        Y = indexes[:,1]
        Z = indexes[:,2]

        #virgin matrix

        # 75th precinct distances:
        lat_min, lat_max, lon_max, lon_min = (40.6218192717505,
                                              40.6951504231971,
                                              -73.90404639808888,
                                              -73.83559344190869)

        lat_diff = lat_max - lat_min # distance in lat that makes up width of precinct 75
        lon_diff = lon_min - lon_max # distance in lon that makes up width of precinct 75

        # dim 1: distance of precinct in lat / lat_spacing
        a = np.zeros((np.round(lat_diff / lat_spacing).astype('int') + 1,
                     np.round(lon_diff / lon_spacing).astype('int') + 1,
                     Z.max() + 1))

        # old version: a = np.zeros((X.max()+1, Y.max()+1, Z.max()+1))

        a[X, Y, Z] = 1

        self.lat_size = a.shape[1]
        self.lon_size = a.shape[2]
        self.img3D_non_conv_train = a

        return self.img3D_non_conv_train


    def gaussian_filtering_train(self):
        '''
          Returns img3D convoluted
        '''

        self.img3D_conv_train = gaussian_filter(self.img3D_non_conv_train,
            sigma = (self.sigma_x, self.sigma_y, self.sigma_z))

        return self.img3D_conv_train

    ########
    # Input: Stacking et all.

    def stacking_train(self, window, lat_step, lon_step, time_step):
        '''
            Returns stacked crimes.
        '''
        grid_offset = np.array([0,0,0]) # Where do you start
        #new steps from precise grid
        grid_spacing = np.array([lat_step , lon_step, time_step])
        #get points coordinates
        coords = np.argwhere(window)
        flat = window.flatten()
        values = flat[flat !=0]
        # Convert point to index
        indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
        X = indexes[:,0]
        Y = indexes[:,1]
        Z = indexes[:,2]
        #virgin matrix: 256 is arbitrary size that works in model
        stacked_crimes = np.zeros((192, 132, Z.max() + 2))

        for i in range(len(indexes)):

            if stacked_crimes[X[i], Y[i], Z[i]] == 0:
                stacked_crimes[X[i], Y[i], Z[i]] = values[i]
            else:
                stacked_crimes[X[i], Y[i], Z[i]] += values[i]

        return stacked_crimes

    def get_observation_target_train(self,
                           obs_timeframe,obs_lat,obs_lon, obs_time,
                           target_timeframe,  tar_lat,tar_lon, tar_time):
        '''
        output an observation of x_length consecutive images and the y_length next images as the target
        obs_step, obs_timeframe, target_step, target_timeframe : unit = hours
        '''

        # sample length to absorb impact of gaussian time sigma
        sample_length = obs_timeframe + (self.raw_z + 1) + target_timeframe

        # finds starting position
        position = np.random.randint(0, self.img3D_conv_train.shape[2] - sample_length)

        # samples in train and test dfs
        subsample = self.img3D_conv_train[:, :, position : position + sample_length]

        # divide the subsample in X and y
        observations = subsample[:, :, : obs_timeframe]

        targets = subsample[:, :, - target_timeframe : ]

        # stacked images
        observation = self.stacking_train(observations, obs_lat, obs_lon, obs_time)

        target = self.stacking_train(targets, tar_lat, tar_lon, tar_time)

        return observation, target

    def get_X_y_train(self, nb_observations_train, obs_tf,obs_lat,obs_lon, obs_time,
                    tar_tf, tar_lat,tar_lon, tar_time):
        '''
        outputs n observations and their associated targets
        '''
        X = []
        y = []

        for n in range(nb_observations_train):
            print(f'Creating observation {n} out of {nb_observations_train}')
            X_subsample, y_subsample = self.get_observation_target_train(obs_tf,
                                        obs_lat,obs_lon, obs_time,
                                        tar_tf,  tar_lat,tar_lon, tar_time)
            X.append(X_subsample)
            y.append(y_subsample)

        X = np.array(X)
        y = np.array(y)

        self.X_train = X
        self.y_train = y

        return self.X_train, self.y_train



    # Test Matrix


    def from_coord_to_matrix_test(self, lat_meters, lon_meters):
        """
        outputs the 3D matrix of all coordinates for a given bucket height and width in meters
        """
        self.lat_meters = lat_meters
        self.lon_meters = lon_meters

        df = self.test_df.copy()
        # add 'time_index' column to df
        #ind = {time:index for index,time in enumerate(np.sort(df['period'].unique()))}
        #df['time_index'] = df['period'].map(ind)
        ind = {time: index for index, time in enumerate(np.sort(df['six_hour_date'].unique()))}
        df['time_index'] = df['six_hour_date'].map(ind)
        #print(df.groupby('time_index').count())
        #initiate matrix
        #40.49611539518921, 40.91553277600008, -74.25559136315213,-73.70000906387347) : NYC boundaries
        #([40.56952999448672, 40.73912795313436],[-74.04189660705046, -73.83355923946421]) : brooklyn boundaries
        #[40.6218192717505, 40.6951504231971],[-73.90404639808888, -73.83559344190869]) :precinct 75 boundaries
        grid_offset = np.array([ -df['latitude'].max() , df['longitude'].min(), 0 ]) # Where do you start
        #from meters to lat/lon step
        lat_spacing, lon_spacing = self.from_meters_to_steps()
        grid_spacing = np.array([lat_spacing , lon_spacing, 1 ]) # What's the space you consider (euclidian here)
        #get points coordinates
        coords = np.array([( -lat, lon,t_ind) for lat, lon,t_ind \
                       in zip(df['latitude'],df['longitude'],df['time_index'])])

        # Convert point to index
        indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
        X = indexes[:,0]
        Y = indexes[:,1]
        Z = indexes[:,2]

        #virgin matrix

        # 75th precinct distances:
        lat_min, lat_max, lon_max, lon_min = (40.6218192717505,
                                              40.6951504231971,
                                              -73.90404639808888,
                                              -73.83559344190869)

        lat_diff = lat_max - lat_min # distance in lat that makes up width of precinct 75
        lon_diff = lon_min - lon_max # distance in lon that makes up width of precinct 75

        # dim 1: distance of precinct in lat / lat_spacing
        a = np.zeros((np.round(lat_diff / lat_spacing).astype('int') + 1,
                     np.round(lon_diff / lon_spacing).astype('int') + 1,
                     Z.max() + 1))

        # old version: a = np.zeros((X.max()+1, Y.max()+1, Z.max()+1))

        a[X, Y, Z] = 1

        self.lat_size = a.shape[1]
        self.lon_size = a.shape[2]
        self.img3D_non_conv_test = a

        return self.img3D_non_conv_test


    def gaussian_filtering_test(self):
        '''
          Returns img3D convoluted
        '''

        self.img3D_conv_test = gaussian_filter(self.img3D_non_conv_test,
            sigma = (self.sigma_x, self.sigma_y, self.sigma_z))

        return self.img3D_conv_test

    ###############
    # Input Replacement: Stacking et all.

    def stacking_test(self, window, lat_step, lon_step, time_step):
        '''
            Returns stacked crimes.
        '''
        # starting point
        grid_offset = np.array([0,0,0])

        #new steps from precise grid
        grid_spacing = np.array([lat_step , lon_step, time_step])

        #get points coordinates
        coords = np.argwhere(window)
        flat = window.flatten()
        values = flat[flat !=0]

        # Convert point to index
        indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
        X = indexes[:,0]
        Y = indexes[:,1]
        Z = indexes[:,2]

        #virgin matrix: 256 absolute size to be stacked to work in model!
        stacked_crimes = np.zeros((192, 132, Z.max() + 2))

        for i in range(len(indexes)):

            if stacked_crimes[X[i], Y[i], Z[i]] == 0:
                stacked_crimes[X[i], Y[i], Z[i]] = values[i]
            else:
                stacked_crimes[X[i], Y[i], Z[i]] += values[i]

        return stacked_crimes

    def get_observation_target_test(self,
                           obs_timeframe,obs_lat,obs_lon, obs_time,
                           target_timeframe,  tar_lat,tar_lon, tar_time):
        '''
        output an observation of x_length consecutive images and the y_length next images as the target
        obs_step, obs_timeframe, target_step, target_timeframe : unit = hours
        '''

        # sample length to absorb impact of gaussian time sigma
        sample_length = obs_timeframe + (self.raw_z + 1) + target_timeframe

        # finds starting position
        position = np.random.randint(0, self.img3D_conv_test.shape[2] - sample_length)

        # samples in train and test dfs
        subsample = self.img3D_conv_test[:, :, position : position + sample_length]

        # divide the subsample in X and y
        observations = subsample[:, :, : obs_timeframe]

        targets = subsample[:, :, - target_timeframe : ]

        # stacked images
        observation = self.stacking_test(observations, obs_lat, obs_lon, obs_time)

        target = self.stacking_test(targets,  tar_lat, tar_lon, tar_time )

        return observation, target

    def get_X_y_test(self, nb_observations_test, obs_tf,obs_lat,obs_lon, obs_time,
                    tar_tf, tar_lat,tar_lon, tar_time):
        '''
        outputs n observations and their associated targets
        '''

        X = []
        y = []

        for n in range(nb_observations_test):
            print(f'Creating observation {n} out of {nb_observations_test}')
            X_subsample, y_subsample = self.get_observation_target_test(obs_tf,
                                        obs_lat,obs_lon, obs_time,
                                        tar_tf,  tar_lat,tar_lon, tar_time)
            X.append(X_subsample)
            y.append(y_subsample)

        X = np.array(X)
        y = np.array(y)

        self.X_test = X
        self.y_test = y

        return self.X_test, self.y_test


    def save_data(self):
        '''
        Saves clean dataframe to clean data pickle
        '''
        root_dir = os.path.dirname(os.path.dirname(__file__))
        X_train_pickle_path = os.path.join(root_dir, 'raw_data', 'X_train_large.pickle')
        y_train_pickle_path = os.path.join(root_dir, 'raw_data', 'y_train_large.pickle')

        X_test_pickle_path = os.path.join(root_dir, 'raw_data', 'X_test_large.pickle')
        y_test_pickle_path = os.path.join(root_dir, 'raw_data', 'y_test_large.pickle')

        with open(X_train_pickle_path, 'wb') as f:
            pickle.dump(self.X_train, f)

        with open(y_train_pickle_path, 'wb') as f:
            pickle.dump(self.y_train, f)

        with open(X_test_pickle_path, 'wb') as f:
            pickle.dump(self.X_test, f)

        with open(y_test_pickle_path, 'wb') as f:
            pickle.dump(self.y_test, f)

    # Used for both Train and Test Dataframes

    # def save_data(self):
    #       '''
    #       Saves clean dataframe to clean data pickle
    #       '''
    #       root_dir = os.path.dirname(os.path.dirname(__file__))
    #       train_pickle_path = os.path.join(root_dir, 'raw_data', 'img_train.pickle')

    #       with open(train_pickle_path, 'wb') as f:
    #          pickle.dump(self.img3D_conv_train, f)

    #       test_pickle_path = os.path.join(root_dir, 'raw_data', 'img_test.pickle')

    #       with open(test_pickle_path, 'wb') as f:
    #          pickle.dump(self.img3D_conv_test, f)


    def preprocessing_X_y(self,nb_observations_train, nb_observations_test, lat_meters, lon_meters, raw_x, raw_y, raw_z,#nb_observations,
     obs_tf, obs_lat, obs_lon, obs_time,
     tar_tf, tar_lat,tar_lon, tar_time):
        '''
        Takes crime row in original dataframe,
        passes through grid, gaussian filter and stacking.
        Returns X and y (both train and test).'''

        print("4. Loading data")
        self.load_data()

        print('5. Splitting into Train and Test Dataframe')
        self.train_test_df()

        print('6a. From coords to matrix: train')
        self.from_coord_to_matrix_train(lat_meters, lon_meters)

        print('6b. From coords to matrix: test')
        self.from_coord_to_matrix_test(lat_meters, lon_meters)

        print('7. Getting sigma values for Gaussian filter')
        self.getting_sigma_values(raw_x, raw_y, raw_z)

        print('8a. Gaussian filtering: Train')
        self.gaussian_filtering_train()

        print('8b. Gaussian filtering: Test')
        self.gaussian_filtering_test()

        print('9a. Getting X, y Train') #nb_observations
        self.get_X_y_train(nb_observations_train, obs_tf,obs_lat,obs_lon, obs_time,
                    tar_tf, tar_lat,tar_lon, tar_time)

        print('9b. Getting X, y Test') #nb_observations
        self.get_X_y_test(nb_observations_test, obs_tf,obs_lat,obs_lon, obs_time,
                    tar_tf, tar_lat,tar_lon, tar_time)

        return self.X_train, self.y_train, self.X_test, self.y_test

