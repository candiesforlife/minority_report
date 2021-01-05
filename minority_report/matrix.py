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
from minority_report.utils import from_meters_to_steps, stacking
# from minority_report.utils import stacking


# Pass train and test in matrix.py
# (Matrix, Gaussian, Stacking)
# Get oversations


class Matrix:

    def __init__(self, lat_meters, lon_meters, raw_x, raw_y, raw_z,\
        nb_observations_train, nb_observations_test,\
        obs_tf, obs_lat, obs_lon, obs_time,\
        tar_tf, tar_lat, tar_lon, tar_time):
        '''Initiate Matrix class with all input parameters.

        lat_meters, lon_meters: meters that define primary grid spacing
        raw_x, raw_y = lat, long distance in meters that gaussian_filter uses
        raw_z = time dimension for gaussian_filter (eg. 12 = 12 * 6h (72h))
        nb_observations_train = number of observations from train_df
        nb_observations_test = number of observations from test_df
        obs_tf = timeframe for input X (e.g. we use 2 weeks to predict...)
        tar_tf = timeframe for output y (e.g. we use ... to predict two days)
        obs_lon, tar_lon = spacial stacking lon for X and y
        obs_lat, tar_lat = spacial stacking lat for X and y
        obs_time, tar_time = temporal stacking (number of 6h timeframes to use)
        '''
        self.train_df = None
        self.test_df = None

        self.img3D_non_conv_train = None
        self.img3D_conv_train = None
        self.img3D_non_conv_test = None
        self.img3D_conv_test = None

        self.lat_meters = lat_meters
        self.lon_meters = lon_meters

        self.raw_x = raw_x
        self.raw_y = raw_y
        self.raw_z = raw_z
        self.sigma_x = None
        self.sigma_y = None
        self.sigma_z = None

        self.nb_observations_train = nb_observations_train
        self.nb_observations_test = nb_observations_test

        self.obs_tf = obs_tf
        self.obs_lat = obs_lat
        self.obs_lon = obs_lon
        self.obs_time = obs_time

        self.tar_tf = tar_tf
        self.tar_lat = tar_lat
        self.tar_lon = tar_lon
        self.tar_time = tar_time

        self.X_test = None
        self.y_test = None

        self.X_train = None
        self.y_train = None


    def load_data(self):
        '''Load train and test dataframes.'''
        root_dir = os.path.dirname(os.path.dirname(__file__))

        train_path = os.path.join(root_dir, 'raw_data', 'train_df.pickle')
        test_path = os.path.join(root_dir, 'raw_data', 'test_df.pickle')

        with open(train_path, 'rb') as f:
            train_df = pickle.load(f)

        with open(test_path, 'rb') as g:
            test_df = pickle.load(g)

        self.train_df = train_df
        self.test_df = test_df

        return self.train_df, self.test_df


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

    def getting_sigma_values(self):
        '''Return three sigma values for gaussian filter.

        Each sigma value represents one of the three dimensions.
        raw_x and raw_y represent the number of meters a crime spreads out over.
        raw_z represents the number of 6h timeslots a crime revetebrates across.
        '''
        self.sigma_x = (self.raw_x / self.lat_meters) / 2
        self.sigma_y = (self.raw_y / self.lon_meters) / 2
        self.sigma_z = self.raw_z / 2

        return self.sigma_x, self.sigma_y, self.sigma_z


    # Train Matrix


    def from_coord_to_matrix_train(self):
        '''Return 3D matrix containing points of crime.

        Each coordinate is assigned to a bucket of size lat_meters and lon_meters.
        '''
        df = self.train_df.copy()

        # Adds 'time_index' column to dataframe
        ind = {time: index for index, time in enumerate(np.sort(df['six_hour_date'].unique()))}
        df['time_index'] = df['six_hour_date'].map(ind)

        # Matrix starting point
        grid_offset = np.array([-df['latitude'].max(), df['longitude'].min(), 0])

        # Converts bucket size (meters) to lat & lon spacing
        lat_spacing, lon_spacing = from_meters_to_steps(self.lat_meters, self.lon_meters)

        # Euclidian spacing
        grid_spacing = np.array([lat_spacing , lon_spacing, 1 ])

        # Gets point coordinates
        coords = np.array([(-lat, lon, t_ind) for lat, lon, t_ind \
                       in zip(df['latitude'], df['longitude'], df['time_index'])])

        # Convert point to index
        indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
        X = indexes[:,0]
        Y = indexes[:,1]
        Z = indexes[:,2]

        # 75th precinct maximum & minimum points
        lat_min, lat_max, lon_max, lon_min = (40.6218192717505,
                                              40.6951504231971,
                                              -73.90404639808888,
                                              -73.83559344190869)

        lat_diff = lat_max - lat_min # Distance in lat that makes up width of precinct 75
        lon_diff = lon_min - lon_max # Distance in lon that makes up width of precinct 75

        # Dim 1: distance of precinct in lat / lat_spacing
        a = np.zeros((np.round(lat_diff / lat_spacing).astype('int') + 1,
                     np.round(lon_diff / lon_spacing).astype('int') + 1,
                     Z.max() + 1))

        a[X, Y, Z] = 1

        self.lat_size = a.shape[1]
        self.lon_size = a.shape[2]
        self.img3D_non_conv_train = a

        return self.img3D_non_conv_train


    def gaussian_filtering_train(self):
        '''Return 3D convoluted image (Gaussian filter).'''
        self.img3D_conv_train = gaussian_filter(self.img3D_non_conv_train,
            sigma = (self.sigma_x, self.sigma_y, self.sigma_z))

        return self.img3D_conv_train

    ########
    # Input: Stacking et all.

    # def stacking(self, window, lat_step, lon_step, time_step):
    #     '''Return stacked 3D images.'''
    #     # Grid starting point
    #     grid_offset = np.array([0, 0, 0])
    #     # Stacking steps to take
    #     grid_spacing = np.array([lat_step , lon_step, time_step])
    #     # Extract point coordinates
    #     coords = np.argwhere(window)
    #     flat = window.flatten()
    #     values = flat[flat != 0]
    #     # Convert point to index
    #     indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
    #     X = indexes[:,0]
    #     Y = indexes[:,1]
    #     Z = indexes[:,2]
    #     # 192 and 132 are arbitrary: size works in model
    #     stacked_crimes = np.zeros((192, 132, Z.max() + 2))

    #     for i in range(len(indexes)):

    #         if stacked_crimes[X[i], Y[i], Z[i]] == 0:
    #             stacked_crimes[X[i], Y[i], Z[i]] = values[i]
    #         else:
    #             stacked_crimes[X[i], Y[i], Z[i]] += values[i]

    #     return stacked_crimes

    def get_observation_target_train(self):
        '''Return an observation of length obs_tf + tar_tf.

        Include puffer of length raw_z to avoid data leakage.
        '''

        # Absorb impact of gaussian time sigma in sample length
        sample_length = self.obs_tf + (self.raw_z + 1) + self.tar_tf

        # Sample starting position
        position = np.random.randint(0, self.img3D_conv_train.shape[2] - sample_length)

        # Extract sample
        subsample = self.img3D_conv_train[:, :, position : position + sample_length]

        # Split sample in X an y
        observations = subsample[:, :, : self.obs_tf]

        targets = subsample[:, :, - self.tar_tf : ]

        # Stack X and y with stacking function from utils.py
        observation = stacking(observations, self.obs_lat, self.obs_lon, self.obs_time)

        target = stacking(targets, self.tar_lat, self.tar_lon, self.tar_time)

        return observation, target

    def get_X_y_train(self):
        '''Return total train observations to be used.'''
        X = []
        y = []

        for n in range(self.nb_observations_train):
            X_subsample, y_subsample = self.get_observation_target_train()
            X.append(X_subsample)
            y.append(y_subsample)

        X = np.array(X)
        y = np.array(y)

        self.X_train = X
        self.y_train = y

        return self.X_train, self.y_train



    # Test Matrix

    def from_coord_to_matrix_test(self):
        '''Return 3D matrix containing points of crime.

        Each coordinate is assigned to a bucket of size lat_meters and lon_meters.
        '''
        df = self.test_df.copy()

        # Adds 'time_index' column to dataframe
        ind = {time: index for index, time in enumerate(np.sort(df['six_hour_date'].unique()))}
        df['time_index'] = df['six_hour_date'].map(ind)

        # Matrix starting point
        grid_offset = np.array([-df['latitude'].max(), df['longitude'].min(), 0])

        # Converts bucket size (meters) to lat & lon spacing
        lat_spacing, lon_spacing = from_meters_to_steps(self.lat_meters, self.lon_meters)

        # Euclidian spacing
        grid_spacing = np.array([lat_spacing , lon_spacing, 1 ])

        # Gets point coordinates
        coords = np.array([(-lat, lon, t_ind) for lat, lon, t_ind \
                       in zip(df['latitude'], df['longitude'], df['time_index'])])

        # Convert point to index
        indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
        X = indexes[:,0]
        Y = indexes[:,1]
        Z = indexes[:,2]

        # 75th precinct maximum & minimum points
        lat_min, lat_max, lon_max, lon_min = (40.6218192717505,
                                              40.6951504231971,
                                              -73.90404639808888,
                                              -73.83559344190869)

        lat_diff = lat_max - lat_min # Distance in lat that makes up width of precinct 75
        lon_diff = lon_min - lon_max # Distance in lon that makes up width of precinct 75

        # Dim 1: distance of precinct in lat / lat_spacing
        a = np.zeros((np.round(lat_diff / lat_spacing).astype('int') + 1,
                     np.round(lon_diff / lon_spacing).astype('int') + 1,
                     Z.max() + 1))

        a[X, Y, Z] = 1

        self.lat_size = a.shape[1]
        self.lon_size = a.shape[2]
        self.img3D_non_conv_test = a

        return self.img3D_non_conv_test


    def gaussian_filtering_test(self):
        '''Return 3D convoluted image (Gaussian filter).'''
        self.img3D_conv_test = gaussian_filter(self.img3D_non_conv_test,
            sigma = (self.sigma_x, self.sigma_y, self.sigma_z))

        return self.img3D_conv_test

    ###############
    # Input Replacement: Stacking et all.

    # def stacking_test(self, window, lat_step, lon_step, time_step):
    #     '''
    #         Returns stacked crimes.
    #     '''
    #     # starting point
    #     grid_offset = np.array([0,0,0])

    #     #new steps from precise grid
    #     grid_spacing = np.array([lat_step , lon_step, time_step])

    #     #get points coordinates
    #     coords = np.argwhere(window)
    #     flat = window.flatten()
    #     values = flat[flat !=0]

    #     # Convert point to index
    #     indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
    #     X = indexes[:,0]
    #     Y = indexes[:,1]
    #     Z = indexes[:,2]

    #     #virgin matrix: 256 absolute size to be stacked to work in model!
    #     stacked_crimes = np.zeros((192, 132, Z.max() + 2))

    #     for i in range(len(indexes)):

    #         if stacked_crimes[X[i], Y[i], Z[i]] == 0:
    #             stacked_crimes[X[i], Y[i], Z[i]] = values[i]
    #         else:
    #             stacked_crimes[X[i], Y[i], Z[i]] += values[i]

    #     return stacked_crimes

    def get_observation_target_test(self):
        '''Return an observation of length obs_tf + tar_tf.

        Include puffer of length raw_z to avoid data leakage.
        '''

        # Absorb impact of gaussian time sigma in sample length
        sample_length = self.obs_tf + (self.raw_z + 1) + self.tar_tf

        # Sample starting position
        position = np.random.randint(0, self.img3D_conv_test.shape[2] - sample_length)

        # Extract sample
        subsample = self.img3D_conv_test[:, :, position : position + sample_length]

        # Split sample in X an y
        observations = subsample[:, :, : self.obs_tf]

        targets = subsample[:, :, - self.tar_tf : ]

        # Stack X and y with stacking function from utils.py
        observation = stacking(observations, self.obs_lat, self.obs_lon, self.obs_time)

        target = stacking(targets, self.tar_lat, self.tar_lon, self.tar_time)

        return observation, target

    def get_X_y_test(self):
        '''Return total test observations to be used.'''
        X = []
        y = []

        for n in range(self.nb_observations_test):
            X_subsample, y_subsample = self.get_observation_target_test()
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


    def preprocessing_X_y(self):
        '''Preprocess crime dataframe to generate model inputs.

        Pass train and test dataframes through all preprocessing steps.
        Plot on grid, pass through gaussian filter and stacking.
        Return X and y for both train and test.
        '''

        print('3. Loading train and test dataframes')
        self.load_data()

        print('4a. From coords to matrix: Train')
        self.from_coord_to_matrix_train()

        print('4b. From coords to matrix: Test')
        self.from_coord_to_matrix_test()

        print('5. Getting sigma values for Gaussian filter')
        self.getting_sigma_values()

        print('6a. Gaussian filtering: Train')
        self.gaussian_filtering_train()

        print('6b. Gaussian filtering: Test')
        self.gaussian_filtering_test()

        print('7a. Getting X, y Train')
        self.get_X_y_train()

        print('7b. Getting X, y Test')
        self.get_X_y_test()

        return self.X_train, self.y_train, self.X_test, self.y_test

