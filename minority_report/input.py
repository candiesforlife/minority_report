import os
import pandas as pd
import numpy as np
import pickle

class Input:

    def __init__(self):
        self.X = None
        self.y = None
        self.img3D_conv = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'img3D-conv.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.img3D_conv = df
        return self.img3D_conv

    def stacking(self, window, lat_step, lon_step, time_step):
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
        #virgin matrix
        stacked_crimes = np.zeros((int(self.img3D.shape[0]/lat_step)+2, int(self.img3D.shape[1]/lon_step)+2,Z.max()+2))
        for i in range(len(indexes)):
            if stacked_crimes[X[i], Y[i], Z[i]] == 0:
                stacked_crimes[X[i], Y[i], Z[i]] = values[i]
            else:
                stacked_crimes[X[i], Y[i], Z[i]] += values[i]
        return stacked_crimes


    def get_observation_target(self,
                           obs_timeframe,obs_lat,obs_lon, obs_time,
                           target_timeframe,  tar_lat,tar_lon, tar_time):
        '''
        output an observation of x_length consecutive images and the y_length next images as the target
        obs_step, obs_timeframe, target_step, target_timeframe : unit = hours
        '''

        length = obs_timeframe + target_timeframe
        position = np.random.randint(0, img3D.shape[2] - length)
        subsample = img3D[:, :, position : position + length]

        observations, targets = np.split(subsample,[obs_timeframe], axis=2) # divide the subsample in X and y

        observation = stacking(self.img3D_conv, observations, obs_lat, obs_lon, obs_time) #get stacked hours for all images

        target = stacking(self.img3D_conv, targets,  tar_lat, tar_lon, tar_time )

        return observation, target


    def get_X_y(self,img3D_conv, nb_observations, obs_tf,obs_lat,obs_lon, obs_time,
                    tar_tf, tar_lat,tar_lon, tar_time):
        '''
        outputs n observations and their associated targets
        '''
        X = []
        y = []
        for n in range(nb_observations):
            print(f'Creating observation {n} out of {nb_observations}')
            X_subsample, y_subsample = get_observation_target(img3D_conv,
                                           obs_tf,obs_lat,obs_lon, obs_time,
                                           tar_tf,  tar_lat,tar_lon, tar_time)
            X.append(X_subsample)
            y.append(y_subsample)

        X = np.array(X)
        y = np.array(y)
        self.X = X
        self.y = y
        return self.X, self.y

    def combining_load_data_and_X_y(self, number_of_observations, x_length, y_length):
        print('8. Creating an Input instance')
        df = self.load_data()
        print('9. Loading the data from the filtered image pickle')
        self.X, self.y = self.get_X_y(df, number_of_observations, x_length, y_length)
        return self.X, self.y


    def save_data(self):
      '''
      Saves clean dataframe to clean data pickle
      '''
      root_dir = os.path.dirname(os.path.dirname(__file__))
      X_pickle_path = os.path.join(root_dir, 'raw_data', 'x.pickle')
      y_pickle_path = os.path.join(root_dir, 'raw_data', 'y.pickle')


      with open(X_pickle_path, 'wb') as f:
         pickle.dump(self.X, f)

      with open(y_pickle_path, 'wb') as f:
         pickle.dump(self.y, f)





