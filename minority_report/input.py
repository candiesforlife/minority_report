
import os
import pandas as pd
import numpy as np
import pickle

class Input:
    # passer de nos map à notre liste de tensors d'entrainement

    def __init__(self):
        self.X = None
        self.y = None

    #IMG3D filtré
    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'img3D-conv.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.img3D_conv = df
        return self.img3D_conv


    def get_observation_target(self,img3D,
                           obs_timeframe,obs_lat,obs_lon, obs_time,
                           target_timeframe,  tar_lat,tar_lon, tar_time):
        '''
        output an observation of x_length consecutive images and the y_length next images as the target
        obs_step, obs_timeframe, target_step, target_timeframe : unit = hours
        '''
        #function from raw to hours
        length = obs_timeframe + target_timeframe
        position = np.random.randint(0, img3D_conv.shape[2] - length)
        subsample = img3D[:, :, position : position + length]
        #print(subsample.shape)
        observations, targets = np.split(subsample,[obs_timeframe], axis=2) # divide the subsample in X and y
        #print(observations.shape)
        #print(observations.min(), observations.max())
        observation = stacking(img3D, observations, obs_lat, obs_lon, obs_time) #get stacked hours for all images
        print(observation.shape)
        #print (targets.shape)
        target = stacking(img3D, targets,  tar_lat, tar_lon, tar_time )
        print(target.shape)
        return observation, target


    def get_X_y(self,img3D_conv, nb_observations, obs_tf,obs_lat,obs_lon, obs_time,
                    tar_tf, tar_lat,tar_lon, tar_time):
        '''
        outputs n observations and their associated targets
        '''
        X = []
        y = []
        for n in range(nb_observations):
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



