
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
        pickle_path = os.path.join(root_dir, 'raw_data', 'filtered-image.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.img3D_conv = df
        return self.img3D_conv


    def get_observation_target(self,img3D_conv,x_length, y_length):
        '''
        output an observation of x_length consecutive images and the y_length next images as the target
        '''
        position = np.random.randint(0,img3D_conv.shape[0]-(x_length + y_length))
        observation = img3D_conv[position:position+ x_length]
        target = img3D_conv[position+ x_length:position + (x_length + y_length)]
        del position #pour recuperer de la memoire dans le notebook
        return observation, target

    def get_X_y(self,img3D_conv, number_of_observations, x_length, y_length):
        '''
        outputs n observations and their associated targets
        '''
        X = []
        y = []
        for n in range(number_of_observations):
            X_subsample, y_subsample = self.get_observation_target(img3D_conv, x_length, y_length)
            X.append(X_subsample)
            y.append(y_subsample)
        X = np.array(X)
        y = np.array(y)

        del X_subsample, y_subsample, n #pour recuperer de la memoire dans le notebook
        return X, y

    def combining_load_data_and_X_y(self, number_of_observations, x_length, y_length):
        print('1. Creating an Input instance')
        df = self.load_data()
        print('2. Loading the data from the filtered image pickle')
        self.X, self.y = self.get_X_y(df, number_of_observations, x_length, y_length)
        return self.X, self.y



