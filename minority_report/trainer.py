
import os
import pandas as pd
import numpy as np
import pickle
import itertools

import matplotlib.pyplot as plt

from datetime import datetime
from scipy.ndimage import gaussian_filter

# from minority_report.input import Input
from minority_report.matrix import Matrix

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers

class Trainer:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.y_pred = None


    # def load_data_from_input_class(self, number_of_observations, x_length, y_length):
    #     self.X, self.y = Input().combining_load_data_and_X_y(number_of_observations, x_length, y_length)
    #     return self.X, self.y

    # def holdout(self):
    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2)
    #     return self.X_train, self.X_test, self.y_train, self.y_test

    def init_model(self,x_length, y_length, lat_size, lon_size):


        print('initializing model')
        model = models.Sequential()
        print('adding conv3D 1')
        model.add(layers.Conv3D(64, kernel_size = (4,4,4), activation = 'relu', padding='same',
                            input_shape = (64, 22, 8,1)))

        print('adding MaxPooling')

        model.add(layers.MaxPooling3D(2))
        print('Flattening')
        model.add(layers.Flatten())
        print('Adding Dense Layer')
        model.add(layers.Dense(64*22*4, activation = 'relu'))
        print('Reshaping')
        model.add(layers.Reshape((64,22,4)))
        print('Compiling')
        model.compile(loss ='mse',
                     optimizer='adam',
                     metrics='mae')
        print('Done !')
        self.model = model
        return self.model

    def fit_model(self,batch_size, epochs, patience):
        self.X_train = self.X_train.reshape(-1, self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3], 1)
        es = EarlyStopping(patience = patience, restore_best_weights=True)
        self.model.fit(self.X_train, self.y_train,
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_split = 0.3,
                      callbacks = es)
        return self.model

    def evaluate_model(self):
        self.X_test = self.X_test.reshape(-1, self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3], 1)
        result = self.model.evaluate(self.X_test, self.y_test)
        return result

    def predict_model(self,):
        self.X_test = self.X_test.reshape(-1, self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3], 1)
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def save_y_pred_to_pickle(self):
        '''
        Saves to  y_pred pickler
        '''
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'y_pred.pickle')

        with open(pickle_path, 'wb') as f:
            pickle.dump(self.y_pred, f)



    def training_model(self, number_of_observations, x_length, y_length, lat_size, lon_size, batch_size, epochs, patience):
        print('7. Getting X, y from instanciating Trainer class ')
        self.load_data_from_input_class(number_of_observations, x_length, y_length)
        print('10. Train test split')
        self.holdout()
        print('11. Init model')
        self.init_model(x_length, y_length, lat_size, lon_size)
        print('12. Fit model')
        self.fit_model(batch_size, epochs, patience)
        print('13. Evaluate')
        self.evaluate_model()
        # print('14. Predict')
        # self.predict_model()
        # print('15. Save y_pred to pickle')
        # self.save_y_pred_to_pickle()
        return self



if __name__ == '__main__':

    print('1. Creating an instance of Matrix class')
    matrix = Matrix()
    print('2. Defining grid steps in meters: 15, 15')
    lat_meters, lon_meters = 15, 15
    print('3. Moving from df to preprocessed X and y')
    # 120m * 120m and 1 week time (28 * 6h images in 1 week)
    raw_x, raw_y, raw_z = 120, 120, 28 # N.B: 28 added as self.raw_z in input class
    obs_lon = 4 # 4 * 15m = 60m
    obs_lat = 4 # 4 * 15m = 60m
    obs_time = 4 # 24h - each obs of X is 14 images (each image is 24h)
    obs_tf = 56 # 4 (slots of 6h) * 14 days = 56 * 6 or 2 weeks (represents two weeks, where each img is 6h)
    tar_lon =  8 # 8 * 15m = 120m
    tar_lat = 10 # 10 * 15m = 150m
    tar_time = 2 # each image is 12h - output: one image of 12h
    tar_tf = 8 # 8 * 6h = 2 days
    nb_observations = 20
    self.X_train, self.y_train, self.X_test, self.y_test = matrix.preprocessing_X_y(lat_meters,
     lon_meters,
     raw_x, raw_y, raw_z,
     nb_observations,
     obs_tf, obs_lat, obs_lon, obs_time,
     tar_tf, tar_lat,tar_lon, tar_time)
    print('10. Saving X, y (train & test) to pickles!')
    matrix.save_data()
    print()




    x_length = 24 #24h avant
    y_length = 3 #3h apres
    number_of_observations = 50 #50 observations
    batch_size = 32
    epochs = 100
    patience = 5
    trainer = Trainer()
    trainer.training_model(number_of_observations, x_length, y_length, lat_size, lon_size, batch_size, epochs, patience)



