
import os
import pandas as pd
import numpy as np
import pickle
import itertools

import matplotlib.pyplot as plt

from datetime import datetime
from scipy.ndimage import gaussian_filter
import pickle
# from minority_report.input import Input
from minority_report.matrix import Matrix

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers
#from google.colab import drive

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

    def load_X_y_pickles(self):
        ''' loading pickles train and test for X and y'''
        root_dir = os.path.dirname(os.path.dirname(__file__))
        X_train_pickle_path = os.path.join(root_dir, 'raw_data', ' X_train_70.pickle')
        y_train_pickle_path = os.path.join(root_dir, 'raw_data', 'y_train_70.pickle')

        X_test_pickle_path = os.path.join(root_dir, 'raw_data', 'X_test_30.pickle')
        y_test_pickle_path = os.path.join(root_dir, 'raw_data', 'y_test_30.pickle')
        # drive.mount('/content/drive/')
        # X_train_pickle_path = ('drive/MyDrive/pickles/large_obs/X_train_140.pickle')
        # X_test_pickle_path = ('drive/MyDrive/pickles/large_obs/X_test_60.pickle')
        # y_train_pickle_path = ('drive/MyDrive/pickles/large_obs/y_train_140.pickle')
        # y_test_pickle_path = ('drive/MyDrive/pickles/large_obs/y_test_60.pickle')

        with open(X_train_pickle_path, 'rb') as f:
            self.X_train = pickle.load(f)
        with open(X_test_pickle_path, 'rb') as f:
            self.X_test = pickle.load(f)
        with open(y_train_pickle_path, 'rb') as f:
            self.y_train = pickle.load(f)
        with open(y_test_pickle_path, 'rb') as f:
            self.y_test = pickle.load(f)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def reshape(self):
        ''' reshaping for the correct channel size before passing into the CNN model.'''
        self.X_train = self.X_train.reshape(-1, self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3],1)
        self.X_test = self.X_test.reshape(-1,  self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3],1)
        self.y_train = self.y_train.reshape(-1, self.y_train.shape[1], self.y_train.shape[2], self.y_train.shape[3],1)
        self.y_test = self.y_test.reshape(-1, self.y_train.shape[1], self.y_train.shape[2], self.y_train.shape[3],1)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def init_model(self):
        print('init model')
        self.model = models.Sequential()

        print('3D conv 1')
        self.model.add(layers.Conv3D(32, kernel_size = (4,4,4), activation = 'relu', padding='same',
                                 input_shape = (256, 256, 18,1))) # 256 and 256 are mandatory but the nb of sheets of X (18) must be changed accordingly with the input
        self.model.add(layers.MaxPooling3D(2))

        print('3D conv 2')
        self.model.add(layers.Conv3D(128, kernel_size = (3,3,3), activation = 'relu', padding='same'))
        self.model.add(layers.MaxPooling3D(2))

        print('3D conv 3')
        self.model.add(layers.Conv3D(64, kernel_size = (2,2,2), activation = 'relu', padding='same'))
        self.model.add(layers.MaxPooling3D(2))

        print('3D conv 4')
        self.model.add(layers.Conv3D(16, kernel_size = (2,2,2), activation = 'relu', padding='same'))
        self.model.add(layers.MaxPooling3D(2))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256))
        self.model.add(layers.Reshape((16,16,1)))

        self.model.add(layers.UpSampling3D(size=(2, 2, 2))) #size[2] must be changed accordingly with the nb of sheets in y (change with other layers)
        self.model.add(layers.UpSampling3D(size=(2, 2, 2)))
        self.model.add(layers.UpSampling3D(size=(2, 2, 1)))
        self.model.add(layers.UpSampling3D(size=(2, 2, 1)))

        self.model.compile(loss ='mse',
                 optimizer='adam',
                 metrics='mae')
        return self.model


    def fit_model(self,batch_size, epochs, patience):
        # self.X_train = self.X_train.reshape(-1, self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3], 1)
        es = EarlyStopping(patience = patience, restore_best_weights=True)
        self.model.fit(self.X_train, self.y_train,
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_split = 0.3,
                      callbacks = es)
        return self.model

    def evaluate_model(self):
        # self.X_test = self.X_test.reshape(-1, self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3], 1)
        result = self.model.evaluate(self.X_test, self.y_test)
        return result

    def predict_model(self,):
        # self.X_test = self.X_test.reshape(-1, self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3], 1)
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



    def training_model(self,batch_size, epochs, patience):

        print('17. Loading X & y pickles, instance variables for the Training instance')
        self.load_X_y_pickles()
        print('18. Reshaping Training instance')
        self.reshape()
        print('19. Initiating & compling CNN model architecturee')
        self.init_model()
        print(f'20. Fitting model with a batch_size of {batch_size}, {epochs} epochs and a patience of {patience}')
        self.fit_model(batch_size, epochs, patience)
        print('21. Evaluating')
        self.evaluate_model()
        print('22. Predicting')
        self.predict_model()
        print('23. Saving y_pred to pickle')
        self.save_y_pred_to_pickle()
        print('24. Done!')
        return self



if __name__ == '__main__':

    print('1. Creating an instance of Matrix class')
    matrix = Matrix()
    print('2. Defining grid steps in meters: 15, 15')
    lat_meters, lon_meters = 15, 15
    print('3. Moving from df to preprocessed X and y')
    # 120m * 120m and 1 week time (28 * 6h images in 1 week)
    raw_x, raw_y, raw_z = 120, 120, 12 # N.B: 28 added as self.raw_z in input class
    obs_lon = 4 # 4 * 15m = 60m
    obs_lat = 4 # 4 * 15m = 60m
    obs_time = 4 # 24h - each obs of X is 14 images (each image is 24h)
    obs_tf = 56 # 4 (slots of 6h) * 14 days = 56 * 6 or 2 weeks (represents two weeks, where each img is 6h)
    tar_lon = 4 # 8 * 15m = 120m
    tar_lat = 4 # 10 * 15m = 150m
    tar_time = 4 # each image is 24h - output: 2 images of 24h each
    tar_tf = 8 # 12 * 6h = 2 days
    nb_observations_train = 70
    nb_observations_test = 30
    X_train, y_train, X_test, y_test = matrix.preprocessing_X_y(nb_observations_train, nb_observations_test,lat_meters, lon_meters, raw_x, raw_y, raw_z, obs_tf, obs_lat, obs_lon, obs_time, tar_tf, tar_lat,tar_lon, tar_time)

    print('10. Saving X, y (train & test) to pickles!')
    matrix.save_data()
    print('11. Checking X shape')
    print(X_train.shape)
    print('13. Checking y shape')
    print(y_train.shape)
    #print(f'14.Finished with getting train & test data + saving it into pickles with {nb_observations}')
    # print('15. Instanciating Trainer class')
    # trainer  = Trainer()
    # batch_size = 32
    # epochs = 200
    # patience = 5
    # print('16. Starting the training of the model')
    # trainer.training_model(batch_size, epochs, patience)




