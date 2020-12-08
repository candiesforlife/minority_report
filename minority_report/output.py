import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

class Output:

    def __init__(self):
        self.y_pred = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'y_pred.pickle')
        with open(pickle_path, 'rb') as f:
            y_pred = pickle.load(f)
        self.y_pred = y_pred
        return self.y_pred


    def from_matrix_to_coord(self,lat_meters, lon_meters):
        """
        gives back the coordinates from a 3D matrix for a given bucket height and width
        """
        results = []
        for observation in self.y_pred:
          # Where do you start
          grid_offset = np.array([0, -40.91553277600008,  -74.25559136315213,])
          #from meters to lat/lon step
          lat_spacing, lon_spacing = self.from_meters_to_steps(lat_meters, lon_meters)
          # What's the space you consider (euclidian here)
          grid_spacing = np.array([1, lat_spacing, lon_spacing])
          # gives coordonates of points where value != 0
          indexes = np.argwhere(observation)
          #print(indexes.shape)
          # index : coords de mes crimes dans mon np array
          result = grid_offset + indexes * grid_spacing
          results.append(result)
        return np.array(results)




if __name__ == '__main__':
    print('1. Creating an instance of output class')
    output = Output()
    print('2. Loading data')
    output.load_data()
    print('3. From matrix to coordinates')
    Matrix()
    coords = output.from_matrix_to_coord(matrix, lat_meters, lon_meters)
    print(coords)
    print('4. From coords to map')
    # to call
