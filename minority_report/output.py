import os

import pandas as pd
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Output:

    def __init__(self):
        self.data = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'img3D-conv.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data


    def from_matrix_to_coord(self,indexes, lat_meters, lon_meters):
        """
        gives back the coordinates from a 3D matrix for a given bucket height and width
        """
        df = self.data.copy()

        # Where do you start
        grid_offset = np.array([0, -40.91553277600008,  -74.25559136315213,])

        #from meters to lat/lon step
        lat_spacing, lon_spacing = self.from_meters_to_coords(df,lat_meters, lon_meters)

        # What's the space you consider (euclidian here)
        grid_spacing = np.array([1, lat_spacing, lon_spacing])

        # index : coords de mes crimes dans mon np array
        result = grid_offset + indexes * grid_spacing
        return result

    def from_coords_to_map(self, series):
        # to be defined
        pass



if __name__ == '__main__':
    print('1. Creating an instance of output class')
    output = Output()
    print('2. Loading data')
    output.load_data()
    print('3. From matrix to coordinates')
    Matrix()
    coords = output.from_matrix_to_coord(indexes, lat_meters, lon_meters)
    print(coords)
    print('4. From coords to map')
    # to call
