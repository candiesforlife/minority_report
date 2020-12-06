
import pandas as pd
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from datetime import datetime
from shapely.geometry import Point

from skimage import io, color
from scipy.ndimage import gaussian_filter
# from shapely.geometry import Point
# from geopandas import GeoSeries

from minority_report.clean_data import CleanData
from minority_report.scaling import Scaling
from scipy.ndimage import gaussian_filter

class GeoImg:

    def __init__(self):
        self.data = None
        self.sample = None
        self.lat_size = None
        self.lon_size = None
        self.indexes = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data

    def from_meters_to_coords(self,lat_meters, lon_meters):
        """
        gives the latitude and longitude step to use for the grid buckets
        lat_meters, lon_meters = lat/lon step
        """
        #Position, decimal degrees
        lat = 40
        lon = -73

        #Earth’s radius, sphere
        R = 6378137

        #offsets in meters
        dn = lat_meters
        de = lon_meters

        #Coordinate offsets in radians
        dLat = dn/R
        dLon = de/(R*np.cos(np.pi*lat/180))

        #OffsetPosition, decimal degrees
        latO = dLat * 180/np.pi
        lonO = dLon * 180/np.pi

        del lat, lon, R, dn, de, dLat, dLon #pour recuperer de la memoire dans le notebook

        return latO, lonO

    def from_coord_to_matrix(self, lat_meters, lon_meters):
        """
        outputs the 3D matrix of all coordinates for a given bucket height and width in meters
        """
        df = self.data.copy()
        #add 'time_index' column to df
        ind = {time:index for index,time in enumerate(np.sort(df['period'].unique()))}

        df['time_index'] = df['period'].map(ind)
        #initiate matrix
        grid_offset = np.array([0, -40.91553277600008,  -74.25559136315213]) # Where do you start
        #from meters to lat/lon step
        lat_spacing, lon_spacing = self.from_meters_to_coords(lat_meters, lon_meters )
        grid_spacing = np.array([1, lat_spacing , lon_spacing]) # What's the space you consider (euclidian here)


        #get points coordinates
        coords = np.array([(t_ind, -lat, lon) for t_ind, lat, lon \
                       in zip(df['time_index'],df['latitude'],df['longitude'])])


        # Convert point to index
        indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')

        Z = indexes[:,0]
        Y = indexes[:,1]
        X = indexes[:,2]

        #virgin matrix
        a = np.zeros((Z.max()+1, Y.max()+1, X.max()+1))

        a[Z, Y, X] = 1

        del ind, grid_offset, lat_spacing, lon_spacing, grid_spacing, coords,Z, Y, X

        self.lat_size = a.shape[1]
        self.lon_size = a.shape[2]
        self.indexes = indexes
        return a, self.lat_size, self.lon_size, self.indexes


    def gaussian_filtering(self,img3D,z,x,y):
        '''
          Returns img3D convoluted
        '''
        img3D_convoluted = gaussian_filter(img3D, sigma=(z,x,y))
        # max_lum = img3D_convoluted.max()
        # for i in range(19):
        #     plt.imshow(img3D_convoluted[i+1,:,:], cmap='gray', vmin=0, vmax=max_lum)
        #     plt.show()
        return img3D_convoluted

    # def from_matrix_to_coord(self,indexes, lat_meters, lon_meters):
    #     """
    #     gives back the coordinates from a 3D matrix for a given bucket height and width
    #     """
    #     df = self.data.copy()

    #     # Where do you start
    #     grid_offset = np.array([0, -40.91553277600008,  -74.25559136315213,])

    #     #from meters to lat/lon step
    #     lat_spacing, lon_spacing = self.from_meters_to_coords(df,lat_meters, lon_meters)

    #     # What's the space you consider (euclidian here)
    #     grid_spacing = np.array([1, lat_spacing, lon_spacing])

    #     result = grid_offset + indexes * grid_spacing
    #     return result


    def plotting_img3D(self, img3D): #data viz check
        for element in img3D:
            plt.imshow(element)
            plt.show()

    def save_data(self):
      '''
      Saves clean dataframe to clean data pickle
      '''
      root_dir = os.path.dirname(os.path.dirname(__file__))
      pickle_path = os.path.join(root_dir, 'raw_data', 'filtered-image.pickle')

      with open(pickle_path, 'wb') as f:
         pickle.dump(self.data, f)



if __name__ == '__main__':
  print('1. Creating an instance of GeoImg class')
  df = GeoImg()

  print("2. Loading data")
  df.load_data()

  print('3. From meters to coordinates ')
  lat_meters = 100
  lon_meters = 100
  df.from_meters_to_coords(lat_meters, lon_meters)

  print('4. From coordinates to matrix')

  #verify order
  img3D, lat_size, lon_size, indexes = df.from_coord_to_matrix(lat_meters, lon_meters)

  print('5. Gaussian filtering')
  img3Dfiltered = df.gaussian_filtering(img3D[0], 3,3,3) #to be defined

  print('6. Saving image filtered to pickle')
  df.save_data()





