
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

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data


    def plotting_img3D(self): #data viz check
        img3D= np.zeros((743,50,50))
        img3D[15,15,15] = 1

        for element in img3D:
            plt.imshow(element)
            plt.show()


    def gaussian_filtering(self,img3D,z,x,y):
        '''
          Returns img3D convoluted
        '''
        img3D_convoluted = gaussian_filter(img3D, sigma=(z,x,y))
        max_lum = img3D_convoluted.max()
        for i in range(19):
            plt.imshow(img3D_convoluted[i+1,:,:], cmap='gray', vmin=0, vmax=max_lum)
            plt.show()
        return img3D_convoluted



if __name__ == '__main__':
  print('initializing geo class')
  df = GeoImg()
  print("loading data")
  df.load_data()
  # print('get a sample of a month-time crimes grouped by hour')
  # lats_per_image, lons_per_image = df.group_by_hour(2016, 12, 17)
  # print('Transforming into geoseries thanks to geopandas')
  # df.get_geoseries(lat, lon)
  # print('saving images to numpy array')
  # images_np_array = df.save_img_to_np_array(lats_per_image,lons_per_image)
  img3D = np.zeros((743,50,50))
  img3D[15,15,15] = 1
  print('applying gaussian filter to np_array')
  df.gaussien_filtering(img3D, 2,2,2)
  print('Finished!')

