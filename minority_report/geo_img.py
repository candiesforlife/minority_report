
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



    def gaussian_filtering(self,img_list,x,y,z):
        img3D_conv = gaussian_filter(img_list, sigma=(x,y,z))
        for i in range(19):
            plt.imshow(img3D_conv[i+1,:,:], cmap='gray')
            plt.show()



if __name__ == '__main__':
  print('initializing geo class')
  df = GeoImg()
  print("loading data")
  df.load_data()
  print('get a sample of a month-time crimes grouped by hour')
  lats_per_image, lons_per_image = df.group_by_hour(2016, 12, 17)
  # print('Transforming into geoseries thanks to geopandas')
  # df.get_geoseries(lat, lon)
  print('saving images to numpy array')
  images_np_array = df.save_img_to_np_array(lats_per_image,lons_per_image)
  print('applying gaussian filter to np_array')
  df.gaussian_filter(images_np_array, 2,2,2)
  print('Finished!')

