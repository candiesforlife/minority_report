
import os
import pandas as pd
import numpy as np
import pickle

class Input:
    # passer de nos map à notre liste de tensors d'entrainement

    def __init__(self):
        self.data = None

    def load_data(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        pickle_path = os.path.join(root_dir, 'raw_data', 'clean.pickle')
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        self.data = df
        return self.data

    def get_observation_target(self,img3D_conv,x_length, y_length):
        '''
        output an observation of x_length consecutive images and the y_length next images as the target
        '''
        position = np.random.randint(0,img3D_conv.shape[0]-(x_length + y_length))
        observation = img3D_conv[position:position+ x_length]
        target = img3D_conv[position+ x_length:position + (x_length + y_length)]
        del position
        return observation, target

    def get_X_y(self,img3D_conv, number_of_observations):
        X = []
        y = []
        for n in range(number_of_observations):
            X_subsample, y_subsample = get_observation_target(img3D_conv)
            X.append(X_subsample)
            y.append(y_subsample)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def from_meters_to_coords(lat_meters, lon_meters):
        """
        gives the latitude and longitude step to use for the grid buckets
        lat_meters, lon_meters = lat/lon step
        """
        #Position, decimal degrees
        lat = 40
        lon = -73

        #Earth’s radius, sphere
        R=6378137

        #offsets in meters
        dn = lat_meters
        de = lon_meters

        #Coordinate offsets in radians
        dLat = dn/R
        dLon = de/(R*np.cos(np.pi*lat/180))

        #OffsetPosition, decimal degrees
        latO = dLat * 180/np.pi
        lonO = dLon * 180/np.pi

        del lat, lon, R, dn, de, dLat, dLon

        return latO, lonO

    def from_coord_to_matrix(df, lat_meters, lon_meters):
        """
        outputs the 3D matrix of all coordinates for a given bucket height and width in meters
        """
        df=df.copy()
        #add 'time_index' column to df
        ind = {time:index for index,time in enumerate(np.sort(df['period'].unique()))}
        df['time_index'] = df['period'].map(ind)

        #initiate matrix
        grid_offset = np.array([0, -40.91553277600008,  -74.25559136315213,]) # Where do you start
        #from meters to lat/lon step
        lat_spacing, lon_spacing = from_meters_to_coords(lat_meters, lon_meters )
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


        a[Z, Y, X]=1

        del ind, grid_offset, lat_spacing, lon_spacing, grid_spacing, coords, indexes, Z, Y, X

        return a, a.shape[1], a.shape[2]


    def from_matrix_to_coord(indexes, lat_meters, lon_meters):
        """
        gives back the coordinates from a 3D matrix for a given bucket height and width
        """
        # Where do you start
        grid_offset = np.array([0, -40.91553277600008,  -74.25559136315213,])

        #from meters to lat/lon step
        lat_spacing, lon_spacing = from_meters_to_coords(lat_meters, lon_meters )

        # What's the space you consider (euclidian here)
        grid_spacing = np.array([1, lat_spacing, lon_spacing])

        result = grid_offset + indexes * grid_spacing
        return result



if __name__ == '__main__':
    input = Input()
    input.load_data()
    X, y = input.get_X_y(img3D_conv, 50)
    X.shape
    y.shape
