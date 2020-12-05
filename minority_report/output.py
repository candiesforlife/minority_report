class Output:

    def __init__(self):
        # self



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

        result = grid_offset + indexes * grid_spacing
        return result

    def from_coords_to_map(self, series):
        # to be defined







if __name__=='main':
    print('3. From matrix to coordinates')
    df.from_matrix_to_coord(indexes, lat_meters, lon_meters)
    print('4. From coords to map')
    # to call
