# class Recycling:

#     # def get_geoseries(self, latitude_per_image, longitude_per_image):
#     #     final_list_geoseries =  []
#     #     sample = self.sample
#     #     for lat, lon in zip(longitude_per_image, latitude_per_image):
#     #         geometry = [Point(xy) for xy in zip(lon, lat)]
#     #         df_geopandas = sample.drop(['longitude', 'latitude'], axis=1)
#     #         geoseries_image = GeoSeries(geometry)
#     #         final_list_geoseries.append(geoseries_image)
#     #     return final_list_geoseries

#     # def visualization_from_geoseries_to_images(self,final_list_geoseries):
#     #     for geoserie in final_list_geoseries:
#     #         fig,ax = plt.subplots(figsize = (10,10))
#     #         g = geoserie.plot(ax = ax, markersize = 20, color = 'red',marker = '*',label = 'NYC')
#     #         plt.show()



#      def group_by_hour_list(self, year, month, day):#, sampling=True):
#         '''
#         get a sample of a month-time crimes grouped by hour
#         inputs = start_date info
#         '''
#         sample = df.data[['period', 'latitude', 'longitude']]

#         #if sampling:
#         inf = sample['period'] > datetime(year, month, day, 0, 0, 0)
#         next_month = month+1
#         next_year = year
#         if month == 12:
#             next_month = 1
#             next_year = year+1
#         #print(next_year, next_month)
#         sup = sample['period'] < datetime(next_year, next_month, day, 0, 0, 0)
#         sample = sample[ inf & sup ]

#         liste = np.sort(np.array(sample['period'].unique()))
#         length = len(liste)
#         lat_per_image = [[coord[1] for coord in np.array(sample[sample['period']== timestamp][['latitude', 'longitude']])]\
#                  for index, timestamp in enumerate(liste)]
#         lon_per_image = [[coord[0] for coord in np.array(sample[sample['period']== timestamp][['latitude', 'longitude']])]\
#                  for index, timestamp in enumerate(liste)]

#         return lat_per_image, lon_per_image


#     def group_by_hour(self, year, month, day):#, sampling=True):
#         '''
#         get a sample of a month-time crimes grouped by hour
#         inputs = start_date info
#         '''
#         sample = self.data[['period', 'latitude', 'longitude']]

#         #if sampling:
#         inf = sample['period'] > datetime(year, month, day, 0, 0, 0)
#         next_month = month+1
#         next_year = year
#         if month == 12:
#             next_month = 1
#             next_year = year+1
#         #print(next_year, next_month)
#         sup = sample['period'] < datetime(next_year, next_month, day, 0, 0, 0)
#         sample = sample[ inf & sup ]

#         liste = np.sort(np.array(sample['period'].unique()))
#         length = len(liste)
#         lats_per_image = []
#         lons_per_image = []
#         for index, timestamp in enumerate(liste):
#             if (index+1) % 100 ==0:
#                 print(f'Grouping timestamp {index+1}/{length}')
#             by_hour = np.array(sample[sample['period']== timestamp][['latitude', 'longitude']])
#             lats_per_image.append([coord[0] for coord in by_hour])
#             lons_per_image.append([coord[1] for coord in by_hour])
#         return lats_per_image, lons_per_image



#     def save_img_to_np_array(self,lats_per_image,lons_per_image):
#         '''
#         Save img to np_array, returns an list of np_array images.
#         '''
#         plt.rcParams["figure.figsize"] = [15,15]
#         img_list = []
#         for i in range(100):

#             fig, ax = plt.subplots()
#             ax.set_xlim(left=-74.25559136315213, right=-73.70000906387347)
#             ax.set_ylim(bottom = 40.49611539518921, top=40.91553277600008)

#             ax.scatter(lons_per_image[i], lats_per_image[i] , color='black')

#             #save plot
#             extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#             fig.savefig('../../images/image.jpeg', bbox_inches=extent)

#             #read plot as np.array
#             img = io.imread('../../images/image.jpeg')
#             grayscale = color.rgb2gray(img)

#             # add np.array to list
#             img_list.append(grayscale)
#             img_list = np.array(img_list)
#             return img_list
