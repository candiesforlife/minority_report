'''Functions that come to use in various classes.'''

import time
import numpy as np

def simple_time_tracker(method):
  def timed(*args, **kw):
      ts = time.time()
      result = method(*args, **kw)
      te = time.time()
      if 'log_time' in kw:
          name = kw.get('log_name', method.__name__.upper())
          kw['log_time'][name] = int(te - ts)
      else:
          print(method.__name__, round(te - ts, 2))
      return result
  return timed

def round_six_hours(period):
  '''Round datetime to closest 6h unit.'''
  if period.hour < 6:
      period = period.replace(hour = 0)

  elif period.hour < 12 :
      period = period.replace(hour = 6)

  elif period.hour < 18 :
      period = period.replace(hour = 12)

  else:
      period = period.replace(hour = 18)

  return period

def from_meters_to_steps(lat_meters, lon_meters):
  '''Return latitude and longitude distance for given meter distance.

  Input desired meter distance for grid buckets (defined in trainer.py).
  Output step to use when determining grid buckets.
  '''
  # Position in decimal degrees
  lat = 40
  lon = -73

  # Earth’s radius (sphere)
  R = 6378137

  # Offset in meters
  dn = lat_meters
  de = lon_meters

  # Coordinate offsets in radians
  dLat = dn / R
  dLon = de / (R * np.cos(np.pi * lat / 180))

  # Offset position, decimal degrees
  latO = dLat * 180 / np.pi
  lonO = dLon * 180 / np.pi

  return latO, lonO

def stacking(window, lat_step, lon_step, time_step):
  '''Return stacked 3D images.'''
  # Grid starting point
  grid_offset = np.array([0, 0, 0])
  # Stacking steps to take
  grid_spacing = np.array([lat_step , lon_step, time_step])
  # Extract point coordinates
  coords = np.argwhere(window)
  flat = window.flatten()
  values = flat[flat != 0]
  # Convert point to index
  indexes = np.round((coords - grid_offset)/grid_spacing).astype('int')
  X = indexes[:,0]
  Y = indexes[:,1]
  Z = indexes[:,2]
  # 192 and 132 are arbitrary: size works in model
  stacked_crimes = np.zeros((192, 132, Z.max() + 2))

  for i in range(len(indexes)):

      if stacked_crimes[X[i], Y[i], Z[i]] == 0:
          stacked_crimes[X[i], Y[i], Z[i]] = values[i]
      else:
          stacked_crimes[X[i], Y[i], Z[i]] += values[i]

  return stacked_crimes

  # 10. Run complete_to_boolean sur df['crime_completed']
  # def crime_completed_to_boolean(self):
  #   """
  #       turns complete/incomplete into boolean value
  #   """
  #   df = self.data.copy()
  #   df['crime_completed'] = df['crime_completed'].replace({'COMPLETED': True, 'INCOMPLETE': False})
  #   self.data = df
  #   return self.data
