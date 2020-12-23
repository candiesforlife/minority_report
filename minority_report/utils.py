##################################
#  DECORATOR
#################################
import time

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

    if period.hour < 6:
        period = period.replace(hour = 0)

    elif period.hour < 12 :
        period = period.replace(hour = 6)

    elif period.hour < 18 :
        period = period.replace(hour = 12)

    else:
        period = period.replace(hour = 18)

    return period


  # 10. Run complete_to_boolean sur df['crime_completed']
  # def crime_completed_to_boolean(self):
  #   """
  #       turns complete/incomplete into boolean value
  #   """
  #   df = self.data.copy()
  #   df['crime_completed'] = df['crime_completed'].replace({'COMPLETED': True, 'INCOMPLETE': False})
  #   self.data = df
  #   return self.data
