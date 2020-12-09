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
