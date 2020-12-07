'''Returns dataframe to clean with unnecessary columns dropped'''

import os
import pandas as pd

from minority_report.data import NYPD

class PreClean:

    def __init__(self):
      # loads core dataframe
      self.data = NYPD().get_data()
