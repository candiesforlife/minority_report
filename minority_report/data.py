'''Loads core dataframe'''

import os
import pandas as pd


class NYPD:

    def get_data(self):
        """
        Imports csv and returns a pandas dataframe
        """

        root_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(root_dir, 'raw_data', 'data.csv')
        df = pd.read_csv(csv_path, low_memory = False)[:500]

        return df
