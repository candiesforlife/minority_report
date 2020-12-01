'''Loads core dataframe'''

import pandas as pd


class NYPD:

    def get_data(self):
        """
        imports csv and returns a pandas dataframe
        """

        file = '../raw_data/data.csv'

        df = pd.read_csv(file)

        return df
