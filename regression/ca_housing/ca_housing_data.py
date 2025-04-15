import pandas as pd
import numpy as np


class CAHousingData:
    def __init__(self):
        self.data = pd.read_csv("../../data/housing-geron.csv")

    def all(self):
        return self.data

    def features(self):
        return self.data.drop(columns=['median_house_value'])

    def labels(self):
        return self.data['median_house_value']

    def numerical_features(self):
        df = self.data.drop(columns=['median_house_value'])
        return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    def categorical_features(self):
        return self.data.select_dtypes(include=['object']).columns.tolist()

    def ocean_categories(self):
        return self.data['ocean_proximity'].unique().tolist()

    def income_categories(self):
        df = self.data.copy()
        df['income_cat'] = pd.cut(df['median_income'],
                                  bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                  labels=[1, 2, 3, 4, 5])
        return df['income_cat']
