"""
Script for training a Machine Learning model for predicting whereas a Restaurant could be considered as
customer's preference.
"""

"""
--------------------------------------------
---------- 1. IMPORTING LIBRARIES ----------
--------------------------------------------
"""
import os
from utils.custom_transformers import import_data
from sklearn.base import BaseEstimator, TransformerMixin


"""
--------------------------------------------
---------- 2. READING THE DATA -------------
--------------------------------------------
"""

# Defining path variables
root = r'D:\Users\thiagoPanini\github_files\kaggle_challenges\kernels\06_bangalore_restaurants'
data_path = r'D:\Users\thiagoPanini\Documents\Datasets\kaggle_ZomatoRestaurants'
raw_filename = 'zomato.csv'

# Reading the data
raw_df = import_data(path=os.path.join(data_path, raw_filename), n_lines=5000)


# Class for applying initial prep on key columns
class PrepareCostAndRate(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Extracting the approx cost feature
        X['approx_cost'] = X['approx_cost(for two people)'].astype(str).apply(lambda x: x.replace(',', '.'))
        X['approx_cost'] = X['approx_cost'].astype(float)

        # Extracting the rate feature
        X['rate_num'] = X['rate'].astype(str).apply(lambda x: x.split('/')[0])
        while True:
            try:
                X['rate_num'] = X['rate_num'].astype(float)
                break
            except ValueError as e1:
                noise_entry = str(e1).split(":")[-1].strip().replace("'", "")
                # print(f'Threating noisy entrance on rate feature: {noise_entry}')
                X['rate_num'] = X['rate_num'].apply(lambda x: x.replace(noise_entry, str(np.nan)))

        return X


# Class for selection the initial features
class InitialFeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self, initial_features=['online_order', 'book_table', 'location', 'rest_type', 'cuisines',
                                         'listed_in(type)', 'listed_in(city)', 'approx_cost', 'rate_num']):
        self.initial_features = initial_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.initial_features]


# Class for creating some features
class RestaurantAdditionalFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, multiples_types=True, total_cuisines=True, top_locations=10, top_cities=10, top_types=10):
        self.multiples_types = multiples_types
        self.total_cuisines = total_cuisines
        self.top_locations = top_locations
        self.top_cities = top_cities
        self.top_types = top_types

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Adding features based on counting of restaurant types and cuisines
        if self.multiples_types:
            X['multiple_types'] = X['rest_type'].astype(str).apply(lambda x: len(x.split(',')))
        if self.total_cuisines:
            X['total_cuisines'] = X['cuisines'].astype(str).apply(lambda x: len(x.split(',')))
            X.drop('cuisines', axis=1, inplace=True)

        # Creating for features for reducing granularity on location
        main_locations = list(X['location'].value_counts().index)[:self.top_locations]
        X['location_feature'] = X['location'].apply(lambda x: x if x in main_locations else 'Other')
        X.drop('location', axis=1, inplace=True)

        # Creating for features for reducing granularity on city
        main_cities = (X['listed_in(city)'].value_counts().index)[:self.top_cities]
        X['city_feature'] = X['listed_in(city)'].apply(lambda x: x if x in main_cities else 'Other')
        X.drop('listed_in(city)', axis=1, inplace=True)

        # Creating for features for reducing granularity on restaurant type
        main_rest_type = (X['rest_type'].value_counts().index)[:self.top_types]
        X['type_feature'] = X['rest_type'].apply(lambda x: x if x in main_rest_type else 'Other')
        X.drop('rest_type', axis=1, inplace=True)

        return X


# Class for creating a target based on a threshold (training only)
class CreateTarget(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=3.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['target'] = X['rate_num'].apply(lambda x: 1 if x >= self.threshold else 0)

        return X


# Class for splitting the data into new (not rated) and old (rated) restaurants
class SplitRestaurants(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Splits the restaurants based on rate column (rated and non rated)
        rated = X[~X['rate_num'].isnull()]
        non_rated = X[X['rate_num'].isnull()]

        # Dropping the rate column
        rated.drop('rate_num', axis=1, inplace=True)
        non_rated.drop('rate_num', axis=1, inplace=True)

        return rated, non_rated
