"""
This python script will allocate all the custom transformers that are specific for the project task.
The idea is to encapsulate the classes and functions used on pipelines to make the train/score codes cleaner.
"""

# Importing libraries
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


"""
-----------------------------------
----- 2. CUSTOM TRANSFORMERS ------
           2.1 Functions
-----------------------------------
"""


def indices_of_top_k(arr, k):
    """
    Selects the top k entries in an array
    :param arr: numpy array (in practice we will feed it with model feature importances array) [np.array]
    :param k: top features integer definition [int]
    :return: sorted array with filtered input array based on k entries
    """
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


"""
-----------------------------------
----- 2. CUSTOM TRANSFORMERS ------
           2.2 Classes
-----------------------------------
"""


class AddCreditFeatures(BaseEstimator, TransformerMixin):
    """
    Adds new custom features for a given dataset
    :param amount_per_year: boolean flag for inputing or not this feature on the dataset [bool, default: True]
    :param weighted_amount_per_year: boolean flag for input or not this feature on the dataset [bool, default: True]
    :return: the fit_transform method returns the dataset with the new features
    """
    def __init__(self, amount_per_year=True, weighted_amount_per_year=True):
        self.amount_per_year = amount_per_year
        self.weighted_amount_per_year = weighted_amount_per_year
        self.aux_cols = []

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # New feature: Amount per year
        if self.amount_per_year:
            X['amount_per_year'] = X['credit_amount'] / X['duration']

        # New feature: Weighted amount per year
        if self.weighted_amount_per_year:
            # Creating dictionaries for mapping saving and checking account features
            checking_acc_map = {
                'rich': 4,
                'moderate': 3,
                'little': 2
            }
            saving_acc_map = {
                'rich': 4,
                'quite_rich': 3.5,
                'moderate': 3,
                'little': 2
            }

            # Mapping checking account and reating new feature
            X['checking_acc_map'] = X['checking_account'].map(checking_acc_map)
            X['saving_acc_map'] = X['saving_accounts'].map(saving_acc_map)
            X['weighted_amount_per_year'] = (X['checking_acc_map'] * X['saving_acc_map'] * X['credit_amount'])\
                                             / X['duration']
            self.aux_cols += ['checking_acc_map', 'saving_acc_map']

        # Dropping aux map columns and returning the DataFrame
        return X.drop(self.aux_cols, axis=1)


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects the top k most important features from a trained model
    :param feature_importances: array with feature importances given by a trained model [array]
    :param k: integer that defines the top features to be filtered from the array
    :return: the fit_transform method returns the dataset filtered by top k most important features
    """
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        #self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X, y=None):
        return X[:, indices_of_top_k(self.feature_importances, self.k)]
