"""
This python script are responsible for reading pkl files of processing pipeline and machine learning model already
trained for the task. This will also build a flow for receiving new data and score it with the model.

* Metadata can be find at: https://www.kaggle.com/kabure/german-credit-data-with-risk
* Reference notebook: ../notebooks/train_GermanCreditRisk.ipynb

--- SUMMARY ---

1. Reading new data received (production)
2. Building Blocks
    2.1 Custom Transformers
    2.2 Applying Pipelines
4. Scoring the data with the trained model

---------------------------------------------------------------
Written by Thiago Panini - Latest version: September 12th 2020
---------------------------------------------------------------
"""

# Importing libraries
import pandas as pd
import os
import numpy as np
from utils.custom_transformers import import_data
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load
from warnings import filterwarnings
filterwarnings('ignore')
from datetime import datetime


"""
-----------------------------------
------- 1. READING NEW DATA -------
-----------------------------------
"""

# Defining path variables for reading the data
data_path = '../data'
filename = 'german_credit_data.csv'

# Reading the file in an optimized way and eliminating the unnamed col
df = import_data(path=os.path.join(data_path, filename))


"""
-----------------------------------
------- 2. BUILDING BLOCKS --------
     2.1 Custom Transformers
-----------------------------------
"""


class AddCreditFeatures(BaseEstimator, TransformerMixin):
    """
    Class for creating new features for this credit risk analysis
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


"""
-----------------------------------
------- 2. BUILDING BLOCKS --------
      2.2 Applying Pipelines
-----------------------------------
"""

# Creating objects for pkl files
posr_pipeline = load('../pipelines/posreading_pipeline.pkl')
prep_pipeline = load('../pipelines/dataprep_pipeline.pkl')

# Applying the pipelines
df_prep = posr_pipeline.fit_transform(df)
X_prep = prep_pipeline.fit_transform(df_prep)


"""
-----------------------------------
----------- 3. SCORING ------------
-----------------------------------
"""

# Reading the trained model
model = load('../models/lightgbm_model.pkl')

# Scoring the model with the predict_proba method
df_prep['model_score'] = model.predict_proba(X_prep)[:, 1]

# Creating bins for splitting the score based on quantiles
bins = df_prep['model_score'].quantile(np.arange(0, 1.01, 0.1)).values
labels = ['Faixa ' + str(i) for i in range(len(bins)-1, 0, -1)]
df_prep['score_bin'] = pd.cut(df_prep['model_score'], bins=bins, labels=labels, include_lowest=True)

# Saving the scored data
df_prep['anomesdia'] = datetime.now().strftime('%Y%m')
df_prep['anomesdia_datetime'] = datetime.now()
df_prep.to_csv('../data/scored_data.csv', index=False)