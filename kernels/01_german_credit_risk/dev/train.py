"""
This python script are responsible for reading, preparing and training a classification model for
Credit Risk analysis on a german financial institute.

* Metadata can be find at: https://www.kaggle.com/kabure/german-credit-data-with-risk
* Reference notebook: ../notebooks/train_GermanCreditRisk.ipynb

--- SUMMARY ---

1. Reading the data
2. Steps for DataPrep
    2.1 Custom tranformers
    2.2 Preprocessing pipeline
    2.3 Full prep pipeline
3. Training
    3.1 Search for best hyperparmeters
    3.2 Saving metrics
    3.4 Saving the final model

---------------------------------------------------------------
Written by Thiago Panini - Latest version: September 12th 2020
---------------------------------------------------------------
"""

# Importing libraries
import os
from utils.custom_transformers import import_data, TargetDefinition, DropDuplicates, SplitData, DummiesEncoding, \
    FeatureSelection, ColsFormatting
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from joblib import dump
import numpy as np
from lightgbm import LGBMClassifier
from utils.ml_utils import BinaryClassifiersAnalysis


"""
-----------------------------------
------- 1. READING THE DATA -------
-----------------------------------
"""
# Defining path variables for reading the data
data_path = '../data'
filename = 'german_credit_data.csv'

# Reading the file in an optimized way and eliminating the unnamed col
df = import_data(path=os.path.join(data_path, filename))
df = df.iloc[:, 1:]
df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
target_class = 'risk'
initial_features = list(df.drop(target_class, axis=1).columns)



"""
-----------------------------------
---------- 2. DATA PREP -----------
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
---------- 2. DATA PREP -----------
     2.2 Posreading Pipeline
-----------------------------------
"""

# Defining a pipeline on pos-reading data
posreading_pipeline = Pipeline([
    ('cols_formatter', ColsFormatting()),
    ('selector', FeatureSelection(features=initial_features)),
    ('features_adder', AddCreditFeatures(amount_per_year=True, weighted_amount_per_year=True))
])

# Applying the pos-reading pipeline
df_prep = posreading_pipeline.fit_transform(df)


"""
-----------------------------------
---------- 2. DATA PREP -----------
     2.3 Pretraining Pipeline
-----------------------------------
"""

# Building up a preprocessing pipeline to be applied into the entire DataFrame
pretraining_pipeline = Pipeline([
    ('target_prep', TargetDefinition(target_col=target_class, pos_class='bad', new_target_name='target')),
    ('dup_dropper', DropDuplicates()),
    ('splitter', SplitData(target='target'))
])

# Applying the pre-training pipeline
df_prep[target_class] = df[target_class]
X_train, X_test, y_train, y_test = pretraining_pipeline.fit_transform(df_prep)


"""
-----------------------------------
---------- 2. DATA PREP -----------
      2.3 Full Prep Pipeline
-----------------------------------
"""

# Splitting the data into categorical and numerical features
num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

# Building a numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Building a categorical pipeline
cat_pipeline = Pipeline([
    ('encoder', DummiesEncoding(dummy_na=True))
])

# Joining the pipelines into a complete one
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Applying the complete pipeline
X_train_prep = full_pipeline.fit_transform(X_train)
X_test_prep = full_pipeline.fit_transform(X_test)

# Returning the features after encoding
features_after_encoding = full_pipeline.named_transformers_['cat'].named_steps['encoder'].features_after_encoding
model_features = num_features + features_after_encoding


"""
-----------------------------------
---------- 2. DATA PREP -----------
     2.5. Saving the pipelines
-----------------------------------
"""

# Creating a pipeline project folder (if not exists)
if not os.path.exists('../pipelines'):
    os.makedirs('../pipelines')

# Saving each of the pipelines created
dump(posreading_pipeline, '../pipelines/posreading_pipeline.pkl')
dump(pretraining_pipeline, '../pipelines/pretraining_pipeline.pkl')
dump(full_pipeline, '../pipelines/dataprep_pipeline.pkl')


"""
-----------------------------------
---------- 3. TRAINING ------------
  3.1 Search for best hyperparams
-----------------------------------
"""

# Defining a range for searching the LGBM model
lgbm_param_grid = {
    'num_leaves': list(range(8, 92, 4)),
    'min_data_in_leaf': [10, 20, 40, 60, 100],
    'max_depth': [3, 4, 5, 6, 8, 12, 16],
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'bagging_freq': [3, 4, 5, 6, 7],
    'bagging_fraction': np.linspace(0.6, 0.95, 10),
    'reg_alpha': np.linspace(0.1, 0.95, 10),
    'reg_lambda': np.linspace(0.1, 0.95, 10),
}

lgbm_fixed_params = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

# Seting up the LGBM classifiers into a dictionary
set_classifiers = {
    'LightGBM': {
        'model': LGBMClassifier(**lgbm_fixed_params),
        'params': lgbm_param_grid
    }
}

# Creating a BinaryClassifierAnalysis object for dealing with training steps
trainer = BinaryClassifiersAnalysis()
trainer.fit(set_classifiers, X_train_prep, y_train, random_search=True, cv=5, verbose=-1)


"""
-----------------------------------
---------- 3. TRAINING ------------
3.2 Saving metrics and features info
-----------------------------------
"""

# Evaluating metrics
performance = trainer.evaluate_performance(X_train_prep, y_train, X_test_prep, y_test, cv=5, save=True, overwrite=False,
                                           performances_filepath='model_performances.csv')

# Feature importance
feat_imp = trainer.feature_importance_analysis(model_features, specific_model='LightGBM', graph=False, save=True,
                                               features_filepath='features_info.csv')

"""
-----------------------------------
---------- 3. TRAINING ------------
    3.3 Saving the final model
-----------------------------------
"""

# Returning the model to be saved
model_name = 'LightGBM'
model = trainer.classifiers_info[model_name]['estimator']

# Creating a folder for saving a model (if not exists) and saving the pkl file
if not os.path.exists('../models'):
    os.makedirs('../models')
dump(model, f'../models/{model_name.lower()}_model.pkl')