"""
This python script are responsible for reading, preparing and training a classification model for
Credit Risk analysis on a german financial institute.

* Metadata can be find at: https://www.kaggle.com/kabure/german-credit-data-with-risk
* Reference notebook: ../notebooks/train_GermanCreditRisk.ipynb

--- SUMMARY ---

1. Reading the data
2. Defining Pipelines
    2.1 Pos-Reading pipeline
    2.2 Pre-Training pipeline
    2.3 Full prep pipeline
3. Training
    3.1 Defining model hyperparameters space
    3.2 Choosing the best model
4. Final Solution
    4.1 Pipeline params Grid Search
    4.2 Saving all pkl files

---------------------------------------------------------------
Written by Thiago Panini - Latest version: September 12th 2020
---------------------------------------------------------------
"""

# Importing libraries
import os
from utils.custom_transformers import import_data, TargetDefinition, DropDuplicates, SplitData, DummiesEncoding, \
    FeatureSelection, ColsFormatting
from dev.transformers import *
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from joblib import dump
import numpy as np
from lightgbm import LGBMClassifier
from utils.ml_utils import BinaryClassifiersAnalysis
from sklearn.model_selection import GridSearchCV


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
target_class = 'risk'
initial_features = [col.lower().strip().replace(' ', '_') for col in list(df.columns)[1:-1]]


"""
-----------------------------------
----- 2. DEFINING PIPELINES -------
     2.1 Pos-Reading Pipeline
-----------------------------------
"""

# Defining a pipeline on pos-reading data
pos_reading_pipeline = Pipeline([
    ('cols_formatter', ColsFormatting()),
    ('selector', FeatureSelection(features=initial_features)),
    ('features_adder', AddCreditFeatures(amount_per_year=True, weighted_amount_per_year=True))
])

# First step: applying the Pos-Reading Pipeline (source: raw df)
df_prep = pos_reading_pipeline.fit_transform(df)


"""
-----------------------------------
----- 2. DEFINING PIPELINES -------
    2.2 Pre-Training Pipeline
-----------------------------------
"""

# Building up a preprocessing pipeline to be applied into the entire DataFrame
pre_training_pipeline = Pipeline([
    ('target_prep', TargetDefinition(target_col=target_class, pos_class='bad', new_target_name='target')),
    ('dup_dropper', DropDuplicates()),
    ('splitter', SplitData(target='target'))
])

# Second step: applying the Pre-Training Pipeline (source: prep df)
df_prep[target_class] = df[target_class]
X_train, X_test, y_train, y_test = pre_training_pipeline.fit_transform(df_prep)
labels = pre_training_pipeline.named_steps['splitter'].y_


"""
-----------------------------------
----- 2. DEFINING PIPELINES -------
     2.3 Full Prep Pipeline
-----------------------------------
"""

# Splitting the data into categorical and numerical features
num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

# Building a numerical and a categorical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
cat_pipeline = Pipeline([
    ('encoder', DummiesEncoding(dummy_na=True))
])

# Joining the pipelines into a complete one
num_cat_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Third step: applying the Full Prep Pipeline (source: train & test)
X_train_prep = num_cat_pipeline.fit_transform(X_train)
X_test_prep = num_cat_pipeline.fit_transform(X_test)

# Returning features after encoding and creating a list with all model features
features_after_encoding = num_cat_pipeline.named_transformers_['cat'].named_steps['encoder'].features_after_encoding
model_features = num_features + features_after_encoding


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
trainer.fit(set_classifiers, X_train_prep, y_train, random_search=True, cv=5, verbose=1)


"""
-----------------------------------
---------- 3. TRAINING ------------
3.2 Saving metrics and features info
-----------------------------------
"""

# Evaluating metrics
performance = trainer.evaluate_performance(X_train_prep, y_train, X_test_prep, y_test, cv=5, save=True,
                                           overwrite=True, performances_filepath='model_performances.csv')

# Feature importance
feat_imp = trainer.feature_importance_analysis(model_features, graph=False, save=True,
                                               features_filepath='features_info.csv')

# Model selected
model = trainer.classifiers_info['LightGBM']['estimator']


"""
-----------------------------------
------ 4. COMPLETE SOLUTION -------
  4.1 Pipeline Params Grid Search
-----------------------------------
"""

# Returning the model to be saved


# Creating a complete pipeline for prep and predict
k = 10
e2e_pipeline = Pipeline([
    ('initial_prep', pos_reading_pipeline),
    ('full_prep', num_cat_pipeline),
    ('feature_selector', TopFeatureSelector(feat_imp['importance'], k)),
    ('model', model)
])

# Defining a param grid for searching best pipelines options
param_grid = [{
    'full_prep__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selector__k': list(range(1, len(feat_imp) + 1))
}]

# Searching for best options
grid_search_prep = GridSearchCV(e2e_pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1)
grid_search_prep.fit(X_train, y_train)

# Returning the best options
imputer_strategy = grid_search_prep.best_params_['full_prep__num__imputer__strategy']
feature_selector_k = grid_search_prep.best_params_['feature_selector__k']

# Updating the e2e pipeline with the best options found on search
e2e_pipeline.named_steps['full_prep'].named_transformers_['num'].named_steps['imputer'].strategy = imputer_strategy
e2e_pipeline.named_steps['feature_selector'].k = feature_selector_k

# Fitting the model again
e2e_pipeline.fit(df, labels)

"""
-----------------------------------
------ 4. COMPLETE SOLUTION -------
     4.3 Saving all pkl files
-----------------------------------
"""

# Creating folders for saving pkl files (if not exists
if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../pipelines'):
    os.makedirs('../pipelines')

# Saving pkl files
dump(pos_reading_pipeline, '../pipelines/pos_reading_pipeline.pkl')
dump(pre_training_pipeline, '../pipelines/pre_training_pipeline.pkl')
dump(num_cat_pipeline, '../pipelines/num_cat_pipeline.pkl')
dump(e2e_pipeline, '../pipelines/e2e_pipeline.pkl')
dump(model, '../models/lightgbm_model.pkl')
