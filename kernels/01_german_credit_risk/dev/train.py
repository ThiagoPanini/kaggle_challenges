"""
This python scripts are responsible for reading, preparing and training a classification model for
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
    3.3 Log results
    3.4 Saving the final model

---------------------------------------------------------------
Written by Thiago Panini - Latest version: September 10th 2020
---------------------------------------------------------------
"""

# Importing libraries
import os
from utils.custom_transformers import import_data, DropDuplicates, SplitData
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from joblib import dump

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

"""
-----------------------------------
---------- 2. DATA PREP -----------
      2.1 Custom Transformers
-----------------------------------
"""


class TargetDefinition(BaseEstimator, TransformerMixin):
    """
    Class for building a rule for the target col and apply it to the dataset in order to
    make the data ready for feeding into an algorithm
    """
    def __init__(self, target_col, pos_class, new_target_name='target'):
        self.target_col = target_col
        self.pos_class = pos_class
        self.new_target_name = new_target_name

        # Sanity check: new_target_name may differ from target_col
        if self.target_col == self.new_target_name:
            print('[WARNING]')
            print(f'New target column named {self.new_target_name} may be different from raw one named {self.target_col}')

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Applying the new target rule based on positive class
        df[self.new_target_name] = df[self.target_col].apply(lambda x: 1 if x == self.pos_class else 0)

        # Dropping the old target column
        return df.drop(self.target_col, axis=1)


"""
-----------------------------------
---------- 2. DATA PREP -----------
     2.2 Preprocessing Pipeline
-----------------------------------
"""

# Building up a preprocessing pipeline to be applied into the entire DataFrame
preprocessing_pipe = Pipeline([
    ('target_prep', TargetDefinition(target_col='risk', pos_class='bad', new_target_name='target')),
    ('dup_dropper', DropDuplicates()),
    ('splitter', SplitData(target='target'))
])

# Applying the preprocessing pipeline
X_train, X_test, y_train, y_test = preprocessing_pipe.fit_transform(df)


"""
-----------------------------------
---------- 2. DATA PREP -----------
      2.3 Full Prep Pipeline
-----------------------------------
"""

# Splitting the data into categorical and numerical features
num_features = [col for col, dtype in X_train.dtypes.items() if dtype != 'object']
cat_features = [col for col, dtype in X_train.dtypes.items() if dtype == 'object']

# Building a categorical pipeline