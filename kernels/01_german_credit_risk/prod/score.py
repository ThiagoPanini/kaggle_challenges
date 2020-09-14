"""
This python script are responsible for reading pkl files of processing pipeline and machine learning model already
trained for the task. This will also build a flow for receiving new data and score it with the model.

* Metadata can be find at: https://www.kaggle.com/kabure/german-credit-data-with-risk
* Reference notebook: ../notebooks/train_GermanCreditRisk.ipynb

--- SUMMARY ---

1. Reading new data received (production)
2. Reading pkl files
3. Scoring the data with the trained model

---------------------------------------------------------------
Written by Thiago Panini - Latest version: September 12th 2020
---------------------------------------------------------------
"""

# Importing libraries
import pandas as pd
import os
from utils.custom_transformers import import_data
from dev.transformers import *
from joblib import load
from datetime import datetime
from warnings import filterwarnings
filterwarnings('ignore')


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
------ 2. READING PKL FILES -------
-----------------------------------
"""

# End to end pipeline
e2e_pipeline = load('../pipelines/e2e_pipeline.pkl')


"""
-----------------------------------
------- 3. SCORING THE DATA -------
-----------------------------------
"""

# Returning the raw proba with predict_proba method
df['model_score'] = e2e_pipeline.predict_proba(df)[:, 1]

# Creating bins for splitting the score based on quantiles
bins = df['model_score'].quantile(np.arange(0, 1.01, 0.1)).values
labels = ['Faixa ' + str(i) for i in range(len(bins)-1, 0, -1)]
df['score_bin'] = pd.cut(df['model_score'], bins=bins, labels=labels, include_lowest=True)
raw_cols = list(df.columns)

# Saving the scored data
df['anomes_scoragem'] = datetime.now().strftime('%Y%m')
df['anomesdia_scoragem'] = datetime.now().strftime('%Y%m%d')
df['datetime_scoragem'] = datetime.now()
order_cols = ['anomes_scoragem', 'anomesdia_scoragem', 'datetime_sdoragem'] + raw_cols
df = df.loc[:, order_cols]
df.to_csv('../data/scored_data.csv', index=False)
