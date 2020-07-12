import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib
import seaborn as sns
import statsmodels.formula.api as smf

df_ori = pd.read_csv('Feature_Matrix_GICU_with_missing_values.csv')

df_new = df_ori.drop(['cohort'], axis=1)

df_new.reset_index()
df_new.set_index('ICUSTAY_ID')

features = df_new[
    ['creatinine', 'po2', 'fio2', 'pco2', 'bp_min', 'bp_max', 'pain', 'k', 'hr_min', 'hr_max', 'gcs_min', 'gcs_max',
     'bun', 'hco3', 'airway', 'resp_min', 'resp_max', 'haemoglobin', 'spo2_min', 'spo2_max', 'temp_min', 'temp_max',
     'na']]

# There should be a line here where you select the feature colulmns from the dataframe e.g.: "X = df_new[features]" and then set the target/label to "y = df.outcome"

Train, Test = train_test_split(features, train_size=0.8, random_state=1234) ## Then this would look like: "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)"

## Instead of ols you should use scikitlearn's Logistic classifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
fit = smf.ols('creatinine~'
              'po2+'
              'fio2+'
              'pco2+'
              'bp_min+'
              'bp_max+'
              'pain+'
              'k+'
              'hr_min+'
              'hr_max+'
              'gcs_min+'
              'gcs_max+'
              'bun+'
              'hco3+'
              'airway+'
              'resp_min+'
              'resp_max+'
              'haemoglobin+'
              'spo2_min+'
              'spo2_max+'
              'temp_min+'
              'temp_max+'
              'na'
              , data = Train).fit()
print(fit.summary())

pred = fit.predict(exog = Test)
RMSE = np.sqrt(mean_squared_error(Test.creatinine, pred))
