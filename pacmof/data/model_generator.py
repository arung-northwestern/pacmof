import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from icecream import ic
path_to_folder = os.path.dirname(os.path.abspath(__file__))
path_to_ddec_data = os.path.join(path_to_folder,'data_DDEC.csv')
path_to_cm5_data = os.path.join(path_to_folder,'data_CM5.csv')
# ic.configureOutput(includeContext=True)
# ic(path_to_cm5_data)
# ic(path_to_ddec_data)
# ic(os.path.exists(path_to_ddec_data))
print("Please wait, compiling the RandomForestRegressor model...\n")
print("This might take a few minutes...")
df_DDEC = pd.read_csv(path_to_ddec_data)
df_CM5 = pd.read_csv(path_to_cm5_data)

X_DDEC = df_DDEC[['EN', 'IP', 'coord_no', 'coord_dis', 'EN_coord_1', 'IP_coord_1', 'EN_coord_2']]
X_CM5 = df_CM5[['EN', 'IP', 'coord_no', 'coord_dis', 'EN_coord_1', 'IP_coord_1', 'EN_coord_2']]
y_DDEC = df_DDEC['DDEC']
y_CM5 = df_CM5['CM5']
X_DDEC_train, X_DDEC_test, y_DDEC_train, y_DDEC_test = train_test_split(X_DDEC, y_DDEC, test_size=0.20, random_state=0)
X_CM5_train, X_CM5_test, y_CM5_train, y_CM5_test = train_test_split(X_CM5, y_CM5, test_size=0.20, random_state=0)
regressor=RandomForestRegressor(bootstrap=False, max_depth=20, max_features= 3, min_samples_leaf= 1, min_samples_split = 2, n_estimators = 500, verbose=2, n_jobs =-1)
print("Training the model based on DDEC charges...")
regressor.fit(X_DDEC_train, y_DDEC_train)
path_to_ddec_pkl = os.path.join(path_to_folder, 'Model_RF_DDEC.pkl')
joblib.dump(regressor, path_to_ddec_pkl, compress=3)
print("Training the model based on CM5 charges...")
path_to_cm5_pkl = os.path.join(path_to_folder, 'Model_RF_CM5.pkl')
regressor.fit(X_CM5_train, y_CM5_train)
joblib.dump(regressor, path_to_cm5_pkl, compress=3)
