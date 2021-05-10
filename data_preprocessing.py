# -*- coding: utf-8 -*-
"""
Created on Fri May  7 18:34:47 2021

@author: gabr8
"""

import tensorflow as tf
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Reading CSV
file = r'C:\Users\gabr8\Downloads\archive\healthcare-dataset-stroke-data.csv'
strokeData = pd.read_csv(file)
features = ['gender', 'age', 'hypertension', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

X = strokeData[features]
y = strokeData['stroke']

#Train test split
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1)
X.reset_index(inplace = True, drop = True)



#Dealing with missing values
nan_imputer = SimpleImputer()
unknown_imputer = SimpleImputer(strategy = 'most_frequent', missing_values = 'Unknown')

#Most frequent imputing aproach for 'Unknown'
X['smoking_status'] = unknown_imputer.fit_transform(X['smoking_status'].values.reshape(-1,1))
#Mean imputing aproach for nan values in bmi feature
X['bmi'] = nan_imputer.fit_transform(X['bmi'].values.reshape(-1,1))



#Dealing with categorical values
#Ordinal parameters Parameters
binaryColumns = ["gender", "ever_married", "Residence_type"] 
smoking_dict = {'formerly smoked' : 1,
              'never smoked' : 0,
              'smokes' : 2}
X_labeled = X.copy()
#Mapping smoking ordinal parameters according to dict
X_labeled['smoking_status'] = X_labeled.smoking_status.map(smoking_dict)

label_encoder = LabelEncoder()
    
for col in binaryColumns:
    X_labeled[col] = label_encoder.fit_transform(X_labeled[col])
    

#Dealing with nominal parameters
#One_hot_encoder
nominalColumns = ['work_type']
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)


X_encoded = pd.DataFrame(OH_encoder.fit_transform(X_labeled[nominalColumns]), columns = OH_encoder.get_feature_names(['work type']))
X_encoded = pd.concat([X_labeled.drop(nominalColumns, axis = 1), X_encoded], axis=1)


#Normalizing the dataframe
X_normal = tf.keras.utils.normalize(X_encoded, axis = 1)


#Saving Data with pickle

















