from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  MinMaxScaler , OneHotEncoder , StandardScaler , OrdinalEncoder , LabelEncoder  
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor




# writing function to clean test and validation df
from sklearn.metrics import mean_squared_error
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dense, LeakyReLU
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(df):
    df['rider_fname'] = df['rider_name'].apply(lambda x: x.split(',')[1])
    df['rider_fname'] = df['rider_fname'].str.strip()
    df['rider_lname'] = df['rider_name'].apply(lambda x: x.split(',')[0])
    df['rider_full_name'] = df['rider_fname'] + " " +df['rider_lname']

    df.rename(columns={'Track_Condition':'Track_Condition_surface' , 
                'track':'track_temperature'},
                    inplace=True )
    
    df['Temp_diff'] = df['Track_Temperature_Celsius'] - df['Ambient_Temperature_Celsius']
    df['Humidity_to_grip'] = df['Humidity_%'] / df['Corners_per_Lap']
    df['Degradation_Impact'] = df['Tire_Degradation_Factor_per_Lap'] * df['Laps']
    df["secs_per_km"] = (df["Circuit_Length_km"] / df["Avg_Speed_kmh"]) * 3600
    df['tire_combo'] = df['Tire_Compound_Front'] +"_" +df["Tire_Compound_Rear"]
    df['is_wet'] = df['Track_Condition_surface'].apply(lambda x : 0 if x =="Dry" else 1 )
    
    df['Corners_per_Km'] = df['Corners_per_Lap'] / df['Circuit_Length_km']
    df['Temp_Condition'] = df['Track_Temperature_Celsius'] * df['is_wet']
    df['Speed_Degradation'] = df['Tire_Degradation_Factor_per_Lap'] * df['Avg_Speed_kmh']
    df['Points_per_Year'] = df['Championship_Points'] / (df['years_active'] + 1)
    df['Finish_Rate'] = df['finishes'] / (df['starts'] + 1)
    df['Podium_Rate'] = df['podiums'] / (df['starts'] + 1)
    df['Win_Rate'] = df['wins'] / (df['starts'] + 1)
    df['Avg_Temp'] = (df['Ambient_Temperature_Celsius'] + df['Track_Temperature_Celsius']) / 2
    df['Est_Error'] = df['secs_per_km'] - df['Lap_Time_Seconds']


    df = df.drop(columns=[
                "Rider_ID",'rider_name','team_name' , 'bike_name',
                'rider_fname', 'rider_lname','circuit_name',
                'position' , 'points' 
    ])

    df['year_x'] = pd.to_datetime(df['year_x'], format='%Y').dt.year

    unique_id = df['Unique ID']
    df = df.drop(columns=['Unique ID'])

    cat_cols = df.select_dtypes(include=['object'])

    for col in cat_cols:
        df[col] = df[col].astype('category')

    df['Championship_Position'] = df['Championship_Position'].astype('category')

    return df , unique_id


def preprocess_data(df , le):
    cat_cols = df.select_dtypes(include="category").columns

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    return df

def data_split(df):
    X = df.drop(columns=['Lap_Time_Seconds'])
    y = df['Lap_Time_Seconds']

    return X , y


def data_scaler(X , y , x_scaler , y_scaler):
    
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(np.array(y).reshape(-1, 1))

    return X_scaled , y_scaled

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def model_trainer(models , X_train , y_train , X_test , y_test):
    for model_name , model in models.items():
        model.fit(X_train , y_train) 
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        print("Model : " , model_name)
        print("Training Score : " , rmse(y_train , y_pred_train))
        print("Testing Score : " , rmse(y_test , y_pred_test))

        plot_feature_importance(model , X_train)

def plot_feature_importance(model , X):
    importances = model.feature_importances_
    features = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=features[:20], y=features.index[:20])
    plt.title("Top 20 Feature Importances")
    plt.show()

# def NN_trainer(model , X_scaled, y_scaled  ):
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss='mse',
#         metrics=[tf.keras.metrics.RootMeanSquaredError()]
#     )

#     history = model.fit(X_scaled , y_scaled , epochs=55 , validation_split=0.2 ,  batch_size=271356 )

#     return model , history

# def NN_result( model,X_test , y_test , X_scaler , y_scaler):

#     X_test = X_scaler.transform(X_test)

#     y_pred_test = y_scaler.inverse_transform(model.predict(X_test))

#     print("Test Score :" , rmse(y_test ,y_pred_test ))



