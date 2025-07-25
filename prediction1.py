from tabnanny import verbose
from lightgbm import LGBMClassifier, LGBMRegressor
import pandas as pd
import matplotlib.pyplot as plt
from helper import clean_data , model_trainer , data_split

df = pd.read_csv('dataset/train.csv')

df ,  df_id = clean_data(df)
val_df , val_df_id = clean_data(df=pd.read_csv('dataset/val.csv'))

X_train , y_train = data_split(df)
X_test , y_test = data_split(val_df)

models = {
    "LGBM": LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.025,
        max_depth=12,
        num_leaves=100,
        random_state=42,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=1.5,
        reg_lambda=2.0,
        
    )
}

model_trainer(
    models=models,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)




