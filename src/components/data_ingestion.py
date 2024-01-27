import pandas as pd
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_data
from sklearn.model_selection import train_test_split



#load data
try:
    data=load_data(r'D:\Projects\BCG_Internship\src\data\final_data.csv')
    print(data.head(3))
    logging.info('Data loaded successfully!')
  
    #split data into train and test
    X = data.drop(['churn'],axis=1)  # Features/independent
    y = data['churn']                #Target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

except CustomException as e:
    raise CustomException('error occured',e)