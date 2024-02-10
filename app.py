#load libraries
import mlflow
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,roc_auc_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


#loading data
X=pd.read_csv('D:\\Projects\\BCG_Internship\\Artifacts\\train.csv')
Y=pd.read_csv('D:\\Projects\\BCG_Internship\\Artifacts\\test.csv')
y=np.ravel(Y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)



estimators =[200,500]
max_depths =[5,7]
for  estimator in estimators:
    for max_depth in max_depths:
        RD=RandomForestClassifier(n_estimators=estimator,max_depth=max_depth)
        RD.fit(X_train,y_train)
        f1=f1_score(y_test,RD.predict(X_test),average='macro')
        precision=metrics.precision_score(y_test,RD.predict(X_test),zero_division=1)
        recall=metrics.recall_score(y_test,RD.predict(X_test))
        
        print(f"f1_score:{f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        # Create a new MLflow Experiment
        mlflow.set_experiment('First flow')

        #start mlflow run
        with mlflow.start_run():
            # Log the parameters of the model to MLflow
            mlflow.log_param('n_estimator',estimators)
            mlflow.log_param('max_depth',max_depths)

            #log the loss metric
            mlflow.log_metric('f1_score',f1)
            mlflow.log_metric('precision',precision)
            mlflow.log_metric('Recall',recall)

            mlflow.sklearn.log_model(RD,"RandomForestClassifier")