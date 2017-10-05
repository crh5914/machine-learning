import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib
url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'  
data = pd.read_csv(url,sep=';')
print(data.head())
print('data shape:{}'.format(data.shape))
print(data.describe())
y = data.quality
X = data.drop('quality',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123,stratify=y)
# X_train_scaled = preprocessing.scale(X_train)
# print('data after scaled:{}'.format(X_train_scaled))
# print('data mean after scaled:{}'.format(X_train_scaled.mean(axis=0)))
# print('data std after scaled:{}'.format(X_train.std(axis=0)))
#scaler = preprocessing.StandardScaler().fit(X_train)
#scaler.transform(X_test)
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))
#print(pipeline.get_params())
hyperparams = {'randomforestregressor__max_features':['auto','sqrt','log2'],'randomforestregressor__max_depth':[None,5,3,1]}
clf = GridSearchCV(pipeline,hyperparams,cv=10)
clf.fit(X_train,y_train)
#print(clf.best_params_)
#print(clf.refit)
y_predict = clf.predict(X_test)
print(r2_score(y_test,y_predict))
print(mean_squared_error(y_test,y_predict))
joblib.dump(clf,'rf_regressor.pkl')
clf2 = joblib.load('rf_regressor.pkl')
clf2.predict(X_test)