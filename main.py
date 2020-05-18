# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:00:05 2020

@author: dimit
"""
import pandas as pd
cd "C:\Users\dimit\Documents\GitHub\Kaggle-House-Pricing-Dataset---Practice"
data = pd.read_csv("HousePrices_HalfMil.csv")
data.head()
data.info()
data.corr([["Prices"]].sort_values("Prices", ascending=False))
import numpy as np
data.corr()[["Prices"]].sort_values("Prices", ascending=False)
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from linear_model import LinearRegression
y = data["Price"]
y = data["Prices"]
X = data.drop(["Prices"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
lr = linear_model.LinearRegression()
model = lr.fit(X_train,y_train)
y_pred = model.predict(x_test)
y_pred = model.predict(X_test)
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

