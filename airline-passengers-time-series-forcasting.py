# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:38:43 2023

@author: Arpit Soni
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# get current working directory
os.getcwd()

# change the current working directory
os.chdir("C:\\Users\sonia\\Downloads\\01-Data-Science\\github-repositories\\Time-Series\\airline-passenger-timeseries")

data = pd.read_csv('airline_passengers.csv')

# date format shall be YYYY-MM-DD to predict passenger traveling in a month

from datetime import datetime

data['Month'] = pd.to_datetime(data['Month'])

# set month as a index value
data.set_index('Month', inplace=True)

data.plot()

# decomposition to check data components

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data['Thousands of Passengers'], period = 12)
decomposition.plot()
plt.show()

# to check my data is stationary or non-stationary
# check name - 'Augmented dickey fuller test' - mandatory step in time series 

from statsmodels.tsa.stattools import adfuller

adfuller(data['Thousands of Passengers'])

def adf_check(timeseries):
    result = adfuller(timeseries)
    print('Augmented Decay Fuller Test to check data is stationary or non-stationary')
    labels = ['ADF test statistics', 'p-value', '#lags', 'No. of observation']
    
    for a,b in zip(result, labels):
        print(b + ":" + str(a))
    if result[1]<= 0.05:
        print("strong evidence against null hypothesis and my timeseries is Stationary")
    else:
        print("Weak evidence against null hypothesis and my timeseries is Non-Stationary")

adf_check(data)    

# now we have to make data as stationary by the help of lag function

data['1st_Diff'] = data['Thousands of Passengers'] - data['Thousands of Passengers'].shift(1)

adf_check(data['1st_Diff'].dropna())    

data['2nd_Diff'] = data['1st_Diff'] - data['1st_Diff'].shift(1)

adf_check(data['2nd_Diff'].dropna())

# now the data has become stationary

'''
AIC = -2LL + 2K
K = parameter = trend(pdq) / seasonality (P D Q)
D / d - difference - integrated

trend
d = 2
p
q

seasonality
D = 1
P
Q

ARIMA - AutoRegressive Integrated Moving Average
AR
I
MA

'''

data['seasonality'] = data['Thousands of Passengers'] - data['Thousands of Passengers'].shift(12)

adf_check(data['seasonality'].dropna())

'''
seasonality
D = 1
P
Q
'''

''' to calculate p,q and P, Q by the help of acf and pacf method
acf = auto correlation
pacf - partial auto correlation

ARIMA - AutoRegressive Integrated Moving Average
AR - P/p - pacf
I = D/d - got it
MA = Q/q - acf

'''

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# to find p value
plot_pacf(data['2nd_Diff'].dropna(), lags=15)
plt.title('Partial Autocorrelation for Trend')
plt.show()

'''
Trend: d = 2, p = 2 and q ? 
correlation :
1) -0.2 to 0.2 - neutral / no correlation 
2) -0.2 to -0.6 and 0.2 to 0.6 : weak correlation 
3) -0.6 to -1 and 0.6 to 1 Strong correlation
+ve sign means directly propotional and -ve sign means inverseLy propotional 

shadded part means no correlation 
the time we get no correlation, we stop checking then 
'''

# to find q value
plot_acf(data['2nd_Diff'].dropna(), lags=15)
plt.title('Autocorrelation for Trend')
plt.show()

# q = 2

# Trend: d = 2, p = 2 and q = 2

# now for seasonality

plot_pacf(data['seasonality'].dropna(), lags=12)
plt.title('Partial Autocorrelation for Seasonality')
plt.show()
# P = 2

plot_acf(data['seasonality'].dropna(), lags=12)
plt.title('Autocorrelation for Seasonality')
plt.show()
# Q = 5

# building time series forcasting

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

model = sm.tsa.statespace.SARIMAX(data['Thousands of Passengers'], order=(2,2,2), seasonal_order = (2,1,5,12))
# order = (p,d,q)
# seasonal_order = (p,d,q,no. of orders)

result = model.fit()

print(result.summary())

# for seasonality Q = 5, AIC = 1018.524
# for seasonality Q = 4, AIC = 1016.717
# for seasonality Q = 3, AIC = 1014.958
# for seasonality Q = 2, AIC = 1014.111
# for seasonality Q = 1, AIC = 1012.201 ---> minimum AIC

model4 = sm.tsa.statespace.SARIMAX(data['Thousands of Passengers'], order=(2,2,2), seasonal_order = (2,1,4,12))
result4 = model4.fit()
print(result4.summary())

model3 = sm.tsa.statespace.SARIMAX(data['Thousands of Passengers'], order=(2,2,2), seasonal_order = (2,1,3,12))
result3 = model3.fit()
print(result3.summary())
s
model2 = sm.tsa.statespace.SARIMAX(data['Thousands of Passengers'], order=(2,2,2), seasonal_order = (2,1,2,12))
result2 = model2.fit()
print(result2.summary())

model1 = sm.tsa.statespace.SARIMAX(data['Thousands of Passengers'], order=(2,2,2), seasonal_order = (2,1,1,12))
result1 = model1.fit()
print(result1.summary())