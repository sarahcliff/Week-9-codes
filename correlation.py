#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy import signal


# In[16]:


df_lst = pd.read_csv('/Users/sarahcliff/Desktop/sentinel files/detrended sentinel data/lstnightdetrended@2021-11-28.csv')
df_sen = pd.read_csv('/Users/sarahcliff/Desktop/sentinel files/pure sentinel data/ombh1sentinel_1@2021-11-30.csv')


# In[18]:


daysince = df_sen['date']
lst_night = df_lst['GRD Measurement']
sen_data = df_sen['Depth']


# In[19]:


print(len(lst_night), len(daysince), len(sen_data))


# In[7]:


#temporarily adjusting dataset
daysince1 = daysince[0:-3]
sen_data1 = sen_data[0:-3]


# In[20]:


#autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(sen_data1, lags=50)
plt.show()
plot_acf(lst_night, lags=50)
plt.show()


# In[23]:


import statsmodels.tsa.stattools as smt
forwards = smt.ccf(sen_data1, lst_night, adjusted=False)
plt.plot(np.linspace(0,len(forwards) *6, len(forwards)), forwards)
plt.xlabel('Lag in days')
plt.ylabel('CCF')
plt.axhline(-1.96/np.sqrt(len(sen_data1)), color='k', ls='--') 
plt.axhline(1.96/np.sqrt(len(lst_night)), color='k', ls='--', label = '95% confidence')

plt.legend()


# In[10]:


#More correlation tests

#covariance
from numpy import cov
covariance = cov(sen_data1, lst_night)
print('covariance =',covariance[1,0])

#pearsons correlation (dont know if this fits)
from scipy.stats import pearsonr
corr, _ = pearsonr(sen_data1, lst_night)
print('Pearsons correlation =',corr)

#Spearmans correlation
from scipy.stats import spearmanr
corr1, _ = spearmanr(sen_data1, lst_night)
print('Spearmans correlation: %.3f' % corr1)


# In[21]:


from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
model = ARIMA(sen_data1, daysince1)
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())


# In[15]:


from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(sen_data1)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[ ]:




