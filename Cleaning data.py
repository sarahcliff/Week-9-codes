#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from datetime import datetime
import scipy


# In[24]:


#string to datetime function imported from MODIS data workbook
def to_datetime_depth(dt, dp):
    datetimevec= []
    depthvec = []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%Y-%m-%d')
            datetimevec.append(datetime_object)
            depthpoint = dp[i]
            depthvec.append(depthpoint)
    return datetimevec, depthvec

#for the albedo data
def to_datetime(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%d/%m/%Y %H:%S')
            str_time = datetime.strftime(datetime_object, '%d/%m/%Y' )
            datetimevec.append(datetime_object)
    return datetimevec

def to_date(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%Y-%m-%d')
            #str_time = datetime.strftime(datetime_object, '%d/%m/%Y' )
            datetimevec.append(datetime_object)
    return datetimevec

#for the ndvi data
def to_date_ndvi(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%Y-%m-%d')
            str_time = datetime.strftime(datetime_object, '%d/%m/%Y' )
            datetimevec.append(datetime_object)
    return datetimevec


# In[25]:


df = pd.read_csv('/Users/sarahcliff/Desktop/MODIS data/raw data/dbh6MODIS2021-12-02.csv')
data = np.array(df['Depth'])
time = to_datetime(df['date'])
print(time[0:5])


# In[36]:


#getting out nan values
#from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array


def interpolate(vector):
    for i in range(1, len(vector)):
        if np.isnan(vector[i]):
            vector[i] = vector[i-1] + (vector[i+1] - vector[i-1])/(2)
    return vector 

def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

def degrade(time_steps, vector):
    newvector = []
    for i in range(0, len(vector)-1):
        if i % time_steps == 0:
            newvector.append(vector[i])
    return newvector

#data_int = pad(data)
data_int = degrade(288, data)
new_time = degrade(288, time)
print(new_time[0:5])


# In[37]:


print(len(data_int))
plt.plot(new_time, data_int)


# In[38]:


#changing dates to days since measurement
def days_since(vec):
    days_since = []
    for i in range(0, len(vec)):
        days_since.append((i+1) * 6)
    return days_since


# In[39]:


daysince = days_since(new_time)


# In[34]:


#errors found from sentinel paper
error_data = np.linspace(0.8, 0.8, len(data_deg))
print(len(error_data))


# In[40]:


#making fft
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from scipy.fft import fft, fftfreq

# Number of samples in normalized_tone
N = 245
Sample = 6

yf = 2.0/len(data_int) * np.abs(fft(data_int)[:len(data_int)//2])[1:]
xf = np.linspace(0.0, len(daysince)/(2.0*(daysince[-1]-daysince[0])), len(daysince)//2)[1:]


plt.plot(np.abs(xf), np.abs(yf))
plt.show()
max_freq=xf[np.argmax(yf)]
power = yf[np.argmax(yf)]
days_max = 1/(max_freq) 
print(days_max)
print(power)


# In[41]:


daysince = np.array(daysince)
days_max = 365.25


# In[42]:


#minimizing chi-squared to develop sinusoid
import scipy.optimize

def chsquare_sin(x,y,yerr, initial_values, power,days_max):
    def sin_model(x, param_vals):
        return power * np.sin(2*np.pi / days_max * x + param_vals[0]) + np.mean(y)
    def chi_squared(model_params, model, x, y, yerr):
        return np.sum((y - model(x, model_params))/yerr)**2
    #deg_freedom = y.size - initial_values.size # Make sure you understand why!
    fit = scipy.optimize.minimize(chi_squared, initial_values, args=(sin_model, x,y, yerr))
    print(fit)
    a_solution = fit.x[0]
    
    phase_in_pi = a_solution / np.pi
    
    plt.figure(figsize=(8,6))
    plt.errorbar(x, 
             y, 
             yerr, 
             marker='o', 
             linestyle='None')

    plt.xlabel('x data (units)') # Axis labels
    plt.ylabel('y data (units)')

# Generate best fit line using model function and best fit parameters, and add to plot
    fit_line = sin_model(x, [a_solution])
    plt.plot(x, fit_line, 'r')
    plt.show()

    return phase_in_pi


# In[46]:


#round 1 of cleaning
in_vals = np.array([1*np.pi/2])
data_int = np.array(data_int)
print(days_max)
phase_shift = chsquare_sin(daysince, data_int, error_data, in_vals, power,days_max)
print(phase_shift)


# In[47]:


#deleting from original dataset round 1
data_new = []
sinusoid = []
mean = np.mean(data_int)
mult =2*np.pi/(days_max)
print(mult)
print(days_max)
for i in range(0, len(data_int)):
    dat = data_int[i] - (power*np.sin(daysince[i]*mult + (4/3)*np.pi)) 
    sin = power * (np.sin(daysince[i] *mult+4/3*np.pi)) + mean
    data_new.append(dat)
    sinusoid.append(sin)
plt.figure()
plt.title('original data + sinusoidal frequency from FFT')
plt.plot(daysince, data_int)
plt.plot(daysince, sinusoid)
plt.xlabel('days since 9th of september 2017')
plt.ylabel('GRD units (detrended)')
plt.show()
plt.title('detrended data')
plt.xlabel('days since 9th of september 2017')
plt.ylabel('GRD units (detrended)')
plt.plot(daysince, data_new)
plt.show()
#plt.plot(daysince, data_int)


# In[48]:


#showing that i detrended the data
#making fft
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from scipy.fft import fft, fftfreq

# Number of samples in normalized_tone
N = 245
Sample = 6

yf =  2.0/len(data_new) * np.abs(fft(data_new)[:len(data_new)//2])[1:] 
xf = np.linspace(0.0, len(daysince)/(2.0*(daysince[-1]-daysince[0])), len(daysince)//2)[1:]

plt.plot(np.abs(xf), np.abs(yf))
plt.title('FFT after detrending')
plt.show()
max_freq_1=xf[np.argmax(yf)]
power_1 = yf[np.argmax(yf)]
days_max_1 = 1/(max_freq_1) 
print(days_max_1, power_1)


# In[49]:


#round 1 of cleaning
in_vals = np.array([1/3*np.pi/2])
data_int = np.array(data_new)
phase_shift_1 = chsquare_sin(daysince, data_new, error_data, in_vals, power_1,days_max_1)
print(phase_shift_1)


# In[53]:


#deleting from original dataset round 1
data_new_new = []
sinusoid = []
mean = np.mean(data_new)
mult =2*np.pi/(days_max_1)
print(mult)
print(days_max)
for i in range(0, len(data_new)):
    dat = data_new[i] - (power_1*np.sin(daysince[i]*mult + (phase_shift_1)*np.pi)) 
    sin = power_1 *(np.sin(daysince[i] *mult+(phase_shift_1)*np.pi)) + mean
    data_new_new.append(dat)
    sinusoid.append(sin)
plt.figure()
plt.title('original data + sinusoidal frequency from FFT')
plt.plot(daysince, data_new)
plt.plot(daysince, sinusoid)
plt.xlabel('days since 9th of september 2017')
plt.ylabel('GRD units (detrended)')
plt.show()
plt.title('detrended data')
plt.xlabel('days since 9th of september 2017')
plt.ylabel('GRD units (detrended)')
plt.plot(daysince, data_new_new)
plt.show()
#plt.plot(daysince, data_int)


# In[52]:


#showing that i detrended the data
#making fft
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from scipy.fft import fft, fftfreq

# Number of samples in normalized_tone
N = 245
Sample = 6

yf =  2.0/len(data_new_new) * np.abs(fft(data_new_new)[:len(data_new_new)//2])[1:] 
xf = np.linspace(0.0, len(daysince)/(2.0*(daysince[-1]-daysince[0])), len(daysince)//2)[1:]

plt.plot(np.abs(xf), np.abs(yf))
plt.title('FFT after detrending')
plt.show()
max_freq_2=xf[np.argmax(yf)]
power_2 = yf[np.argmax(yf)]
days_max_2 = 1/(max_freq_2) 
print(days_max_2, power_2)


# In[113]:


#cleaning out lunar cycle
daysmax_lun = 29.5
power_lun = 0.1 #approximate
init_values_lun = np.array([np.pi/3])
phase_shift_lun = chsquare_sin(daysince, data_new, error_data, init_values_lun, power_lun, daysmax_lun)


# In[114]:


data_new_new = []
sinusoid = []
mean = np.mean(data_new_new)
mult =2*np.pi/(daysmax_lun)
print(mult)
print(daysmax_lun)
for i in range(0, len(data)):
    dat = data_new[i] - (power_lun*np.sin(daysince[i]*mult + (phase_shift_lun)*np.pi)) 
    sin = (np.sin(daysince[i] *mult+(phase_shift_lun)*np.pi)) + mean
    data_new_new.append(dat)
    sinusoid.append(sin)
plt.figure()
plt.title('original data + sinusoidal frequency from FFT')
plt.plot(daysince, data_new)
plt.plot(daysince, sinusoid)
plt.xlabel('days since 9th of september 2017')
plt.ylabel('GRD units (detrended)')
plt.show()
plt.title('detrended data')
plt.xlabel('days since 9th of september 2017')
plt.ylabel('GRD units (detrended)')
plt.plot(daysince, data_new_new)
plt.show()
#plt.plot(daysince, data_int)


# In[33]:


#showing that i detrended the data for lunar cycle
#making fft
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from scipy.fft import fft, fftfreq

# Number of samples in normalized_tone
N = 245
Sample = 6

yf =  2.0/len(data_new_new) * np.abs(fft(data_new_new)[:len(data_new)//2])[1:] 
xf = np.linspace(0.0, len(daysince)/(2.0*(daysince[-1]-daysince[0])), len(daysince)//2)[1:]

plt.plot(np.abs(xf), np.abs(yf))
plt.title('FFT after detrending')
plt.show()
max_freq_lun=xf[np.argmax(yf)]
power_lun_1 = yf[np.argmax(yf)]
days_max_lun_1 = 1/(max_freq_1) 
print(days_max_lun_1, power_lun_1)


# In[28]:


#linear test
import scipy.optimize
def chsquare_lin(x,y,yerr, initial_values):
    def linear_model(x, param_vals):
        return param_vals[0] + param_vals[1]*x
    def chi_squared(model_params, model, x, y, yerr):
        return np.sum((y - model(x, model_params))/yerr)**2
    #deg_freedom = y.size - initial_values.size # Make sure you understand why!
    fit = scipy.optimize.minimize(chi_squared, initial_values, args=(linear_model, x,y, yerr))
    print(fit)
    a_solution = fit.x[0]
    b_solution = fit.x[1]
    
    plt.figure(figsize=(8,6))
    plt.errorbar(x, 
             y, 
             yerr, 
             marker='o', 
             linestyle='None')

    plt.xlabel('x data (units)') # Axis labels
    plt.ylabel('y data (units)')

# Generate best fit line using model function and best fit parameters, and add to plot
    fit_line = linear_model(x, [a_solution, b_solution])
    plt.plot(x, fit_line, 'r')
    plt.show()

    return a_solution, b_solution


# In[29]:


data_new_new = np.array(data_int)
x_fit, y_fit = chsquare_lin(daysince, data_int, error_data, [np.mean(data_int), 1])


# In[32]:


#stationary tests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

#dickey fuller 
X = data_int
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
#Kwiatkowski-Phillips-Schmidt-Shin test
result_kpss = kpss(data_int, regression = 'c', nlags = 'auto')
print('KPSS Statistic: %f' % result_kpss[0])
print('p-value: %f' % result_kpss[1])
print('Critical Values:')
for key, value in result_kpss[3].items():
	print('\t%s: %.3f' % (key, value))


# In[33]:


#testing if mean and variance of data remain the same
mean = []
variance = []
for i in range(1, len(data_int)):
    mean.append(np.mean(data_int[0:i]))
    variance.append(np.var(data_int[0:i]))
    
plt.plot(daysince[1:], mean)
plt.title('mean')
plt.show()
plt.plot(daysince[1:], variance)
plt.title('variance')


# In[104]:


#shows we have a pretty stationary dataset


# In[125]:


new_time = list(new_time)
start_index= new_time.index(datetime(2017, 9, 12, 0, 0))
lst_adjust = data_new_new[start_index:-1]
date_adjust = daysince[start_index:-1]
plt.plot(date_adjust, lst_adjust)


# In[126]:


time_array_df = date_adjust
data = {'Days Since': time_array_df, 'GRD Measurement': lst_adjust}  
df_data = pd.DataFrame(data)
#creating filename
filename='lstnightdetrended@'+ str(datetime.now().strftime("%Y-%m-%d"))+'.csv'
df_data.to_csv(filename)


# In[ ]:




