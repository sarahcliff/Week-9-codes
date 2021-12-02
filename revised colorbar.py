#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:38:11 2021

@author: sarahcliff
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:09:04 2021

@author: sarahcliff
"""

#COPERNICUS/S1_GRD
#importing everything 
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geemap
import ee
import pandas as pd
from datetime import datetime
#import geopandas as gpd
ee.Initialize()


#functions to define
def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            
    return unique_list

def degrade(time_steps, vector):
    newvector = []
    for i in range(0, len(vector)):
        if i % time_steps == 0:
            newvector.append(vector[i])
    return newvector

start_date = "2017-09-15"
end_date = "2017-09-21"
date_vec_init = list(pd.date_range(start=start_date,end=end_date))
date_vec = degrade(6,date_vec_init)

for i in range(0,len(date_vec)):
    date_vec[i] = str(date_vec[i].strftime("%Y-%m-%d"))


#defining scope    
scope = 10#scopes 
latmin_init, lonmin_init = 55.05788, -2.703469
latmax_init, lonmax_init =55.06788, -2.713469
point = ee.Geometry.Point([lonmin_init, latmin_init]) 


#code to make matrix
matrix = np.zeros((scope, scope))
AscImg = []
DescImg = []
points_list = []
empty_dates = list()

for k in range(0, scope):
    lonmin = lonmin_init + 0.010*k
    lonmax = lonmax_init + 0.01*k
    for j in range(0, scope):
        latmin = latmin_init + 0.010*j
        latmax = latmax_init + 0.01*j
        
        point = ee.Geometry.Point([lonmin, latmin]) 
        polygon = [[[lonmin,latmin],[lonmax,latmin],[lonmin, latmax], [lonmax, latmax]]]
        finalpoly = ee.Geometry.Polygon(polygon) 
        
        points_list = []

          
        imgVV = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.eq('instrumentMode', 'IW')).filterBounds(point).select('VV')
        desc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        #asc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
            
        descfirst = desc.first()
        

        
         

        data = geemap.ee_to_numpy(ee_object = descfirst, region = point)
        if isinstance(data, np.ndarray) == True:
            points_list.append(data)
            
        newpoints_list = unique(points_list)

        matrix[j,k] = newpoints_list[0]
        #print(newpoints_list)
    print('completed 10%')
print(matrix)

#plotting the shenanigans
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
data2D = np.random.random((50, 50))
im = plt.imshow(matrix, cmap="hot")
plt.colorbar(im)
plt.show()