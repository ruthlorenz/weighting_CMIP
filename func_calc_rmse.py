#!/usr/bin/python
'''
File Name : func_calc_rmse.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 10-03-2016
Modified: Mon 04 Jul 2016 06:51:47 PM CEST
Purpose: calculate RMSE distance measure between all models 
         and between all models and obs
'''
import numpy as np
import netCDF4 as nc # to work with NetCDF files
from os.path import expanduser
home = expanduser("~") # Get users home directory
import sys
sys.path.insert(0,home+'/scripts/plot_scripts/utils/')
from calc_RMSE_obs_mod_3D import rmse_3D

def func_calc_rmse(data_model, data_obs, lat, lon, model_names, kind=None):
        #rmse for dependecy between models an qualtiy compared to obs in last row/column
        rmse_all = np.empty((len(model_names)+1,len(model_names)+1),dtype=float)
        if (type(data_model) is dict):
                for j in xrange(len(model_names)+1):
			if (j<len(model_names)):
                        	compare = model_names[j]
                        for jj in xrange(len(model_names)+1):
				if (jj<len(model_names)):
                                	ref = model_names[jj]
                                if (j == jj):
                                        rmse_all[j,jj] = 0.0
                                elif (j==len(model_names)):
                                        rmse_all[j,jj] = rmse_3D(data_obs,data_model[ref],lat,lon,tp=kind)
                                elif (jj==len(model_names)):
                                        rmse_all[j,jj] = rmse_3D(data_model[compare],data_obs,lat,lon,tp=kind)
                                else:
                                        rmse_all[j,jj] = rmse_3D(data_model[compare],data_model[ref],lat,lon,tp = kind)
        else:
                for j in xrange(len(model_names)):
                    for jj in xrange(len(model_names)): 
                                if (j == jj):
                                        rmse_all[j,jj] = 0.0
                                elif (j==len(model_names)):
                                        rmse_all[j,jj] = rmse_3D(data_obs,data_model[jj],lat,lon,tp=kind)
                                elif (jj==len(model_names)):
                                        rmse_all[j,jj] = rmse_3D(data_model[j],data_obs,lat,lon,tp=kind)
                                else:
                                        rmse_all[j,jj] = rmse_3D(data_model[j],data_model[jj],lat,lon,tp = kind)
        #save all rmse between models in netcdf
        #rmse_tmp = np.vstack([rmse_d,rmse_q])
        #rmse_qplus = np.append(rmse_q, 0)
        #rmse_all = np.column_stack([rmse_tmp,rmse_qplus])

        return rmse_all
