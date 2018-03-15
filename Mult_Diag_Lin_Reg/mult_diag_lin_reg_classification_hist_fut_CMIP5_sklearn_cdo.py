#!/usr/bin/python
'''
File Name : mult_diag_lin_reg_classification_hist_fut_CMIP5_sklearn_cdo.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 21-07-2017
Modified: Fri 21 Jul 2017 01:07:38 PM CEST
Purpose: Calculate multivariate linear regression for CMIP5 data
         print classification report 	 
	 using sklearn python packages (http://scikit-learn.org)
         Scikit-learn: Machine Learning in Python, Pedregosa et al.,
                       JMLR 12, pp. 2825-2830, 2011.
'''
import numpy as np
import netCDF4 as nc
import glob as glob
import os
from sklearn import linear_model
from sklearn.feature_selection import RFE, f_regression
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor

from os.path import expanduser
home = expanduser("~") # Get users home directory
import sys
import math
sys.path.insert(0, home+'/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
from func_write_netcdf import func_write_netcdf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

###
# Define input & output
###
experiment = 'rcp85'
archive = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'

target_var = 'TXx'
target_file = 'CLIM'
res_name_target = 'ANN'
target_time = 'MAX'
target_mask = 'maskT'
freq = 'ann'

# NAM end selection tasmaxCLIM
#diag_var = ['tasmax', 'rsds', 'pr', 'tos', 'tasmax', 'tasmax'] #'tasmax', 'rsds', 'pr', 'tos', 'tasmax', 'tasmax'
#var_file = ['CLIM','TREND', 'CLIM', 'STD', 'STD', 'TREND'] #'CLIM', 'TREND', 'CLIM', 'STD', 'STD', 'TREND'
#res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA'] #, 'JJA'
#masko = ['maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT'] #
#NAM end selection tasmaxSTD
#diag_var = ['huss', 'pr'] #'tasmax', 'tos', 'huss', 'tasmax', 'pr'
#var_file = ['CLIM',  'STD'] #'STD',  'STD', 'CLIM', 'TREND', 'STD'
#res_name = ['JJA', 'JJA'] #'JJA', 'JJA', 'JJA', 'JJA', 'JJA'
# EUR end selection tasSCALE
#diag_var = ['rsds'] #'tas', 'tas', 'tos', 'rsds'
#var_file = ['CLIM'] # 'CLIM', 'TREND', 'TREND', 'CLIM'
#res_name = ['JJA'] #, 'JJA', 'JJA', 'JJA', 'JJA'
#masko = ['maskT'] #, 'maskT'
# CNEU end selection tasSCALE
#diag_var = ['tas', 'tas',  'tos'] #'tas', 'tas',  'tos', 'tas', 'ef', 'pr'
#var_file = ['TREND', 'CLIM', 'TREND'] # 'TREND', 'STD', 'TREND', 'CLIM', 'CLIM', 'TREND'
# EUR & CNEU
diag_var = ['TXx',   'pr',  'TXx'] #'ef', 
var_file = ['TREND', 'TREND', 'STD']    #, 'TREND'
res_name = ['ANN',  'JJA',  'ANN']   #, 'JJA'
res_time = ['MAX',  'MEAN', 'MAX']   #'MEAN',  
freq_v =   ['ann',  'mon',   'ann']  #'mon', 
masko = ['maskT', 'maskT', 'maskT'] #'maskT',

nvar = len(diag_var)

ols = linear_model.LinearRegression(normalize = True) 
Labelols = "OLS"

tsr = linear_model.TheilSenRegressor(random_state = 50, copy_X = True)
Labeltsr = "Theil-Sen"

ridgeCV = linear_model.RidgeCV(normalize = True)
Labelridge = "RidgeCV"

lassoCV = linear_model.LassoCV(precompute = 'auto', cv = 5,
                               selection = 'random', normalize = True)
Labellasso = "LassoCV"

rf_maxdepth = 4 # max depth in Random Forest method
rf = RandomForestRegressor(random_state = 0, n_estimators = 100,
                           max_features = 'sqrt', max_depth = rf_maxdepth)
Labelrf = "RandomForest"

outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/%s/Mult_Var_Lin_Reg/' %(target_var)
region = 'CNEU'          #cut data over region?

syear_hist = 1980
eyear_hist = 2014
syear_fut = 2065
eyear_fut = 2099

nyears = eyear_hist - syear_hist + 1
grid = 'g025'

if (os.access(outdir, os.F_OK) == False):
        os.makedirs(outdir)

###read model data
print "Read model data"
## find all matching files in archive, loop over all of them
# first count matching files in folder
models_t = list()
model_names = list()
if region:
    name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s_%s.nc' %(
        archive, target_var, freq, target_mask, target_var, freq, experiment,
        syear_fut, eyear_fut, res_name_target, target_time, target_file,
        region)
else:
    name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s.nc' %(
        archive, target_var, freq, target_mask, target_var, freq, experiment,
        syear_fut, eyear_fut, res_name_target, target_time, target_file)
nfiles_targ = len(glob.glob(name))
print str(nfiles_targ) + ' matching files found for target variable'
for filename in glob.glob(name):
    models_t.append(filename.split('_')[4] + ' ' + filename.split('_')[6])
overlap = models_t
for v in xrange(len(diag_var)):
    models_v = list()
    if region:
        name_v = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s_%s.nc' %(
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
            freq_v[v], experiment, syear_hist, eyear_hist, res_name[v],
            res_time[v], var_file[v], region)
    else:
        name_v = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s.nc' %(
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v], freq_v[v],
            experiment, syear_hist, eyear_hist, res_name[v], res_time[v],
            var_file[v])

    for filename in glob.glob(name_v):
        models_v.append(filename.split('_')[4] + ' ' +
                        filename.split('_')[6])
    #find overlapping files for all variables
    overlap = list(set(models_v) & set(overlap))
    del models_v
nfiles = len(overlap)
print str(nfiles) + ' matching files found for all variables'
target = np.empty((nfiles))
f = 0
for filename in glob.glob(name):
    model_name = filename.split('_')[4] + ' ' + filename.split('_')[6]
    if model_name in overlap:
        #print "Read "+filename+" data"
        fh = nc.Dataset(filename, mode = 'r')
        temp_fut = fh.variables[target_var][:] # global data, time, lat, lon
        lat = fh.variables['lat'][:]
        lon = fh.variables['lon'][:]
        tmp = fh.variables[target_var]
        target_unit = tmp.units
        try:
            Fill = tmp._FillValue
        except AttributeError:
            Fill = 1e+20
        fh.close()
        if (target_file != 'SCALE'):
            if region:
                filename_hist = '%s/%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
                    archive, target_var, freq, target_mask, target_var, freq,
                    filename.split('_')[4], experiment, filename.split('_')[6],
                    syear_hist, eyear_hist,
                    res_name_target, target_time, target_file, region)
            else:
                filename_hist = '%s/%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s.nc' %(
                    archive, target_var, freq, target_mask, target_var, freq,
                    filename.split('_')[4], experiment, filename.split('_')[6],
                    syear_hist, eyear_hist, res_name_target, target_time,
                    target_file)

            fh = nc.Dataset(filename_hist, mode = 'r')
            temp_hist = fh.variables[target_var][:] # global data,time,lat,lon
            fh.close()
            temp_mod = temp_fut - temp_hist
        else:
            temp_mod = temp_fut

        #check that time axis and grid is identical for model0 and modelX
        if f != 0:
            if temp_mod0.shape != temp_mod.shape:
                print 'Warning: Dimension for models are different!'
                continue
        else:
            temp_mod0 = temp_mod[:]

        model = filename.split('_')[5]
        if model == 'ACCESS1.3':
            model = 'ACCESS1-3'
        elif model == 'FGOALS_g2':
            model = 'FGOALS-g2'
        ens = filename.split('_')[6]
        model_names.append(model + '_' + ens)

        if isinstance(temp_mod, np.ma.core.MaskedArray):
            #print type(temp_mod), temp_mod.shape
            temp_mod = temp_mod.filled(np.NaN)

        if (target_var == 'sic') and  (model == "EC-EARTH"):
            with np.errstate(invalid='ignore'):
                temp_mod[temp_mod < 0.0] = np.NaN
        # average over area and save value for each model
        w_lat = np.cos(lat * (4.0 * math.atan(1.0) / 180))
        ma_target = np.ma.masked_array(temp_mod,
                                       np.isnan(temp_mod))
        tmp1_latweight = np.ma.average(np.squeeze(ma_target),
                                       axis = 0,
                                       weights = w_lat)
        target[f] = np.nanmean(tmp1_latweight.filled(np.nan))
        f = f + 1
    else:
        continue

data = np.empty((nfiles, nvar), float, Fill)
data_unit = list()
for v in xrange(len(diag_var)):
    if region:
        name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s_%s.nc' %(
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v], freq_v[v],
            experiment, syear_hist, eyear_hist, res_name[v], res_time[v],
            var_file[v], region)
    else:
        name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%sMEAN_%s.nc' %(
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v], freq_v[v],
            experiment, syear_hist, eyear_hist, res_name[v], var_file[v]) 
    f = 0
    for filename in glob.glob(name):
        model_name = filename.split('_')[4] + ' ' + filename.split('_')[6]
        if model_name in overlap:
            #print "Read " + filename + " data"
            fh = nc.Dataset(filename, mode = 'r')
            lon = fh.variables['lon'][:]
            lat = fh.variables['lat'][:]
            temp_mod = fh.variables[diag_var[v]][:] # global data
            #check that time axis and grid is identical for models
            if f != 0:
                if temp_mod0.shape != temp_mod.shape:
                    print 'Warning: Dimension for model0 and modelX is different!'
                    continue
            else:
                temp_mod0 = temp_mod[:]
                tmp = fh.variables[diag_var[v]]
                data_unit.append(tmp.units)
            fh.close()
            model = filename.split('_')[5]
            if model == 'ACCESS1.3':
                model = 'ACCESS1-3'
            elif model == 'FGOALS_g2':
                model = 'FGOALS-g2'
            ens = filename.split('_')[6]
            model_names.append(model + '_' + ens)

            if isinstance(temp_mod, np.ma.core.MaskedArray):
                #print type(temp_mod), temp_mod.shape
                temp_mod = temp_mod.filled(np.NaN)
                #print type(temp_mod), temp_mod.shape

            if (diag_var[v] == 'sic') and  (model == "EC-EARTH"):
                with np.errstate(invalid = 'ignore'):
                    temp_mod[temp_mod < 0.0] = np.NaN
            # average over area and save value for each model
            w_lat = np.cos(lat * (4.0 * math.atan(1.0) / 180))
            ma_hist = np.ma.masked_array(temp_mod,
                                         np.isnan(temp_mod))
            tmp2_latweight = np.ma.average(np.squeeze(ma_hist),
                                           axis = 0,
                                           weights = w_lat)
            data[f, v] = np.nanmean(tmp2_latweight.filled(np.nan))
            f = f + 1
        else:
            continue

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    data, np.squeeze(target.reshape(-1, 1)), test_size = 0.5, random_state = 10)

# fit regressions
ols.fit(X_train, y_train)
tsr.fit(X_train, y_train)
ridgeCV.fit(X_train, y_train)
lassoCV.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
#OLS
print('OLS')
# The coefficients
#print('Coefficients OLS: \n', ols.coef_)
# The mean square error
print("Residual sum of squares OLS: %.2f"
      % np.mean((ols.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score OLS: %.2f' % ols.score(X_test, y_test))

#CrossValidation
cv = 5
scoresOLS = cross_val_score(ols, X_test, y_test, cv = cv)
print("Accuracy OLS: %0.2f (+/- %0.2f)" % (scoresOLS.mean(),
                                           scoresOLS.std() * 2))
print()
#Theil-Sen regressor
print(Labeltsr)
#print('Coefficients: \n', tsr.coef_)
print("Residual sum of squares: %.2f"
      % np.mean((tsr.predict(X_test) - y_test) ** 2))
print('Variance score: %.2f' % tsr.score(X_test, y_test))
scores = cross_val_score(tsr, X_test, y_test, cv = cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()
# RidgeCV
print(Labelridge)
#print('Coefficients: \n', ridgeCV.coef_)
print("Residual sum of squares: %.2f"
      % np.mean((ridgeCV.predict(X_test) - y_test) ** 2))
print('Variance score: %.2f' % ridgeCV.score(X_test, y_test))
scores = cross_val_score(ridgeCV, X_test, y_test, cv = cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()
# LassoCV
print(Labellasso)
#print('Coefficients: \n', lassoCV.coef_)
print("Residual sum of squares: %.2f"
      % np.mean((lassoCV.predict(X_test) - y_test) ** 2))
print('Variance score: %.2f' % lassoCV.score(X_test, y_test))
scores = cross_val_score(lassoCV, X_test, y_test, cv = cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print()
# RandomForest
print(Labelrf)
print("Residual sum of squares: %.2f"
      % np.mean((rf.predict(X_test) - y_test) ** 2))
print('Variance score: %.2f' % rf.score(X_test, y_test))
scores = cross_val_score(rf, X_test, y_test, cv = cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print()
