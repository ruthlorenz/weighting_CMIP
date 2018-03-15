#!/usr/bin/python
'''
File Name : mder_CMIP5_sklearn.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 01-07-2016
Modified: Tue 24 Oct 2017 04:24:35 PM CEST
Purpose: Calculate multivariate linear regression for CMIP5 data
	 using sklearn python packages (http://scikit-learn.org)
         Scikit-learn: Machine Learning in Python, Pedregosa et al.,
                       JMLR 12, pp. 2825-2830, 2011.

'''
import numpy as np
import netCDF4 as nc
import glob as glob
import os
from sklearn import linear_model
from sklearn.metrics import r2_score
#from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import RFE, RFECV
from os.path import expanduser
home = expanduser("~") # Get users home directory
import sys
import math
sys.path.insert(0, home + '/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
from func_write_netcdf import func_write_netcdf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc

rc('text', usetex = True)
rc('text.latex', preamble = '\usepackage{color}')

###
# Define input & output
###
experiment = 'rcp85'
archive = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'

target_var = 'tasmax'
target_file = 'CLIM'
target_mask = 'maskT'
freq = 'mon'
res_name_target = 'JJA'
target_time = 'MEAN'
#NAM
diag_var = ['tasmax', 'pr', 'tasmax', 'pr', 'huss', 'hfls', 'tos',
            'tasmax', 'rsds']
var_file = ['CLIM', 'CLIM', 'STD', 'STD', 'STD', 'STD', 'STD', 'TREND', 'TREND']
res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA']
masko = ['maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF',
         'maskT', 'maskT']
res_time = ['MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
            'MEAN', 'MEAN']
freq_v =   ['mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon']

# NAM end selection
#diag_var = ['tasmax', 'pr', 'rsds',  'tasmax', 'rsds'] 
#var_file = ['CLIM', 'CLIM', 'CLIM', 'TREND', 'TREND']
#res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA']

#target_var = 'tasmax'
#target_file = 'STD'
#target_mask = 'maskT'
#freq = 'mon'
#res_name_target = 'JJA'
#target_time = 'MEAN'
#NAM
#diag_var = ['tasmax', 'pr', 'rsds', 'tos', 'huss', 'huss', 'tos', 'tasmax']
#var_file = ['STD', 'STD', 'STD', 'STD', 'CLIM', 'TREND', 'CLIM', 'TREND']
#res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA']
#masko = ['maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT', 'maskF', 'maskT']
#res_time = ['MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN','MEAN']
#freq_v = ['mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon']

#EUR
#target_var = 'tas'
#target_file = 'SCALE'
#target_mask = 'maskT'
#diag_var = ['tas',   'rlus',  'ef',    'tas',   'psl',   'tos',   'psl',   'psl',   'ef',    'tasclt', 'rnet', 'tos', 'tasclt'] #
#var_file = ['CLIM',  'CLIM',  'CLIM',  'TREND', 'STD',   'STD',   'CLIM',  'STD',   'TREND', 'CORR',  'CLIM',  'STD', 'CORR'] #
#res_name = ['JJA',   'JJA',   'JJA',   'JJA',   'MAM',   'MAM',   'SON',   'SON',   'SON',   'SON',   'DJF',   'DJF', 'DJF']
#masko =    ['maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT']

#EUR & CNEU
#target_var = 'TXx'
#target_file = 'CLIM'
#target_mask = 'maskT'
#res_name_target = 'ANN'
#target_time = 'MAX'
#freq = 'ann'

#EUR
#diag_var = ['tas',   'rsds',  'ef',    'tas',    'tos',    'psl',  'psl',   'psl',   'ef',    'rnet', 'tos', 'tasclt'] #
#var_file = ['CLIM',  'CLIM',  'CLIM',  'TREND',  'STD',   'STD',   'CLIM',  'STD',   'TREND', 'CLIM',  'STD',  'CORR'] #
#res_name = ['JJA',   'JJA',   'JJA',   'JJA',    'MAM',   'MAM',   'SON',   'SON',   'SON',   'DJF',   'DJF',   'DJF']
#masko =    ['maskT', 'maskT', 'maskT', 'maskT',  'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT']

#CNEU
#diag_var = ['TXx',  'TXx',  'TXx',   'rsds',  'tas',   'psl',   'pr',   'rnet',   'psl',   'ef',    'tos',  'tos',   'rnet',  'huss',  'tasclt','tos',  'rnet',  'psl',  'tas'] #
#var_file = ['STD',  'TREND','CLIM',  'CLIM',  'STD',   'STD',   'TREND', 'TREND', 'TREND', 'TREND', 'CLIM', 'STD',   'STD',   'STD',   'CORR',  'STD',  'STD',   'STD',  'TREND'] #
#res_name = ['ANN',  'ANN',  'ANN',   'JJA',   'JJA',   'JJA',   'JJA',   'JJA',   'JJA',   'JJA',   'MAM',  'MAM',   'MAM',   'MAM',   'MAM',   'SON',  'SON',   'SON',  'SON']
#res_time = ['MAX',  'MAX',  'MAX',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',  'MEAN', 'MEAN']
#masko =    ['maskT','maskT','maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF','maskF', 'maskT', 'maskT', 'maskT', 'maskF','maskT', 'maskT','maskT']
#freq_v =   ['ann',  'ann',  'ann',   'mon',   'mon',   'mon',   'mon',   'mon',   'mon',   'mon',   'mon',  'mon',   'mon',   'mon',   'mon',   'mon',  'mon',   'mon',  'mon']

#EUR
#diag_var = ['TXx',   'TXx','tos',   'rnet', 'tos',   'hfls',  'pr',    'rnet', 'ef',   'rnet',  'ef',   'rsds', 'hfls',  'tasclt', 'pr',    'rsds']
#var_file = ['TREND', 'CLIM', 'CLIM',  'CLIM', 'STD',    'STD',   'TREND', 'TREND','TREND','STD',   'STD',  'TREND','TREND', 'CORR',   'CLIM',  'CLIM']
#res_name = ['ANN',   'ANN',   'JJA',   'JJA',  'JJA',   'JJA',   'JJA',   'JJA',  'JJA',  'MAM',   'SON',  'SON',  'SON',   'SON',    'DJF',   'DJF']
#res_time = ['MAX',   'MAX',   'MEAN',  'MEAN', 'MEAN',  'MEAN', 'MEAN',  'MEAN', 'MEAN', 'MEAN',  'MEAN', 'MEAN', 'MEAN',  'MEAN',   'MEAN',  'MEAN']
#masko =    ['maskT', 'maskT', 'maskF', 'maskT','maskF', 'maskT', 'maskT', 'maskT','maskT','maskT', 'maskT','maskT','maskT', 'maskT',  'maskT', 'maskT']
#freq_v =   ['ann',   'ann',   'mon',   'mon',  'mon',   'mon',   'mon',   'mon',  'mon',  'mon',   'mon',  'mon',  'mon',   'mon',    'mon',   'mon']

# CNEU clt
#target_var = 'clt'
#target_file = 'CLIM'
#target_mask = 'maskF'
#res_name_target = 'JJA'
#target_time = 'MEAN'
#freq = 'mon'

#diag_var = ['clt', 'tasmax', 'rnet', 'huss', 'ef', 'tos',
#            'clt', 'tas', 'pr', 'rnet', 'huss', 'hurs',
#            'clt', 'tasmax', 'pr', 'rsds', 'huss', 'ef', 'tashuss']
#var_file = ['CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'CORR']
#res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA']
#res_time = ['MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN',
#            'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN',
#            'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN']
#masko =    ['maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF',
#            'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF',
#            'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT'] 
#freq_v =   ['mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#            'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#            'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon']

# CNEU hurs
#target_var = 'hurs'
#target_file = 'CLIM'
#target_mask = 'maskF'
#res_name_target = 'JJA'
#target_time = 'MEAN'
#freq = 'mon'

#diag_var = ['hurs', 'pr', 
#            'hurs', 'tasmax', 'tasmin',
#            'hurs', 'pr', 'psl', 'hfls', 'tasclt', 'tashuss']
#var_file = ['CLIM', 'CLIM', 
#            'STD', 'STD', 'STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'CORR', 'CORR']
#res_name = ['JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA']
#res_time = ['MEAN', 'MEAN',
#            'MEAN', 'MEAN', 'MEAN',
#            'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN', 'MEAN']
#masko =    ['maskF', 'maskT', 
#            'maskF', 'maskT', 'maskT',
#            'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT'] 
#freq_v =   ['mon', 'mon',
#            'mon', 'mon', 'mon',
#            'mon', 'mon', 'mon', 'mon', 'mon', 'mon']

nvar = len(diag_var)

ols = linear_model.LinearRegression(normalize = True) 
Labelols = "OLS"
#clf = linear_model.Ridge(normalize = True)
#Labelclf = "Ridge"
#clf = linear_model.RidgeCV(normalize = True)
#Labelclf = "RidgeCV"
#clf = linear_model.BayesianRidge(compute_score = True)
#Labelclf = "BayesianRidge"
clf = linear_model.TheilSenRegressor(random_state = 50, copy_X = True)
Labelclf = "Theil-Sen"

outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/%s/Mult_Var_Lin_Reg/' %(target_var)
region = 'NAM'          #cut data over region?

syear_hist = 1980
eyear_hist = 2014
syear_fut = 2065
eyear_fut = 2099

nyears = eyear_hist - syear_hist + 1
grid = 'g025'

if (os.access(outdir, os.F_OK) == False):
        os.makedirs(outdir)

### read model data
print "Read model data"
## find all matching files in archive, loop over all of them
# first count matching files in folder
models_t = list()
model_names = list()
if region:
    name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s_%s.nc' %(
        archive, target_var, freq, target_mask, target_var, freq, experiment,
        syear_fut, eyear_fut, res_name_target, target_time, target_file, region)
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
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
            freq_v[v], experiment, syear_hist, eyear_hist, res_name[v],
            res_time[v], var_file[v])

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
        if tmp.units == 'degC':
            target_unit = '$^{\circ}$C'
        else:
            target_unit = tmp.units
        try:
            Fill = tmp._FillValue
        except AttributeError:
            Fill = 1e+20
        fh.close()
        if (target_file != 'SCALE'):
            if region:
                filename_hist = '%s/%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s_%s.nc'%(
                    archive, target_var, freq, target_mask, target_var, freq,
                    filename.split('_')[4], experiment, filename.split('_')[6],
                    syear_hist, eyear_hist, res_name_target, target_time,
                    target_file, region)
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

        # check that time axis and grid is identical for model0 and modelX
        if f != 0:
            if temp_mod0.shape != temp_mod.shape:
                print 'Warning: Dimension for models are different!'
                continue
        else:
            temp_mod0 = temp_mod[:]

        model = filename.split('_')[4]
        if model == 'ACCESS1.3':
                model = 'ACCESS1-3'
        elif model == 'FGOALS_g2':
                model = 'FGOALS-g2'
        ens = filename.split('_')[6]
        model_names.append(model + '_' + ens)

        if isinstance(temp_mod, np.ma.core.MaskedArray):
            temp_mod = temp_mod.filled(np.NaN)

        if (target_var == 'sic') and  (model == "EC-EARTH"):
            with np.errstate(invalid='ignore'):
                temp_mod[temp_mod < 0.0] = np.NaN
        # average over area and save value for each model
        w_lat = np.cos(lat * (4.0 * math.atan(1.0) / 180))
        ma_target = np.ma.masked_array(temp_mod, np.isnan(temp_mod))
        tmp1_latweight = np.ma.average(np.squeeze(ma_target),
                                       axis = 0, weights = w_lat)
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
        name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s.nc' %(
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v], freq_v[v],
            experiment, syear_hist, eyear_hist, res_name[v], res_time[v],
            var_file[v]) 
    f = 0
    for filename in glob.glob(name):
        model_name = filename.split('_')[4] + ' ' + filename.split('_')[6]
        if model_name in overlap:
            #print "Read " + filename + " data"
            fh = nc.Dataset(filename, mode = 'r')
            lon = fh.variables['lon'][:]
            lat = fh.variables['lat'][:]
            temp_mod = fh.variables[diag_var[v]][:] # global data
            # check that time axis and grid is identical for models
            if f != 0:
                if temp_mod0.shape != temp_mod.shape:
                    print('Warning: Dimension for model0 and modelX' + 
                          ' is different!')
                    continue
            else:
                temp_mod0 = temp_mod[:]
                tmp = fh.variables[diag_var[v]]
            if tmp.units == 'degC':
                units = '$^\circ$C'
            else:
                units = tmp.units
            fh.close()
            model = filename.split('_')[4]
            if model == 'ACCESS1.3':
                model = 'ACCESS1-3'
            elif model == 'FGOALS_g2':
                model = 'FGOALS-g2'
            ens = filename.split('_')[6]

            if isinstance(temp_mod, np.ma.core.MaskedArray):
                temp_mod = temp_mod.filled(np.NaN)

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
    data_unit.append(units)

###read obs data
print "Read obs data"
obs_areaavg = np.empty((nvar))
eint_obs_areaavg = np.empty((nvar))
m2_obs_areaavg = np.empty((nvar))
for v in xrange(len(diag_var)):
    for obsdata in ['Obs', 'ERAint', 'MERRA2']:
        if ((obsdata == 'Obs') and
            ((diag_var[v] == 'rlus') or (diag_var[v] == 'rsds') or
             (diag_var[v] == 'rnet') or (diag_var[v] == 'clt'))):
            if region:
                name = '%s/%s/%s/%s/%s_%s_%s_2000-%s_%s%s_%s_%s.nc' %(
                    archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                    freq_v[v], obsdata, eyear_hist, res_name[v], res_time[v],
                    var_file[v], region)
            else:
                name = '%s/%s/%s/%s/%s_%s_%s_2000-%s_%s%s_%s.nc' %(
                    archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                    freq_v[v], obsdata, eyear_hist, res_name[v], res_time[v],
                    var_file[v])
        elif ((obsdata == 'Obs') and ((diag_var[v] == 'tasclt'))):
            if res_name[v] == 'DJF':
                if region:
                    name = '%s/%s/%s/%s/%s_%s_%s_2001-2010_%s%s_%s_%s.nc' %(
                        archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                        freq_v[v], obsdata, res_name[v], res_time[v],
                        var_file[v], region)
                else:
                    name = '%s/%s/%s/%s/%s_%s_%s_2001-2010_%s%s_%s.nc' %(
                        archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                        freq_v[v], obsdata, res_name[v], res_time[v],
                        var_file[v])
            elif res_name[v] == 'SON':
                if region:
                    name = '%s/%s/%s/%s/%s_%s_%s_2000-2010_%s%s_%s_%s.nc' %(
                        archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                        freq_v[v], obsdata, res_name[v], res_time[v],
                        var_file[v], region)
                else:
                    name = '%s/%s/%s/%s/%s_%s_%s_2000-2010_%s%s_%s.nc' %(
                        archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                        freq_v[v], obsdata, res_name[v], res_time[v],
                        var_file[v])
            else:
                if region:
                    name = '%s/%s/%s/%s/%s_%s_%s_2000-2011_%s%s_%s_%s.nc' %(
                        archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                        freq_v[v], obsdata, res_name[v], res_time[v],
                        var_file[v], region)
                else:
                    name = '%s/%s/%s/%s/%s_%s_%s_2000-2011_%s%s_%s.nc' %(
                        archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                        freq_v[v], obsdata, res_name[v], res_time[v],
                        var_file[v])
        elif ((obsdata == 'Obs') and ((diag_var[v] == 'ef'))):
            obs_areaavg[v] = np.nan
            continue
        elif ((obsdata == 'ERAint') and
              ((diag_var[v] == 'huss') or (diag_var[v] == 'tashuss'))):
            eint_obs_areaavg[v] = np.nan
            continue
        else:
            if region:
                name = '%s/%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
                    archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                    freq_v[v], obsdata, syear_hist, eyear_hist, res_name[v],
                    res_time[v], var_file[v], region)
            else:
                name = '%s/%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s.nc' %(
                    archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                    freq_v[v], obsdata, syear_hist, eyear_hist, res_name[v],
                    res_time[v], var_file[v])
        fh = nc.Dataset(name, mode = 'r')
        lon = fh.variables['lon'][:]
        lat = fh.variables['lat'][:]
        temp_obs = fh.variables[diag_var[v]][:] # global data, time, lat, lon
        fh.close()
        if isinstance(temp_obs, np.ma.core.MaskedArray):
            temp_obs = temp_obs.filled(np.NaN)
        # average over area
        ma_obs = np.ma.masked_array(temp_obs, np.isnan(temp_obs))
        w_lat = np.cos(lat * (4.0 * math.atan(1.0) / 180))
        tmp_latweight = np.ma.average(np.squeeze(ma_obs), axis = 0,
                                      weights = w_lat)
        if (obsdata == 'MERRA2'):
            m2_obs_areaavg[v] = np.nanmean(tmp_latweight.filled(np.nan))
        elif (obsdata == 'ERAint'):
            eint_obs_areaavg[v] = np.nanmean(tmp_latweight.filled(np.nan))
        elif (obsdata == 'Obs'):
            obs_areaavg[v] = np.nanmean(tmp_latweight.filled(np.nan))
if not region:
    region = 'GLOBAL'
colors = ['purple', 'royalblue', 'seagreen', 'darkorange', 'teal',
          'deepskyblue', 'darkgrey', 'deeppink', 'gold', 'grey', 'blue', 'red',
          'green', 'yellow', 'black', 'crimson', 'magenta', 'cyan', 'chocolate',
          'steelblue', 'plum', 'indigo', 'indianred', 'goldenrod', 'blueviolet',
          'darkblue', 'maroon', 'limegreen']
fig = plt.figure(figsize = (3.5*nvar, 3.5*nvar))
gs = gridspec.GridSpec(nvar, nvar)
#keep same fonts for LaTeX R^2
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#matplotlib.rcParams['xtick.labelsize'] = 15
#matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams.update({'font.size': 15})

ma_change_target_areaavg = np.ma.masked_array(target, np.isnan(target))
for v in range(nvar):
    ma_data_areaavg = np.ma.masked_array(data[:, v], np.isnan(data[:, v]))
    corr = np.ma.corrcoef(ma_data_areaavg, ma_change_target_areaavg)
    #print('Corr %s %s and %s %s: %s' %(diag_var[v], var_file[v], target_var, target_file, round(corr[0, 1], 3)))
    clf.fit(data[:, v].reshape(-1, 1), np.squeeze(target.reshape(-1, 1)))
    r2_clf = r2_score(target.reshape(-1, 1),
                      clf.predict(data[:, v].reshape(-1, 1)))
    ols.fit(data[:, v].reshape(-1, 1), target.reshape(-1, 1))
    r2_ols = r2_score(target.reshape(-1, 1),
                      ols.predict(data[:, v].reshape(-1, 1)))
    # The coefficients
    #print('Coefficients: \n', clf.coef_)
    # The mean square error
    #print("Residual sum of squares: %.2f"
    #      % np.mean((clf.predict(data[:,v].reshape(-1, 1)) - target.reshape(-1, 1)) ** 2))
    #print("R^2: %.2f" %(r2_clf))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % clf.score(data[:,v].reshape(-1, 1),
    #                                         target.reshape(-1, 1)))
    # Plot outputs
    ax = plt.subplot(gs[0, v])
    ax.scatter(data[:, v], target,  color = colors[v], label = 'True')
    ax.plot(data[:, v].reshape(-1, 1),
            ols.predict(data[:, v].reshape(-1, 1)), color = 'gray',
            linewidth = 3, label = 'Predicted')
    ax.plot(data[:, v].reshape(-1, 1),
            clf.predict(data[:, v].reshape(-1, 1)), color = 'black',
            linewidth = 3, label = 'Predicted')
    ax.axvline(x = obs_areaavg[v], color = 'crimson', label = 'Obs')
    ax.axvline(x = m2_obs_areaavg[v], color = 'darkred', label = 'MERRA2')
    ax.axvline(x = eint_obs_areaavg[v], color = 'hotpink', label = 'ERAint')
    ax.text(0.3, 1.048, 'R$^2$: %.2f,' %(r2_clf),
            horizontalalignment = 'center', verticalalignment = 'center',
            transform = ax.transAxes, fontsize = 12)
    ax.text(0.7, 1.034, '{R$_{OLS}^{2}$: %.2f}' %(r2_ols),
            horizontalalignment = 'center', verticalalignment = 'center',
            transform = ax.transAxes, fontsize = 12, color = 'gray')
    #ax.text(0.25, 0.93,'Corr = %.2f' %(round(corr[0, 1], 3)), ha = 'center',
    #        va = 'center', transform = ax.transAxes)
#    if (diag_var[v] == "huss"):
#        ax.set_xlim([np.min(data[:, v]) - 0.0001,
#                     np.max(data[:, v]) + 0.0001])
#        # Rewrite the x labels
#        x_labels = ax.get_xticks()
#        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
#        plt.locator_params(axis = 'x', nbins = 5)
    if (diag_var[v] == "ef") and (var_file[v] == "TREND"):
        ax.set_xlim([- 0.004, 0.004])
        # Rewrite the x labels
        x_labels = ax.get_xticks()
        #ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
        plt.locator_params(axis = 'x', nbins = 5)

    ax.set_xlabel('%s %s %s [%s]' %(diag_var[v], var_file[v], res_name[v],
                                    data_unit[v]))
    ax.set_ylabel('$\Delta$ %s %s %s [%s]' %(target_var, target_file,
                                             res_name_target, target_unit))

    for x in range(nvar - (v + 1)):
        #print v, x
        ma_datav_areaavg = np.ma.masked_array(data[:, v],
                                              np.isnan(data[:, v]))
        ma_data_areaavg = np.ma.masked_array(data[:, x + v + 1],
                                             np.isnan(data[:, x + v + 1]))
        corr = np.ma.corrcoef(ma_data_areaavg, ma_datav_areaavg)
        #print('Corr %s %s and %s %s: %s' %(diag_var[x+v+1], var_file[x+v+1], diag_var[v], var_file[v], round(corr[0, 1], 3)))
        clf.fit(data[:, x + v + 1].reshape(-1, 1),
                np.squeeze(data[:, v].reshape(-1, 1)))
        ols.fit(data[:, x + v + 1].reshape(-1, 1),
                data[:, v].reshape(-1, 1))
        r2_clf = r2_score(data[:, v].reshape(-1, 1),
                          clf.predict(data[:, x + v + 1].reshape(-1, 1)))
        r2_ols = r2_score(data[:, v].reshape(-1, 1),
                          ols.predict(data[:, x + v + 1].reshape(-1, 1)))                
        # The coefficients
        #print('Coefficients: \n', clf.coef_)
        # The mean square error
        #print("Residual sum of squares: %.2f"
        #      % np.mean((clf.predict(data[:, x + v + 1].reshape(-1, 1)) 
        #                 - data[:, v].reshape(-1, 1)) ** 2))
        #print("R^2: %.2f" %(r2_clf))
        # Explained variance score: 1 is perfect prediction
        #print('Variance score: %.2f' % clf.score(data[:, x + v + 1].reshape(-1, 1), data[:, v].reshape(-1, 1)))

        # Plot outputs
        ax2 = plt.subplot(gs[v + 1, x + v + 1])
        p1 = ax2.scatter(data[:, x + v + 1], data[:, v],
                         color = colors[v], label = 'True')
        l1 = ax2.plot(data[:, x + v + 1].reshape(-1, 1),
                      ols.predict(data[:, x + v + 1].reshape(-1, 1)),
                      color = 'gray', linewidth = 3,
                      label = 'Predicted OLS')
        l2 = ax2.plot(data[:, x + v + 1].reshape(-1, 1),
                      clf.predict(data[:, x + v + 1].reshape(-1, 1)),
                      color = 'black', linewidth = 3,
                       label = 'Predicted %s' %(Labelclf))
        p2 = ax2.scatter(obs_areaavg[x + v + 1], obs_areaavg[v],
                         color = 'crimson', marker = "^", label = 'OBS')
        p3 = ax2.scatter(m2_obs_areaavg[x + v + 1], m2_obs_areaavg[v],
                         color = 'darkred', marker = "^",
                         label = 'MERRA2')
        p4 = ax2.scatter(eint_obs_areaavg[x + v + 1],
                         eint_obs_areaavg[v], color = 'hotpink', 
                         marker = "^", label = 'ERAint')

        #ax2.set_title(r"R$^2$: %.2f" % (r2_clf), fontsize = 12)
        ax2.text(0.3, 1.048, 'R$^2$: %.2f,' %(r2_clf),
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 transform = ax2.transAxes, fontsize = 12)
        ax2.text(0.7, 1.034, '{R$_{OLS}^{2}$: %.2f}' %(r2_ols),
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 transform = ax2.transAxes, fontsize = 12, color = 'gray')
        #ax2.text(0.25, 0.93,'Corr = %.2f' %(round(corr[0, 1], 3)),
        #         ha = 'center', va='center', transform = ax2.transAxes)
#        if (diag_var[x+v+1] == "huss"):
#            ax2.set_xlim([np.min(data[:, x + v + 1]) - 0.0001,
#                          np.max(data[:, x + v + 1]) + 0.0001])
#            # Rewrite the x labels
#            x_labels = ax2.get_xticks()
#            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
#            plt.locator_params(axis = 'x', nbins = 5)
#        if (diag_var[v] == "huss"):
#            ax2.set_ylim([np.min(data[:, v]) - 0.0001,
#                          np.max(data[:, v]) + 0.0001])
#            # Rewrite the x labels
#            y_labels = ax2.get_yticks()
#            ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
#            plt.locator_params(axis = 'y', nbins = 5)
        if (diag_var[v] == "ef") and (var_file[v] == "TREND"):
            ax2.set_xlim([- 0.004, 0.004])
            # Rewrite the x labels
            x_labels = ax2.get_xticks()
            #ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
            plt.locator_params(axis = 'x', nbins = 5)
        if (diag_var[0] == "ef") and (var_file[0] == "TREND"):
            ax2.set_ylim([- 0.004, 0.004])
            # Rewrite the x labels
            y_labels = ax2.get_yticks()
            #ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
            plt.locator_params(axis = 'y', nbins = 5)

        ax2.set_xlabel('%s %s %s [%s]' %(diag_var[x + v + 1],
                                         var_file[x + v + 1],
                                         res_name[x + v + 1],
                                         data_unit[x + v + 1]))
        ax2.set_ylabel('%s %s %s [%s]' %(diag_var[v], var_file[v], res_name[v],
                                         data_unit[v]))
        #plt.xticks(())

fig.legend((p1, p2, p3, p4), ('CMIP5', 'OBS', 'MERRA2', 'ERAint'),
           'lower left', scatterpoints = 1)
plt.tight_layout()
#plt.show()
plt.savefig('%spanel_regressions_%s_change_%s_%s_%s_%s.pdf' %(outdir, Labelclf,
                                                              target_var,
                                                              target_file,
                                                              res_name_target,
                                                              region))     

# same plot but only first two rows
fig = plt.figure(figsize = (3.5 * nvar, 7.5))
gs = gridspec.GridSpec(2, nvar)
for v in range(nvar):
    ma_data_areaavg = np.ma.masked_array(data[:, v], np.isnan(data[:, v]))
    corr = np.ma.corrcoef(ma_data_areaavg, ma_change_target_areaavg)
    #print('Corr %s %s and %s %s: %s' %(diag_var[v], var_file[v], target_var, target_file, round(corr[0, 1], 3)))
    clf.fit(data[:, v].reshape(-1, 1), np.squeeze(target.reshape(-1, 1)))
    r2_clf = r2_score(target.reshape(-1, 1),
                      clf.predict(data[:, v].reshape(-1, 1)))
    ols.fit(data[:, v].reshape(-1, 1), target.reshape(-1, 1))
    r2_ols = r2_score(target.reshape(-1, 1),
                      ols.predict(data[:, v].reshape(-1, 1)))
    # Plot outputs
    ax = plt.subplot(gs[0, v])
    ax.scatter(data[:, v], target,  color = 'gray', label = 'True')
    ax.plot(data[:, v].reshape(-1, 1),
            ols.predict(data[:, v].reshape(-1, 1)), color = 'gray',
            linewidth = 3, label = 'Predicted')
    ax.plot(data[:, v].reshape(-1, 1),
            clf.predict(data[:, v].reshape(-1, 1)), color = 'black',
            linewidth = 3, label = 'Predicted')
    ax.axvline(x = obs_areaavg[v], color = 'crimson', label = 'OBS')
    ax.axvline(x = m2_obs_areaavg[v], color = 'darkred', label = 'MERRA2')
    ax.axvline(x = eint_obs_areaavg[v], color = 'hotpink', label = 'ERAint')
    ax.text(0.3, 1.048, 'R$^2$: %.2f,' %(r2_clf),
            horizontalalignment = 'center', verticalalignment = 'center',
            transform = ax.transAxes, fontsize = 12)
    ax.text(0.7, 1.034, '{R$_{OLS}^{2}$: %.2f}' %(r2_ols),
            horizontalalignment = 'center', verticalalignment = 'center',
            transform = ax.transAxes, fontsize = 12, color = 'gray')
    #ax.text(0.25, 0.93,'Corr = %.2f' %(round(corr[0, 1], 3)), ha = 'center',
    #        va = 'center', transform = ax.transAxes)
#    if (diag_var[v] == "huss"):
#        ax.set_xlim([np.min(data[:, v]) - 0.0001,
#                     np.max(data[:, v]) + 0.0001])
#        # Rewrite the x labels
#        x_labels = ax.get_xticks()
#        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
#        plt.locator_params(axis = 'x', nbins = 5)
    if (diag_var[v] == "ef") and (var_file[v] == "TREND"):
        ax.set_xlim([- 0.004, 0.004])
        # Rewrite the x labels
        x_labels = ax.get_xticks()
        #ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
        plt.locator_params(axis = 'x', nbins = 5)
    ax.set_xlabel('%s %s %s [%s]' %(diag_var[v], var_file[v], res_name[v],
                                    data_unit[v]))
    ax.set_ylabel('$\Delta$ %s %s %s [%s]' %(target_var, target_file,
                                             res_name_target, target_unit))

    if (v != 0):
        ma_datav_areaavg = np.ma.masked_array(data[:, v],
                                              np.isnan(data[:, v]))
        ma_data_areaavg = np.ma.masked_array(data[:, 0],
                                             np.isnan(data[:, 0]))
        corr = np.ma.corrcoef(ma_data_areaavg, ma_datav_areaavg)

        clf.fit(data[:, v].reshape(-1, 1),
                np.squeeze(data[:, 0].reshape(-1, 1)))
        ols.fit(data[:, v].reshape(-1, 1),
                data[:, 0].reshape(-1, 1))
        r2_clf = r2_score(data[:, 0].reshape(-1, 1),
                          clf.predict(data[:, v].reshape(-1, 1)))
        r2_ols = r2_score(data[:, 0].reshape(-1, 1),
                          ols.predict(data[:, v].reshape(-1, 1)))
        # Plot outputs
        ax2 = plt.subplot(gs[1, v])
        p1 = ax2.scatter(data[:, v], data[:, 0],
                         color = 'gray', label = 'True')
        l1 = ax2.plot(data[:, v].reshape(-1, 1),
                      ols.predict(data[:, v].reshape(-1, 1)),
                      color = 'gray', linewidth = 3,
                      label = 'Predicted OLS')
        l2 = ax2.plot(data[:, v].reshape(-1, 1),
                      clf.predict(data[:, v].reshape(-1, 1)),
                      color = 'black', linewidth = 3,
                       label = 'Predicted %s' %(Labelclf))
        p2 = ax2.scatter(obs_areaavg[v], obs_areaavg[0],
                         color = 'crimson', marker = "^", label = 'OBS')
        p3 = ax2.scatter(m2_obs_areaavg[v], m2_obs_areaavg[0],
                         color = 'darkred', marker = "^",
                         label = 'MERRA2')
        p4 = ax2.scatter(eint_obs_areaavg[v],
                         eint_obs_areaavg[0], color = 'hotpink', 
                         marker = "^", label = 'ERAint')

        #ax2.set_title(r"R$^2$: %.2f" % (r2_clf), fontsize = 12)
        ax2.text(0.3, 1.048, 'R$^2$: %.2f,' %(r2_clf),
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 transform = ax2.transAxes, fontsize = 12)
        ax2.text(0.7, 1.034, '{R$_{OLS}^{2}$: %.2f}' %(r2_ols),
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 transform = ax2.transAxes, fontsize = 12, color = 'gray')
        #ax2.text(0.25, 0.93,'Corr = %.2f' %(round(corr[0, 1], 3)),
        #         ha = 'center', va='center', transform = ax2.transAxes)
#        if (diag_var[v] == "huss"):
#            ax2.set_xlim([np.min(data[:, v]) - 0.0001,
#                          np.max(data[:, v]) + 0.0001])
#            # Rewrite the x labels
#            x_labels = ax2.get_xticks()
#            ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
#            plt.locator_params(axis = 'x', nbins = 5)
#        if (diag_var[0] == "huss"):
#            ax2.set_ylim([np.min(data[:, 0]) - 0.0001,
#                          np.max(data[:, 0]) + 0.0001])
#            # Rewrite the x labels
#            y_labels = ax2.get_yticks()
#            ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
#            plt.locator_params(axis = 'y', nbins = 5)
        if (diag_var[v] == "ef") and (var_file[v] == "TREND"):
            ax2.set_xlim([- 0.004, 0.004])
            # Rewrite the x labels
            x_labels = ax2.get_xticks()
            #ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
            plt.locator_params(axis = 'x', nbins = 5)
        if (diag_var[0] == "ef") and (var_file[0] == "TREND"):
            ax2.set_ylim([- 0.004, 0.004])
            # Rewrite the x labels
            y_labels = ax2.get_yticks()
            #ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0e'))
            plt.locator_params(axis = 'y', nbins = 5)

        ax2.set_xlabel('%s %s %s [%s]' %(diag_var[v], var_file[v], res_name[v],
                                         data_unit[v]))
        ax2.set_ylabel('%s %s %s [%s]' %(diag_var[0], var_file[0], res_name[0],
                                         data_unit[0]))
        #plt.xticks(())

fig.legend((p1, p2, p3, p4), ('CMIP5', 'OBS', 'MERRA2', 'ERAint'),
           'lower left', scatterpoints = 1)
gs.tight_layout(fig, h_pad = 2.0)
#plt.show()
plt.savefig('%spanel_regressions_%s_change_%s_%s_%s_%s_tworows.pdf' %(
    outdir, Labelclf, target_var, target_file, res_name_target, region))

# same plot but only first two rows in portrait
fig = plt.figure(figsize = (16, 20))
if nvar <= 4:
    gs = gridspec.GridSpec(nvar, 2)
else:
    gs = gridspec.GridSpec(int(math.floor(nvar / 2.0)) + 1, 4)

mmm_target = np.ma.mean(ma_change_target_areaavg)
print 'Multi model mean: ' + str(mmm_target)
for v in range(nvar):
    ma_data_areaavg = np.ma.masked_array(data[:, v], np.isnan(data[:, v]))
    mmm_data = np.ma.mean(ma_data_areaavg)
    corr = np.ma.corrcoef(ma_data_areaavg, ma_change_target_areaavg)

    clf.fit(data[:, v].reshape(-1, 1), np.squeeze(target.reshape(-1, 1)))
    r2_clf = r2_score(target.reshape(-1, 1),
                      clf.predict(data[:, v].reshape(-1, 1)))
    ols.fit(data[:, v].reshape(-1, 1), target.reshape(-1, 1))
    r2_ols = r2_score(target.reshape(-1, 1),
                      ols.predict(data[:, v].reshape(-1, 1)))
    # Plot outputs
    if v <= 4:
        pos = v
        ax = plt.subplot(gs[pos, 0])
    else:
        try:
            pos2 = pos2 + 1
        except NameError:
            pos2 = pos - v + 2
        ax = plt.subplot(gs[pos2, 1])
    ax.scatter(data[:, v], target,  color = 'gray', label = 'True')
    ax.scatter(mmm_data, mmm_target,  color = 'dimgray',
               marker = "s", s = 50, label = 'CMIP5 mmm')
    ax.plot(data[:, v].reshape(-1, 1),
            ols.predict(data[:, v].reshape(-1, 1)), color = 'gray',
            linewidth = 3, label = 'Predicted')
    ax.plot(data[:, v].reshape(-1, 1),
            clf.predict(data[:, v].reshape(-1, 1)), color = 'black',
            linewidth = 3, label = 'Predicted')
    ax.axvline(x = obs_areaavg[v], color = 'crimson', linewidth = 2,
               label = 'OBS')
    ax.axvline(x = m2_obs_areaavg[v], color = 'darkred', linewidth = 2,
               label = 'MERRA2')
    ax.axvline(x = eint_obs_areaavg[v], color = 'hotpink', linewidth = 2,
               label = 'ERAint')
    ax.text(0.3, 1.048, 'R$^2$: %.2f,' %(r2_clf),
            horizontalalignment = 'center', verticalalignment = 'center',
            transform = ax.transAxes, fontsize = 15)
    ax.text(0.7, 1.034, '{R$_{OLS}^{2}$: %.2f}' %(r2_ols),
            horizontalalignment = 'center', verticalalignment = 'center',
            transform = ax.transAxes, fontsize = 15, color = 'gray')
    if (diag_var[v] == "ef") and (var_file[v] == "TREND"):
        ax.set_xlim([- 0.004, 0.004])
        # Rewrite the x labels
        x_labels = ax.get_xticks()
        plt.locator_params(axis = 'x', nbins = 5)
    if ((region == 'CNA') and (diag_var[v] == "tasmax") and 
        (var_file[v] == "TREND")):
        ax.set_xlim([- 0.0, 0.14])
        # Rewrite the x labels
        x_labels = ax.get_xticks()
        plt.locator_params(axis = 'x', nbins = 7)

    ax.set_xlabel('%s %s %s [%s]' %(diag_var[v], var_file[v], res_name[v],
                                    data_unit[v]))
    ax.set_ylabel('$\Delta$ %s %s %s [%s]' %(target_var, target_file,
                                             res_name_target, target_unit))

    if (v != 0):
        ma_datav_areaavg = np.ma.masked_array(data[:, v],
                                              np.isnan(data[:, v]))
        ma_data_areaavg = np.ma.masked_array(data[:, 0],
                                             np.isnan(data[:, 0]))
        mmm_data = np.ma.mean(ma_datav_areaavg)
        mmm_data0 = np.ma.mean(ma_data_areaavg)
        corr = np.ma.corrcoef(ma_data_areaavg, ma_datav_areaavg)

        clf.fit(data[:, v].reshape(-1, 1),
                np.squeeze(data[:, 0].reshape(-1, 1)))
        ols.fit(data[:, v].reshape(-1, 1),
                data[:, 0].reshape(-1, 1))
        r2_clf = r2_score(data[:, 0].reshape(-1, 1),
                          clf.predict(data[:, v].reshape(-1, 1)))
        r2_ols = r2_score(data[:, 0].reshape(-1, 1),
                          ols.predict(data[:, v].reshape(-1, 1)))
        # Plot outputs
        if v <= 4:
            ax2 = plt.subplot(gs[pos, 2])
        else:
            ax2 = plt.subplot(gs[pos2, 3])
        p1 = ax2.scatter(data[:, v], data[:, 0],
                         color = 'gray', label = 'True')
        l1 = ax2.plot(data[:, v].reshape(-1, 1),
                      ols.predict(data[:, v].reshape(-1, 1)),
                      color = 'gray', linewidth = 3,
                      label = 'Predicted OLS')
        l2 = ax2.plot(data[:, v].reshape(-1, 1),
                      clf.predict(data[:, v].reshape(-1, 1)),
                      color = 'black', linewidth = 3,
                       label = 'Predicted %s' %(Labelclf))
        p2 = ax2.scatter(obs_areaavg[v], obs_areaavg[0],
                         color = 'crimson', marker = "^", s = 50, label = 'OBS')
        p3 = ax2.scatter(m2_obs_areaavg[v], m2_obs_areaavg[0],
                         color = 'darkred', marker = "^", s = 50,
                         label = 'MERRA2')
        p4 = ax2.scatter(eint_obs_areaavg[v],
                         eint_obs_areaavg[0], color = 'hotpink', s = 50, 
                         marker = "^", label = 'ERAint')
        p5 = ax2.scatter(mmm_data, mmm_data0,  color = 'dimgray',
                         marker = "s", s = 50, label = 'CMIP5 mmm')

        ax2.text(0.3, 1.048, 'R$^2$: %.2f,' %(r2_clf),
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 transform = ax2.transAxes, fontsize = 15)
        ax2.text(0.7, 1.034, '{R$_{OLS}^{2}$: %.2f}' %(r2_ols),
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 transform = ax2.transAxes, fontsize = 15, color = 'gray')
        if (diag_var[v] == "ef") and (var_file[v] == "TREND"):
            ax2.set_xlim([- 0.004, 0.004])
            # Rewrite the x labels
            x_labels = ax2.get_xticks()
            plt.locator_params(axis = 'x', nbins = 5)
        if (diag_var[0] == "ef") and (var_file[0] == "TREND"):
            ax2.set_ylim([- 0.004, 0.004])
            # Rewrite the x labels
            y_labels = ax2.get_yticks()
            plt.locator_params(axis = 'y', nbins = 5)

        if ((region == 'CNA') and (diag_var[v] == "tasmax") and 
            (var_file[v] == "TREND")):
            ax2.set_xlim([- 0.0, 0.14])
            # Rewrite the x labels
            x_labels = ax2.get_xticks()
            plt.locator_params(axis = 'x', nbins = 7)
        if ((region == 'CNA') and (diag_var[0] == "tasmax") and 
            (var_file[0] == "TREND")):
            ax2.set_ylim([- 0.0, 0.14])
            # Rewrite the x labels
            y_labels = ax2.get_yticks()
            plt.locator_params(axis = 'y', nbins = 7)

        ax2.set_xlabel('%s %s %s [%s]' %(diag_var[v], var_file[v], res_name[v],
                                         data_unit[v]))
        ax2.set_ylabel('%s %s %s [%s]' %(diag_var[0], var_file[0], res_name[0],
                                         data_unit[0]))

fig.legend((p1, p5, p2, p3, p4), ('CMIP5', 'CMIP5 mean', 'OBS', 'MERRA2', 'ERAint'),
           'upper right', scatterpoints = 1)
gs.tight_layout(fig, h_pad = 2.0)
#plt.show()
plt.savefig('%spanel_regressions_%s_change_%s_%s_%s_%s_tworows_port.pdf' %(
    outdir, Labelclf, target_var, target_file, res_name_target, region))

# fit regressions
ols.fit(data, target.reshape(-1, 1))
clf.fit(data, np.squeeze(target.reshape(-1, 1)))

# The coefficients
print('Coefficients OLS: \n', ols.coef_)
print('Coefficients: \n', clf.coef_)
# The mean square error
print("Residual sum of squares OLS: %.2f"
      % np.mean((ols.predict(data) - target.reshape(-1, 1)) ** 2))
print("Residual sum of squares: %.2f"
      % np.mean((clf.predict(data) - target.reshape(-1, 1)) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score OLS: %.2f' % ols.score(data, target.reshape(-1, 1)))
print('Variance score: %.2f' % clf.score(data, target.reshape(-1, 1)))

#CrossValidation
cv = 5
scoresOLS = cross_val_score(ols, data, target, cv = cv)
print("Accuracy OLS: %0.2f (+/- %0.2f)" % (scoresOLS.mean(),
                                           scoresOLS.std() * 2))
scores = cross_val_score(clf, data, target, cv = cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

rfecv = RFECV(estimator = ols, step = 1)
rfecv.fit(data, target)

print("Optimal number of features OLS: %d" % rfecv.n_features_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()
plt.savefig('%sCV_nfeatures_%s_change_%s_%s_%s_%s.pdf' %(
    outdir, Labelols, target_var, target_file, res_name_target, region))  

rfecv = RFECV(estimator = clf, step = 1)
rfecv.fit(data, target)

print("Optimal number of features : %d" % rfecv.n_features_)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
#plt.show()
plt.savefig('%sCV_nfeatures_%s_change_%s_%s_%s_%s.pdf' %(
    outdir, Labelclf, target_var, target_file, res_name_target, region))
  
rfe = RFE(clf, n_features_to_select = rfecv.n_features_)
rfe.fit(data, target.reshape(-1, 1))
importances = rfe.ranking_
indices = np.argsort(importances) #[::-1]
print("Feature ranking:")
for f in range(data.shape[1]):
    print("%d. feature %d (%s %s %s %f)" % (f + 1, indices[f],
                                         diag_var[indices[f]],
                                         var_file[indices[f]],
                                         res_name[indices[f]],
                                         importances[indices[f]]))

from statsmodels.stats.outliers_influence import variance_inflation_factor
data_mean = np.nanmean(data, 0)
data_anom = data - data_mean
vif = [variance_inflation_factor(data_anom, i) for i in range(
    data_anom.shape[1])]
myRoundedList = [ round(elem, 2) for elem in vif ]
print(myRoundedList)
