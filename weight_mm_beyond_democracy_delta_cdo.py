#!/usr/bin/python
'''
File Name : weight_mm_beyond_democracy_delta_cdo.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 08-02-2017
Modified: Thu 15 Mar 2018 04:46:17 PM CET
Purpose: calculate weighted multi model mean based on
         approach from Knutti et al 2017 GRL
         target plotted is delta change between future and historical
'''
import netCDF4 as nc # to work with NetCDF files
import numpy as np
from netcdftime import utime
import datetime as dt
import math
import copy
import os # operating system interface
from os.path import expanduser
home = expanduser("~") # Get users home directory
import glob
import sys
sys.path.insert(0, home + '/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
from func_calc_wu_wq import calc_wu, calc_wq, calc_weights_approx
from func_eval_wmm_nonwmm_error_indexI import error_indexI
from draw_utils import draw
import matplotlib.pyplot as plt
import operator
from random import shuffle
import itertools
from calc_RMSE_obs_mod_3D import rmse_3D
from func_write_netcdf import func_write_netcdf
###
# Define input
###
# multiple variables possible but deltas need to be available
# and first variable determines plotting and titles in plots
diag_var = ['tasmax', 'rsds', 'pr', 'tos']
# climatology:CLIM, variability:STD, trend:TREND
var_file = ['CLIM', 'TREND', 'CLIM', 'STD']
# seasonal : 'JJA', annual = 'ANN'
res_name = ['JJA', 'JJA', 'JJA', 'JJA']
masko = ['maskT', 'maskT', 'maskT', 'maskF']
# weight of individual fields, all equal weight 1 at the moment
fields_weight = [1, 1, 1, 1]

syear_eval = [1980, 1980, 1980, 1980]
#syear_eval = [1980, 2000]
eyear_eval = [2014, 2014, 2014, 2014]
s_year1 = 1980
e_year1 = 2014
s_year2 = 2065
e_year2 = 2099

# choose region if required
region = 'NAM'
if region != None:
    area = region
else:
    area = 'GLOBAL'

obsdata = 'MERRA2' #ERAint, MERRA2, Obs
obsname = 'MERRA2' #'ERAint', 'HadGHCND (CERES, GPCP, HadISST)'
err = 'RMSE' #'perkins_SS', 'RMSE'
err_var = 'rmse' #'SS', 'rmse'

experiment = 'rcp85'
#grid = 'g025'
freq = 'mon'

indir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'
outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/Eval_Weight/%s/%s/' %(diag_var[0], area)
if (os.access(outdir, os.F_OK) == False):
    os.makedirs(outdir)

# levels for contours in plots
lev = np.arange(2.0, 7.0, 0.5)
lev_d = [-0.18, -0.14, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.14, 0.18]

## free parameter "radius of similarity" , minimum: internal variability ~0.04
## the larger the more distant models are considered similar
sigma_S2 = 0.6  #NAM: 0.6, CNEU: 0.7, std:0.6 #1.0963
## free parameter "radius of model quality"
## minimum is smallest obs. bias seen in ensemble ~+-1.8**2
## wide: mean intermodel distance in CMIP5
sigma_D2 = 0.5 #std: 0.7 #1.8353, EUR: 0.6. NAM: 0.5

###
# Read data
###
## read rmse for var_file
## first read model names for all var_file from text file
## select all models that occur for all var_files,
## or if testing select number of simulations defined in test
rmse_models = dict()
overlap = list()
for v in xrange(len(var_file)):
    rmsefile = '%s%s/%s/%s/%s_%s_%s_all_%s_%s-%s' %(indir, diag_var[v], freq,
                                                    masko[v], diag_var[v],
                                                    var_file[v], res_name[v],
                                                    experiment, syear_eval[v],
                                                    eyear_eval[v])
    if (os.access(rmsefile + '.txt', os.F_OK) == True):
        rmse_models[var_file[v]] = np.genfromtxt(rmsefile + '.txt',
                                                 delimiter = '',
                                                 dtype = None).tolist()[: - 1]
        if v > 0:
            overlap = list(set(rmse_models[var_file[v]]) & set(overlap))
        else:
            overlap = rmse_models[var_file[v]]
        try:
            overlap = overlap[0:test]
        except (NameError):
            pass

indices = dict()
rmse_d = np.ndarray((len(var_file), len(overlap), len(overlap)))
rmse_q = np.ndarray((len(var_file), len(overlap)))
for v in xrange(len(var_file)):
    rmsefile = '%s%s/%s/%s/%s_%s_%s_all_%s_%s-%s' %(indir, diag_var[v], freq,
                                                    masko[v], diag_var[v],
                                                    var_file[v], res_name[v],
                                                    experiment, syear_eval[v],
                                                    eyear_eval[v])
    ## find indices of model_names in rmse_file
    rmse_models[var_file[v]] = np.genfromtxt(rmsefile + '.txt', delimiter = '',
                                             dtype = None).tolist()[: - 1]
    for m in xrange(len(overlap)):
        indices[overlap[m]] = rmse_models[var_file[v]].index(overlap[m])
    sorted_ind = sorted(indices.items(), key = operator.itemgetter(1))
    ind = [x[1] for x in sorted_ind]
    model_names = [x[0] for x in sorted_ind]
    if (os.access('%s_%s_%s_%s.nc' %(rmsefile, area, obsdata, err),
                  os.F_OK) == True):
        print err + ' already exist, read from netcdf'
        fh = nc.Dataset('%s_%s_%s_%s.nc' %(rmsefile, area, obsdata, err),
                        mode = 'r')
        rmse_all = fh.variables[err_var]
        rmse_d[v, :, :] = rmse_all[ind, ind]
        rmse_q[v, :] = rmse_all[ - 1, ind]
        fh.close()
    else:
        print "RMSE delta matrix does not exist yet, exiting"
        sys.exit
delta_u = np.ndarray((len(var_file), len(model_names), len(model_names)))
delta_q = np.ndarray((len(var_file), len(model_names)))
for v in xrange(len(var_file)):
    ## normalize rmse by median
    med = np.nanmedian(rmse_d[v, :, :])
    delta_u[v, :, :] = rmse_d[v, :, :] / med
    delta_q[v, :] = rmse_q[v, :] / med
  ## average deltas over fields,
## taking field weight into account (all 1 at the moment)
field_w_extend_u = np.reshape(np.repeat(fields_weight,
                                        len(model_names) * len(model_names)),
                              (len(fields_weight), len(model_names),
                               len(model_names)))
delta_u = np.sqrt(np.nansum(field_w_extend_u * delta_u, axis = 0)
                  / np.nansum(fields_weight))
    
field_w_extend_q = np.reshape(np.repeat(fields_weight, len(model_names)),
                              (len(fields_weight), len(model_names)))
delta_q = np.sqrt(np.nansum(field_w_extend_q * delta_q, axis = 0)
                  / np.nansum(fields_weight))

## read obs data
print "Read %s data" %(obsdata)
if region:
    obsfile_ts = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN_%s.nc' %(
        indir, diag_var[0], freq, masko[0], diag_var[0], freq, obsdata,
        syear_eval[0], eyear_eval[0], res_name[0], region)
else:
    obsfile_ts = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN.nc' %(
        indir, diag_var[0], freq, masko[0], diag_var[0], freq, obsdata,
        syear_eval[0], eyear_eval[0], res_name[0])
fh = nc.Dataset(obsfile_ts, mode = 'r')
temp_obs_ts = fh.variables[diag_var[0]][:]
lat = fh.variables['lat'][:]
lon = fh.variables['lon'][:]

time = fh.variables['time']
cdftime = utime(time.units, calendar = time.calendar)
obsdates = cdftime.num2date(time[:])
obsyears = np.asarray([obsdates[i].year for i in xrange(len(obsdates))])
fh.close()
if isinstance(temp_obs_ts, np.ma.core.MaskedArray):
    temp_obs_ts = temp_obs_ts.filled(np.nan)

if region:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN_%s_%s.nc' %(
        indir, diag_var[0], freq, masko[0], diag_var[0], freq, obsdata,
        syear_eval[0], eyear_eval[0], res_name[0], 'CLIM', region)
else:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN_%s.nc' %(
        indir, diag_var[0], freq, masko[0], diag_var[0],freq, obsdata,
        syear_eval[0], eyear_eval[0], res_name[0], 'CLIM')
fh = nc.Dataset(obsfile, mode = 'r')
temp_obs_clim = fh.variables[diag_var[0]][:]
fh.close()
if isinstance(temp_obs_clim, np.ma.core.MaskedArray):
    maskmiss = temp_obs_clim.mask.copy()
    temp_obs_clim = temp_obs_clim.filled(np.nan)
if region:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN_%s_%s.nc' %(
        indir, diag_var[0], freq, masko[0], diag_var[0], freq, obsdata,
        syear_eval[0], eyear_eval[0], res_name[0], 'STD', region)
else:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN_%s.nc' %(
        indir, diag_var[0], freq, masko[0], diag_var[0],freq, obsdata,
        syear_eval[0], eyear_eval[0], res_name[0], 'STD')
fh = nc.Dataset(obsfile, mode = 'r')
temp_obs_std = fh.variables[diag_var[0]][:]
fh.close()
if isinstance(temp_obs_std, np.ma.core.MaskedArray):
    temp_obs_std = temp_obs_std.filled(np.nan)
if region:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN_%s_%s.nc' %(
        indir, diag_var[0], freq,masko[0], diag_var[0], freq, obsdata,
        syear_eval[0], eyear_eval[0], res_name[0], var_file[0], region)
else:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%sMEAN_%s.nc' %(
        indir, diag_var[0], freq, masko[0], diag_var[0], freq,
        obsdata, syear_eval[0], eyear_eval[0], res_name[0], var_file[0])
fh = nc.Dataset(obsfile, mode = 'r')
temp_obs_var = fh.variables[diag_var[0]][:]
lat = fh.variables['lat'][:]
lon = fh.variables['lon'][:]
fh.close()
if isinstance(temp_obs_var, np.ma.core.MaskedArray):
    temp_obs_var = temp_obs_var.filled(np.nan)

## calculate area means
rad = 4.0 * math.atan(1.0) / 180
w_lat = np.cos(lat * rad) # weight for latitude differences in area

tmp_latweight = np.ma.empty((len(obsyears), len(lon)))
ma_temp_obs_ts = np.ma.masked_array(temp_obs_ts, np.isnan(temp_obs_ts))
for ilon in xrange(len(lon)):
    tmp_latweight[:, ilon] = np.ma.average(ma_temp_obs_ts[:, :, ilon],
                                           axis = 1, weights = w_lat)
obs_ts_areaavg = np.nanmean(tmp_latweight.filled(np.nan), axis = 1)

ma_temp_obs_var = np.ma.masked_array(temp_obs_var, np.isnan(temp_obs_var))
tmp_latweight = np.ma.average(ma_temp_obs_var[0, :, :], axis = 0,
                              weights = w_lat)
obs_var_areaavg = np.nanmean(tmp_latweight.filled(np.nan), axis = 0)

### read model data
## same models as in rmse file
print "Read model data"
d_temp_mod1 = dict()
d_temp_mod = dict()
d_temp_mod_areaavg = dict()
nfiles = len(model_names)
print '%s matching files' %(str(nfiles))
for f in xrange(len(model_names)):
    model = model_names[f].split('_', 1)[0]
    ens = model_names[f].split('_', 1)[1]
    if region:
        modfile1 = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%sMEAN_%s_%s.nc' %(
            indir, diag_var[0], freq, masko[0], diag_var[0], freq, model,
            experiment, ens, s_year1, e_year1, res_name[0], var_file[0], region)
    else:
        modfile1 = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%sMEAN_%s.nc' %(
            indir, diag_var[0], freq, masko[0], diag_var[0], freq, model,
            experiment, ens, s_year1, e_year1, res_name[0], var_file[0])
    fh = nc.Dataset(modfile1, mode = 'r')
    temp_mod1 = fh.variables[diag_var[0]][:]
    fh.close()

    if region:
        modfile2 = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%sMEAN_%s_%s.nc' %(
            indir, diag_var[0], freq, masko[0], diag_var[0], freq, model,
            experiment, ens, s_year2, e_year2, res_name[0], var_file[0], region)
    else:
        modfile2 = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%sMEAN_%s.nc' %(
            indir, diag_var[0], freq, masko[0], diag_var[0], freq, model,
            experiment, ens, s_year2, e_year2, res_name[0], var_file[0])
    fh = nc.Dataset(modfile2, mode = 'r')
    tmp = fh.variables[diag_var[0]]
    unit = tmp.units
    temp_mod2 = fh.variables[diag_var[0]][:]
    fh.close()

    try:
        tmp1 = np.ma.array(temp_mod1,
                           mask = np.tile(maskmiss,
                                          (temp_mod1.shape[0], 1)))
        tmp2 = np.ma.array(temp_mod2,
                           mask = np.tile(maskmiss,
                                          (temp_mod2.shape[0], 1)))
        d_temp_mod1[model + '_' + ens] = tmp1.filled(np.nan)
        d_temp_mod[model + '_' + ens] = tmp2.filled(np.nan) - tmp1.filled(np.nan)
    except (NameError):
        d_temp_mod1[model + '_' + ens] = temp_mod1
        d_temp_mod[model + '_' + ens] = temp_mod2 - temp_mod1

    ma_temp_mod = np.ma.masked_array(d_temp_mod[model + '_' + ens], 
                                     np.isnan(d_temp_mod[model + '_' + ens]))
    tmp_latweight = np.ma.average(np.squeeze(ma_temp_mod), axis = 0,
                                  weights = w_lat)
    d_temp_mod_areaavg[model + '_' + ens] = np.nanmean(tmp_latweight.filled(
        np.nan))
    del tmp_latweight, temp_mod1, temp_mod2

## area average for models only first ensemble
d_temp_mod_ens1 = dict()
d_temp_mod_ens1_areaavg = dict()
for key, value in d_temp_mod.iteritems():
    if 'r1i1p1' in key:
        ma_temp_mod = np.ma.masked_array(value, np.isnan(value))
        tmp_latweight = np.ma.average(np.squeeze(ma_temp_mod), axis = 0,
                                      weights = w_lat)
        d_temp_mod_ens1[key] = value
        d_temp_mod_ens1_areaavg[key] = np.nanmean(tmp_latweight.filled(np.nan))
        del tmp_latweight

model_keys_ens1 = [key for key, value in sorted(
    d_temp_mod.iteritems()) if 'r1i1p1' in key]

###
# Calculate weights
###
print "Calculate weights for model dependence (u:uniqueness) and quality (q)"
wu_end = calc_wu(delta_u, model_names, sigma_S2)
wq_end = calc_wq(delta_q, model_names, sigma_D2)

###
# Calculate weighted multi-model climatologies
###
print "Calculate weighted and non-weighted model means"
model_keys = sorted(d_temp_mod.keys())       
approx_wmm = calc_weights_approx(wu_end, wq_end, model_keys, d_temp_mod,
                                 var_file = var_file[0])

approx_wmm_eval = calc_weights_approx(wu_end, wq_end, model_keys, d_temp_mod1,
                                      var_file = var_file[0])

## area average of weighted model mean
ma_approx_wmm = np.ma.masked_array(approx_wmm['approx'],
                                   np.isnan(approx_wmm['approx']))
tmp_latweight = np.ma.average(np.squeeze(ma_approx_wmm), axis = 0,
                              weights = w_lat)
approx_wmm_areaavg = np.nanmean(tmp_latweight.filled(np.nan))

## test if sum of all weights equals one
weights = approx_wmm['weights']
if (type(weights) is dict):
    dims_w = len(weights.values())
    if (dims_w == len(model_names)):
        if (np.sum(weights.values()) != 1.0):
            print "Warning: Sum of all weights does not equal 1 but %s" %(
                str(np.sum(weights.values())))
else:
    dims_w = weights.shape
    if (dims_w[0] == len(model_names)):
        if (np.sum(weights) != 1.0):
            print "Warning: Sum of all weights does not equal 1 but %s" %(
                str(np.sum(weights)))
    else:
        if (np.sum(weights[0, 0, :]) != 1.0):
            print "Warning: Sum of all weights does not equal 1 but %s" %(
                str(np.sum(weights[0, 0, :])))
# calculate unweighted mean
# first average over initial conditions ensembles per model
models = [x.split('_')[0] for x in d_temp_mod.keys()]
mult_ens = []
seen = set()
for m in models:
    if m not in seen:
        seen.add(m)
    else:
        mult_ens.append(m)
# make sure every model only once in list
list_with_mult_ens = set(mult_ens)
d_avg_ens = dict()
for key, value in d_temp_mod.iteritems():
    if key.split('_')[0] in list_with_mult_ens:
        #find other ensemble members
        ens_mem = [value2 for key2, value2 in sorted(
            d_temp_mod.iteritems()) if key.split('_')[0] in key2]
        d_avg_ens[key.split('_')[0]] = np.nanmean(ens_mem, axis = 0)
    else:
        d_avg_ens[key.split('_')[0]] = value
        
tmp2 = 0
if (var_file[0] == 'STD'):
    for key, value in d_avg_ens.iteritems():
        tmp_pow = np.power(value, 2)
        tmp2 = tmp2 + tmp_pow
    mm = np.sqrt(tmp2 / len(set(models)))
else: 
    for key, value in d_avg_ens.iteritems():
        tmp2 = tmp2 + value
    mm = tmp2 / len(set(models))
del tmp2
print mm.shape, np.nanmax(mm), np.nanmin(mm)

# calculate unweighted mean for evaluation period
# first average over initial conditions ensembles per model
models = [x.split('_')[0] for x in d_temp_mod1.keys()]
mult_ens1 = []
seen = set()
for m in models:
    if m not in seen:
        seen.add(m)
    else:
        mult_ens1.append(m)
# make sure every model only once in list
list_with_mult_ens = set(mult_ens1)
d_avg_ens1 = dict()
for key, value in d_temp_mod1.iteritems():
    if key.split('_')[0] in list_with_mult_ens:
        #find other ensemble members
        ens_mem1 = [value2 for key2, value2 in sorted(
            d_temp_mod1.iteritems()) if key.split('_')[0] in key2]
        d_avg_ens1[key.split('_')[0]] = np.nanmean(ens_mem1, axis = 0)
    else:
        d_avg_ens1[key.split('_')[0]] = value

tmp2 = 0
if (var_file[0] == 'STD'):
    for key, value in d_avg_ens1.iteritems():
        tmp_pow = np.power(value, 2)
        tmp2 = tmp2 + tmp_pow
    mm_eval = np.sqrt(tmp2 / len(set(models)))
else: 
    for key, value in d_avg_ens1.iteritems():
        tmp2 = tmp2 + value
    mm_eval = tmp2 / len(set(models))
del tmp2
print mm_eval.shape, np.nanmax(mm_eval), np.nanmin(mm_eval)

# calculate unweighted mean only using r1i1p1
#tmp4 = 0
#if (var_file[0] == 'std'):
#    for key, value in d_temp_mod_ens1.iteritems():
#        tmp_pow = np.power(value, 2)
#        tmp4 = tmp4 + tmp_pow
#    mm_ens1 = np.sqrt(tmp4 / len(model_keys_ens1))
#else:
#    for key, value in d_temp_mod_ens1.iteritems():
#        tmp4 = tmp4 + value
#    mm_ens1 = tmp4 / len(model_keys_ens1)
#print mm_ens1.shape, np.nanmax(mm_ens1), np.nanmin(mm_ens1)

#tmp5 = 0
#for key, value in d_temp_mod_ens1_ts_areaavg.iteritems():
#    tmp5 = tmp5 + value
#
#mm_ens1_ts_areaavg = tmp5 / len(model_keys_ens1)
#print mm_ens1_ts_areaavg.shape
#print np.nanmax(mm_ens1_ts_areaavg), np.nanmin(mm_ens1_ts_areaavg)

# area mean non-weighted
ma_mm = np.ma.masked_array(mm, np.isnan(mm))
if len(mm.shape) == 3:
    tmp_latweight = np.ma.empty((len(mm), len(lon)))
    for ilon in xrange(len(lon)):
        tmp_latweight[:, ilon] = np.ma.average(ma_mm[:, :, ilon],
                                               axis = 1, weights = w_lat)
        mm_areaavg = np.nanmean(tmp_latweight.filled(np.nan), axis = 1)
else:
   tmp_latweight = np.ma.average(ma_mm, axis = 0, weights = w_lat)
   mm_areaavg = np.nanmean(tmp_latweight.filled(np.nan))

## evaluate weighted multi-model mean with error index I^2
I2 = error_indexI(np.squeeze(approx_wmm_eval['approx']), np.squeeze(mm_eval),
                  np.squeeze(temp_obs_clim), np.squeeze(temp_obs_std),
                  lat, lon, res_name[0])
print 'I2 = ' + str(round(I2, 3))

## calculate RMSE for weighted and unweighted multi-model means
rmse_wmm = rmse_3D(approx_wmm_eval['approx'], temp_obs_clim, lat, lon)
rmse_mm = rmse_3D(mm_eval, temp_obs_clim, lat, lon)
diff_rmse = rmse_wmm - rmse_mm
print 'Diff RMSE WMM-MM = ' + str(round(diff_rmse, 3))

# calculate random weighted mean as baseline for I2
I2_rand = list()
diff_rand_rmse_iter = list()
for _ in itertools.repeat(None, 1000):
    rand_nrs = range(0, len(weights.values()))
    shuffle(rand_nrs)
    count = 0
    tmp6 = 0
    for key, value in d_temp_mod1.iteritems():
        rand_weight = weights.values()[rand_nrs[count]]
        tmp6 = tmp6 + rand_weight * value
        count = count + 1
    random_mm_eval = tmp6 / np.nansum(weights.values())

    I2_iter = error_indexI(np.squeeze(approx_wmm_eval['approx']),
                           np.squeeze(random_mm_eval),
                           np.squeeze(temp_obs_clim), np.squeeze(temp_obs_std),
                           lat, lon, res_name[0])
    I2_rand.append(I2_iter)
    
    rmse_rand_mm = rmse_3D(random_mm_eval, temp_obs_clim, lat, lon)
    diff_rand_rmse = rmse_wmm - rmse_rand_mm
    diff_rand_rmse_iter.append(diff_rand_rmse)
    
I2_rand_avg = np.median(I2_rand, 0)

print 'I2_rand = ' + str(round(I2_rand_avg, 3))
print 'Diff RMSE WMM-rand_MM = ' + str(round(np.median(diff_rand_rmse_iter), 3))


swu_txt = str(np.round(sigma_S2, 3))
swq_txt = str(np.round(sigma_D2, 3))

if len(mm.shape) == 3:
    plot_mm = np.mean(mm, axis = 0)
elif len(mm.shape) == 2:
    plot_mm = mm

if len(approx_wmm['approx'].shape) == 3:
    plot_wmm = np.mean(approx_wmm['approx'], axis = 0)
elif len(approx_wmm['approx'].shape) == 2:
    plot_wmm = approx_wmm['approx']

bias_wmm = np.squeeze(approx_wmm_eval['approx']) - np.squeeze(temp_obs_clim)

bias_mm = np.squeeze(mm_eval) - np.squeeze(temp_obs_clim)

## save data for further plotting and panelling
if (os.access('%s/ncdf/' %outdir, os.F_OK) == False):
    os.makedirs('%s/ncdf/' %outdir)
    print 'created directory %s/ncdf/' %outdir

outfile = '%s/ncdf/plot_delta_wmm_%s%s_%s_latlon_%s_%s_%s_%s_%s_%s.nc' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), obsdata, 
    area, err, swu_txt, swq_txt)
func_write_netcdf(outfile, plot_wmm, diag_var[0], lon, lat, var_units = unit,
                  Description = '%s %s %s weighted multi-model mean' %(
                      diag_var[0], var_file[0], res_name[0]),
                  comment = '%s' %(I2))
outfile = '%s/ncdf/plot_delta_mm_%s%s_%s_latlon_%s_%s_%s_%s_%s_%s.nc' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), obsdata,
    area, err, swu_txt, swq_txt)
func_write_netcdf(outfile, plot_mm, diag_var[0], lon, lat,
                  var_units = unit,
                  Description = '%s %s %s equal multi-model mean' %(
                      diag_var[0], var_file[0], res_name[0]))

outfile = '%s/ncdf/plot_bias_wmm_%s%s_%s_latlon_%s_%s_%s_%s_%s_%s.nc' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), obsdata, 
    area, err, swu_txt, swq_txt)
func_write_netcdf(outfile, bias_wmm, diag_var[0], lon, lat, var_units = unit,
                  Description = '%s %s %s weighted multi-model mean bias' %(
                      diag_var[0], var_file[0], res_name[0]),
                  comment = '%s' %(I2))
outfile = '%s/ncdf/plot_bias_mm_%s%s_%s_latlon_%s_%s_%s_%s_%s_%s.nc' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), obsdata,
    area, err, swu_txt, swq_txt)
func_write_netcdf(outfile, bias_mm, diag_var[0], lon, lat,
                  var_units = unit,
                  Description = '%s %s %s equal multi-model mean bias' %(
                      diag_var[0], var_file[0], res_name[0]))

###
# Plotting
###
print "Plot data"
try:
    draw(plot_wmm, lat, lon, title = "%s [%s] weighted multi-model mean %s-%s minus %s-%s" %(
        diag_var[0], unit, s_year2, e_year2, s_year1, e_year2), region = region, levels = lev)
except NameError:
    draw(plot_wmm, lat, lon, title = "%s [%s] weighted multi-model mean %s-%s minus %s-%s" %(
        diag_var[0], unit, s_year2, e_year2, s_year1, e_year2), region = region)
plt.savefig('%s%s_%s_%s_%s_%s_%s_%s_swu%s_swq%s_delta_wmm_map.pdf' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var),
    obsdata, area, err, swu_txt, swq_txt))

draw(plot_mm, lat, lon,
     title = "%s [%s] multi-model mean %s-%s minus %s-%s" %(
	diag_var[0], unit, s_year2, e_year2, s_year1, e_year2),
     region = region, levels = lev)
plt.savefig('%s%s_%s_%s_%s_%s_%s_%s_swu%s_swq%s_delta_mm_map.pdf' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), obsdata,
    area, err, swu_txt, swq_txt))

#plot maps

diff = plot_wmm - plot_mm
try:
    draw(diff, lat, lon, title = "Delta %s [%s] weighted - non-weighted, I$^2$ = %s" %
         (diag_var[0], unit, str(round(I2, 2))), levels = lev_d,
         colors = "RdBu_r", region = region)
except NameError:
    min_diff = np.nanmin(diff)
    max_diff = np.nanmax(diff)
    end = np.mean([abs(round(max_diff)), abs(round(min_diff))])
    delta = (end * 2) / 9.0
    if (end == end + delta):
        lev_d = [ - 0.09, - 0.07, - 0.05, - 0.03, - 0.01,
                  0.01, 0.03, 0.05, 0.07, 0.09]
    else:
        lev_d = np.arange( - end, end + delta, delta)       
    draw(diff, lat, lon,
         title = "Delta %s [%s] weighted - non-weighted, I$^2$ = %s" %(
             diag_var[0], unit, str(round(I2, 2))), levels = lev_d,
         colors = "RdBu_r", region = region)
plt.savefig('%sdiff_delta_%s_%s_%s_%s_%s_%s_%s_swu%s_swq%s_wmm-mm_map.pdf' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), obsdata,
    area, err, swu_txt, swq_txt))

#plot bias over evaluation period
lev_b = [-2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25]

draw(bias_wmm, lat, lon, title = "Bias %s [%s] weighted - %s, I$^2$ = %s" %
         (diag_var[0], unit, obsdata, str(round(I2, 2))), levels = lev_b,
         colors = "RdBu_r", region = region)
plt.savefig('%sbias_%s_%s_%s_%s_%s_%s_swu%s_swq%s_wmm-%s_%s--%s_map.pdf' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), area, err, 
    swu_txt, swq_txt, obsdata, s_year1, e_year2))

draw(bias_mm, lat, lon, title = "Bias %s [%s] non-weighted - %s, I$^2$ = %s" %
         (diag_var[0], unit, obsdata, str(round(I2, 2))), levels = lev_b,
         colors = "RdBu_r", region = region)
plt.savefig('%sbias_%s_%s_%s_%s_%s_%s_swu%s_swq%s_mm-%s_%s--%s_map.pdf' %(
    outdir, diag_var[0], var_file[0], res_name[0], len(diag_var), area, err, 
    swu_txt, swq_txt, obsdata, s_year1, e_year2))
