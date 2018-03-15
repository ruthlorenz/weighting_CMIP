#!/usr/bin/python
'''
File Name : calc_opt_sigmas_cdo.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 08-02-2017
Modified: Thu 15 Mar 2018 05:03:23 PM CET
Purpose: calculate optimal sigmas for weighting
	 multi model mean, diagnostics precalculated
         using cdo in calc_diag_cmip5_cdo.py  

'''
import netCDF4 as nc # to work with NetCDF files
import numpy as np
from netcdftime import utime
import datetime as dt
#from scipy import signal
import math
import os # operating system interface
from os.path import expanduser
home = expanduser("~") # Get users home directory
import glob
import sys
sys.path.insert(0, home + '/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
from func_calc_wu_wq import calc_wu, calc_wq, calc_weights_approx 
from func_calc_corr import calc_corr
from func_eval_wmm_nonwmm_error_indexI import error_indexI
from func_calc_inpercentile import calc_inpercentile
import matplotlib.pyplot as plt
###
# Define input
###
# multiple variables possible but deltas need to be available
# and first variable determines plotting and titles in plots
diag_var = ['tasmax', 'rsds' , 'pr', 'tos']  #NAM tasmaxCLIM
#diag_var = ['TXx', 'TXx', 'pr'] #EUR TXx
# climatology:clim, variability:std, trend:trnd
var_file = ['CLIM', 'TREND', 'CLIM', 'STD'] #NAM tasmaxCLIM
#var_file = ['CLIM', 'TREND', 'TREND'] #EUR TXx
# use ocean masked RMSE?
masko = ['maskT', 'maskT', 'maskT', 'maskF']
# kind is cyc: annual cycle, mon: monthly values, seas: seasonal
res_name = ['JJA', 'JJA', 'JJA', 'JJA']
#res_name = ['ANN', 'ANN', 'JJA'] #, 'JJA'
res_time = ['MEAN', 'MEAN', 'MEAN', 'MEAN']
#res_time = ['MAX', 'MAX', 'MEAN']
# weight of individual fields, all equal weight 1 at the moment
fields_weight = [1, 1, 1, 1] #, 1

# choose region if required
region = 'NAM'
if region != None:
    area = region
else:
    area = 'GLOBAL'
syear_eval = 1980
eyear_eval = 2014

experiment = 'rcp85'
freq = ['mon', 'mon', 'mon', 'mon']

obsdata = 'MERRA2' #ERAint or MERRA2
err = 'RMSE' #'perkins_SS', 'RMSE'
err_var = 'rmse' #'SS', 'rmse'

## method to calculate optimal sigmas, correlation or IError or inpercentile
method = 'IError' #'inpercentile'  

indir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'
outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/Eval_Weight/%s/' %(diag_var[0])
if (os.access(outdir, os.F_OK) == False):
    os.makedirs(outdir)

sigma_size = 41

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
    rmsefile = '%s%s/%s/%s/%s_%s_%s_all_%s_%s-%s' %(indir, diag_var[v], freq[v],
                                                    masko[v], diag_var[v],
                                                    var_file[v],
                                                    res_name[v], experiment,
                                                    syear_eval, eyear_eval)
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
    rmsefile = '%s%s/%s/%s/%s_%s_%s_all_%s_%s-%s' %(indir, diag_var[v], freq[v],
                                                    masko[v], diag_var[v],
                                                    var_file[v], res_name[v],
                                                    experiment, syear_eval,
                                                    eyear_eval)
    ## find indices of model_names in rmse_file
    rmse_models[var_file[v]] = np.genfromtxt(rmsefile + '.txt', delimiter = '',
                                             dtype = None).tolist()[: - 1]
    for m in xrange(len(overlap)):
        indices[overlap[m]] = rmse_models[var_file[v]].index(overlap[m])
    ind = sorted(indices.values())
    model_names = sorted(indices.keys())

    if (os.access('%s%s/%s/%s/%s_%s_%s_all_%s_%s-%s_%s_%s_%s.nc' %(
        indir, diag_var[v], freq[v], masko[v], diag_var[v], var_file[v], 
        res_name[v], experiment, syear_eval, eyear_eval, area, obsdata, err),
            os.F_OK) == True):
        print err + ' already exist, read from netcdf'
        fh = nc.Dataset('%s%s/%s/%s/%s_%s_%s_all_%s_%s-%s_%s_%s_%s.nc' %(
            indir, diag_var[v], freq[v], masko[v], diag_var[v], var_file[v],
            res_name[v], experiment, syear_eval, eyear_eval, area, obsdata,
            err), mode = 'r')
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
    obsfile_ts = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s.nc' %(indir, diag_var[0],
                                                          freq[0], masko[0],
                                                          diag_var[0], freq[0],
                                                          obsdata, syear_eval,
                                                          eyear_eval,
                                                          res_name[0],
                                                          res_time[0], region)
else:
    obsfile_ts = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s.nc' %(indir, diag_var[0],
                                                       freq[0], masko[0],
                                                       diag_var[0], freq[0],
                                                       obsdata, syear_eval,
                                                       eyear_eval, res_name[0],
                                                       res_time[0])
fh = nc.Dataset(obsfile_ts, mode = 'r')
temp_obs_ts = fh.variables[diag_var[0]][:]
lat = fh.variables['lat'][:]
lon = fh.variables['lon'][:]

time = fh.variables['time']
cdftime = utime(time.units, calendar = time.calendar)
obsdates = cdftime.num2date(time[:])
obsyears = np.asarray([obsdates[i].year for i in xrange(len(obsdates))])
fh.close()

if (method == 'IError'):
    if region:
        obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
            indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0],
            obsdata, syear_eval, eyear_eval, res_name[0], res_time[0], 
            'CLIM', region)
    else:
        obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s.nc' %(
            indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0],
            obsdata, syear_eval,eyear_eval, res_name[0], res_time[0], 'CLIM')
    fh = nc.Dataset(obsfile, mode = 'r')
    temp_obs_clim = fh.variables[diag_var[0]][:]
    fh.close()
    if region:
        obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
            indir, diag_var[0], freq[v], masko[0], diag_var[0], freq[v],
            obsdata, syear_eval, eyear_eval, res_name[0], res_time[0], 'STD',
            region)
    else:
        obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s.nc' %(
            indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0],
            obsdata, syear_eval, eyear_eval, res_name[0], res_time[0], 'STD')
    fh = nc.Dataset(obsfile, mode = 'r')
    temp_obs_std = fh.variables[diag_var[0]][:]
    fh.close()
if region:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
        indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0], obsdata,
        syear_eval, eyear_eval, res_name[0], res_time[0], var_file[0], region)
else:
    obsfile = '%s%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s.nc' %(
        indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0],
        obsdata, syear_eval, eyear_eval, res_name[0], res_time[0], var_file[0])
fh = nc.Dataset(obsfile, mode = 'r')
temp_obs_var = fh.variables[diag_var[0]][:]
lat = fh.variables['lat'][:]
lon = fh.variables['lon'][:]
fh.close()

## calculate area means
rad = 4.0 * math.atan(1.0) / 180
w_lat = np.cos(lat * rad) # weight for latitude differences in area

tmp_latweight = np.ma.empty((len(obsyears), len(lon)))
ma_temp_obs_ts = np.ma.masked_array(temp_obs_ts, np.isnan(temp_obs_ts))
for ilon in xrange(len(lon)):
    tmp_latweight[:, ilon] = np.ma.average(ma_temp_obs_ts[:, :, ilon],
                                           axis = 1, weights = w_lat)
temp_obs_ts_areaavg = np.nanmean(tmp_latweight.filled(np.nan), axis = 1)

ma_temp_obs_var = np.ma.masked_array(temp_obs_var, np.isnan(temp_obs_var))
tmp_latweight = np.ma.average(ma_temp_obs_var[0, :, :], axis = 0, weights = w_lat)
temp_obs_var_areaavg = np.nanmean(tmp_latweight.filled(np.nan), axis = 0)

if (method == 'IError'):
    ma_temp_obs_clim = np.ma.masked_array(np.squeeze(temp_obs_clim),
                                          np.isnan(temp_obs_clim))
    tmp_latweight = np.ma.average(ma_temp_obs_clim, axis = 0,
                                  weights = w_lat)
    temp_obs_clim_areaavg = np.nanmean(tmp_latweight.filled(np.nan))

    ma_temp_obs_std = np.ma.masked_array(np.squeeze(temp_obs_std),
                                         np.isnan(temp_obs_std))
    tmp_latweight = np.ma.average(ma_temp_obs_std, axis = 0, weights = w_lat)
    temp_obs_std_areaavg = np.nanmean(tmp_latweight.filled(np.nan))

### read model data
## same models as in rmse file
print "Read model data"
d_temp_mod_ts_areaavg = dict()
d_temp_mod_areaavg = dict()
nfiles = len(model_names)
print '%s matching files' %(str(nfiles))
for f in xrange(len(model_names)):
    model = model_names[f].split('_', 1)[0]
    ens = model_names[f].split('_', 1)[1]
    if region:
        modfile_ts = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s.nc' %(
            indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0], model,
            experiment, ens, syear_eval, eyear_eval, res_name[0], res_time[0],
            region)
    else:
        modfile_ts = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s.nc' %(
            indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0], model,
            experiment, ens, syear_eval, eyear_eval, res_name[0], res_time[0])

    fh = nc.Dataset(modfile_ts, mode = 'r')
    temp_mod_ts = fh.variables[diag_var[0]][:]
    lat = fh.variables['lat'][:]
    lon = fh.variables['lon'][:]
    fh.close()
    if region:
        modfile = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
            indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0], model,
            experiment, ens, syear_eval, eyear_eval, res_name[0], res_time[0],
            var_file[0], region)
    else:
        modfile = '%s%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s.nc' %(
            indir, diag_var[0], freq[0], masko[0], diag_var[0], freq[0], model,
            experiment, ens, syear_eval, eyear_eval, res_name[0], res_time[0],
            var_file[0])
    fh = nc.Dataset(modfile, mode = 'r')
    temp_mod = fh.variables[diag_var[0]][:]
    fh.close()

    ## calculate weighted area average
    ma_temp_mod_ts = np.ma.masked_array(temp_mod_ts, np.isnan(temp_mod_ts))
    tmp_latweight = np.ma.empty((len(obsyears), len(lon)))
    for ilon in xrange(len(lon)):
        tmp_latweight[:, ilon] = np.ma.average(ma_temp_mod_ts[:, :, ilon],
                                               axis = 1, weights = w_lat)
    d_temp_mod_ts_areaavg[model + '_' + ens] = np.nanmean(tmp_latweight.filled(np.nan), axis = 1)

    ma_temp_mod = np.ma.masked_array(temp_mod, np.isnan(temp_mod))
    tmp_latweight = np.ma.average(np.squeeze(ma_temp_mod), axis = 0,
                                  weights = w_lat)
    d_temp_mod_areaavg[model + '_' + ens] = np.nanmean(tmp_latweight.filled(np.nan))
    del tmp_latweight

print "Find optimal sigmas"
## get sigmas, test over range of sigmas and find either:
## ideal correlation of weighted mean with original values
## (method = 'correlation')
## largest IError metric between weighted mean and unweighted mean
## (method = 'IError')
## calculate which sigmas result in values in between 90% percentile
## (method = 'inpercentile')
tmp = np.mean(delta_u)
sigma_S2 = np.linspace(tmp - 0.9 * tmp, tmp + 0.9 * tmp, sigma_size)  # wu
sigma_D2 = np.linspace(tmp - 0.9 * tmp, tmp + 0.9 * tmp, sigma_size)  # wq

## for perfect model approach only use one ensemble per model
ind_ens1 = [i for i in range(len(model_names)) if not model_names[i].endswith('r1i1p1')]
delta_u_ens1 = np.delete(delta_u, ind_ens1, axis = 0)
model_names_ens1 = np.delete(model_names, ind_ens1, axis = 0)
w_u = calc_wu(delta_u_ens1, model_names_ens1, sigma_S2)
w_q = calc_wq(delta_u_ens1, model_names_ens1, sigma_D2)
print "wu and wq calculated for all sigmas"

temp_mod_ts_ens1 = [value for key, value in sorted(d_temp_mod_ts_areaavg.iteritems()) if 'r1i1p1' in key]
model_keys_ts = [key for key, value in sorted(d_temp_mod_ts_areaavg.iteritems()) if 'r1i1p1' in key]

ntim = temp_mod_ts_ens1[0].shape[0]

temp_mod_avg_ens1 = [value for key, value in sorted(d_temp_mod_areaavg.iteritems()) if 'r1i1p1' in key]
model_keys = [key for key, value in sorted(d_temp_mod_areaavg.iteritems()) if 'r1i1p1' in key]

print len(temp_mod_avg_ens1)
print model_keys
print "Calculate approximations and weights"
tmp_wmm_ts = calc_weights_approx(w_u, w_q, model_keys_ts, temp_mod_ts_ens1)
tmp_wmm_avg = calc_weights_approx(w_u, w_q, model_keys, temp_mod_avg_ens1)

if (method == 'correlation'):
    print "Calculate correlations to determine optimal sigmas"
    approx = tmp_wmm_ts['approx']
    corr = calc_corr(sigma_S2, sigma_D2, approx, np.array(temp_mod_ts_ens1,
                                                          dtype = float),
                     model_names_ens1)
    ## calculate indices of optimal sigmas
    ind1, ind2 = np.unravel_index(np.argmax(corr), corr.shape)
    sigma_S2_end = sigma_S2[ind1]
    sigma_D2_end = sigma_D2[ind2]
    print "Optimal sigmas are %s and %s" %(str(sigma_S2_end), str(sigma_D2_end))

    ## plot optimal sigmas in correlation sigma space
    levels = np.arange(np.min(corr), np.max(corr), 0.01)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    cax = plt.contourf(sigma_D2, sigma_S2, corr, levels, cmap = plt.cm.YlOrRd,
                       extend = 'both')
    cbar = plt.colorbar(cax, orientation = 'vertical')
    lx = plt.xlabel('$\sigma_D$', fontsize = 18)
    ly = plt.ylabel('$\sigma_S$', fontsize = 18)
    sax = plt.scatter(sigma_D2_end, sigma_S2_end, marker = 'o', c = 'k',s = 5)
    plt.savefig('%s/corr_sigmas_%s_%s_%s.pdf' %(outdir, diag_var[0],
                                                diag_var[-1], area))
elif (method == 'IError'):
    print "Calculate error Index I to determine optimal sigmas"
    tmp_mm = np.nanmean(np.array(temp_mod_ts_ens1, dtype = float), axis = 0)
    I2_sigmas = np.empty((len(sigma_S2),len(sigma_D2)))
    for s2 in xrange(len(sigma_S2)):
        for d2 in xrange(len(sigma_D2)):
            ## calculate average weight from perfect model approach
            weights = np.nanmean(tmp_wmm_ts['weights'][s2, d2, :, :], axis = 0)
            ## calculate wmm for this sigma pair
            tmp_mod = 0
            for m in xrange(len(model_names_ens1)):
                tmp_mod= tmp_mod + tmp_wmm_ts['approx'][s2, d2, m, :] * weights[m]
            tmp_wmm_clim = tmp_mod / np.nansum(weights)
            I2_sigmas[s2, d2] = error_indexI(tmp_wmm_clim, tmp_mm, 
                                             temp_obs_clim_areaavg,
                                             temp_obs_std_areaavg)
    ## calcuate indices of smallest I2 error to determine optimal sigmas
    ind1, ind2 = np.unravel_index(np.argmin(I2_sigmas), I2_sigmas.shape)
    sigma_S2_end = sigma_S2[ind1]
    sigma_D2_end = sigma_D2[ind2]
    print "Optimal sigmas are %s and %s" %(str(sigma_S2_end), str(sigma_D2_end))
elif (method == 'inpercentile'):
    print "Use percentile method to determine optimal sigmas"
    test_perc = calc_inpercentile(tmp_wmm_avg['weights'],
                                  np.array(temp_mod_avg_ens1, dtype = float))
    fout = nc.Dataset('percent_%s_%s_%s_%s_%s_%s_%s.nc' %(
        diag_var[0], var_file[0], res_name[0], len(diag_var), area,
        err, obsdata), mode = 'w')
    fout.createDimension('S2', len(sigma_S2))
    fout.createDimension('D2', len(sigma_D2))
    s2out = fout.createVariable('S2','f8',('S2'), fill_value = 1e20)
    setattr(s2out,'Longname','sigma quality')
    d2out = fout.createVariable('D2','f8',('D2'), fill_value = 1e20)
    setattr(d2out,'Longname','sigma dependency')
    testout = fout.createVariable('test','f8',('S2', 'D2'), fill_value = 1e20)
    setattr(testout,'Longname',' ')
    setattr(testout,'units','-')
    setattr(testout,'description',' ')

    s2out[:] = sigma_S2[:]
    d2out[:] = sigma_D2[:]
    testout[:] = test_perc[:]

    # Set global attributes
    setattr(fout,"author","Ruth Lorenz @IAC, ETH Zurich, Switzerland")
    setattr(fout,"contact","ruth.lorenz@env.ethz.ch")
    setattr(fout,"creation date", dt.datetime.today().strftime('%Y-%m-%d'))
    setattr(fout,"comment","")
    setattr(fout,"Script", "calc_opt_sigmas_cdo.py")
    fout.close()
