#!/usr/bin/python
'''
File Name : plot_panel_eval_weight_ts.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 26-04-2017
Modified: Wed 26 Apr 2017 11:25:44 AM CEST
Purpose: plot panel with timeseries of weighted and unweighted
         model means over NorthAmerica region


'''
# Load modules for this script

import numpy as np
import netCDF4 as nc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap, addcyclic
from pylab import *
from subprocess import call
import os # operating system interface
from netcdftime import utime
import datetime as dt
import csv
###
# Define input
###
variable = 'tasmax'
diag = 'CLIM'
region = 'NAM'
obsdata = ['MERRA2', 'ERAint', 'Obs']
obsname = ['MERRA2', 'ERAint', 'OBS']
diagnum = ['1', '2', '3', '4', '5','6']
wu = '0.6'
wq = '0.5'
path = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/Eval_Weight'
indir = '%s/%s/%s/ncdf' %(path, variable, region)
outdir = '%s/%s/%s/plots' %(path, variable, region)

if (os.access('%s' %outdir, os.F_OK) == False):
    os.makedirs('%s' %outdir)
    print 'created directory %s' %outdir

ncol = len(obsdata)
nrow = len(diagnum)
plottype= ".pdf"

degree_sign = u'\N{DEGREE SIGN}'
convert = 0 # set to zero if no conversion to different unit
unit = degree_sign + "C"
ymin = 20
ymax = 45
###
# read data
###
data_obs = {}
data_mm = {}
data_lowmm = {}
data_upmm = {}
data_wmm = {}
data_lowwmm = {}
data_upwmm = {}
d_spread_change = {}
d_mean_change = {}
# loop over diagnostics number
for d in range(0, nrow):
    dnum = str(d + 1)
    for o in range(0, ncol):
        path = '%s/%s%s_JJA_%s_%s_%s_RMSE_swu%s_swq%s_wmm_ts.nc' %(
            indir, variable, diag, dnum, obsdata[o], region, wu, wq)
        print path
        ifile = nc.Dataset(path)
        obs = ifile.variables['obs_ts_areaavg'][:]
        mm = ifile.variables['mm_ts_areaavg'][:]
        lower_mm=  ifile.variables['lower_ts_mm'][:]
        upper_mm = ifile.variables['upper_ts_mm'][:]
        wmm = ifile.variables['wmm_ts_areaavg'][:]
        lower_wmm=  ifile.variables['lower_ts_wmm'][:]
        upper_wmm = ifile.variables['upper_ts_wmm'][:]

        data_obs[obsdata[o] +'_'+ dnum] = obs - convert
        data_mm[obsdata[o] +'_'+ dnum] = mm - convert
        data_lowmm[obsdata[o] +'_'+ dnum] = lower_mm - convert
        data_upmm[obsdata[o] +'_'+ dnum] = upper_mm - convert
        data_wmm[obsdata[o] +'_'+ dnum] = wmm - convert
        data_lowwmm[obsdata[o] +'_'+ dnum] = lower_wmm - convert
        data_upwmm[obsdata[o] +'_'+ dnum] = upper_wmm - convert
        d_spread_change[obsdata[o] +'_'+ dnum] = round(np.mean(
            lower_wmm[-20 : ] - upper_wmm[-20 : ]) - np.mean(lower_mm[-20 : ] 
                                                             - upper_mm[-20 : ]), 2)
        d_mean_change[obsdata[o] +'_'+ dnum] = round(np.mean(wmm[-20 : ] - mm[-20 : ]), 2)
        if (d == 0) and (o == 0):
            obstime = ifile.variables['obstime']
            try:
                time_cal = obstime.calendar
            except AttributeError:
                time_cal = "standard"
            cdftime = utime(obstime.units, calendar = time_cal)
            obsdates = cdftime.num2date(obstime[:])
            obsyears = np.asarray([obsdates[i].year for i in xrange(len(obsdates))])

            time = ifile.variables['time']
            try:
                time_cal = time.calendar
            except AttributeError:
                time_cal = "standard"
            cdftime = utime(time.units, calendar = time_cal)
            dates = cdftime.num2date(time[:])
            years = np.asarray([dates[i].year for i in xrange(len(dates))])

        ifile.close()

###
# plotting part using gridspec
###
plt.close('all')
if (len(diagnum) == 4):
    obs_ts_areaavg = [data_obs['%s_1' %(obsdata[0])],
                      data_obs['%s_1' %(obsdata[1])],
                      data_obs['%s_1' %(obsdata[2])],
                      data_obs['%s_2' %(obsdata[0])],
                      data_obs['%s_2' %(obsdata[1])],
                      data_obs['%s_2' %(obsdata[2])],
                      data_obs['%s_3' %(obsdata[0])],
                      data_obs['%s_3' %(obsdata[1])],
                      data_obs['%s_3' %(obsdata[2])],
                      data_obs['%s_4' %(obsdata[0])],
                      data_obs['%s_4' %(obsdata[1])],
                      data_obs['%s_4' %(obsdata[2])]]
    mm_ts_areaavg = [data_mm['%s_1' %(obsdata[0])],
                     data_mm['%s_1' %(obsdata[1])],
                     data_mm['%s_1' %(obsdata[2])],
                     data_mm['%s_2' %(obsdata[0])],
                     data_mm['%s_2' %(obsdata[1])],
                     data_mm['%s_2' %(obsdata[2])],
                     data_mm['%s_3' %(obsdata[0])],
                     data_mm['%s_3' %(obsdata[1])],
                     data_mm['%s_3' %(obsdata[2])],
                     data_mm['%s_4' %(obsdata[0])],
                     data_mm['%s_4' %(obsdata[1])],
                     data_mm['%s_4' %(obsdata[2])]]
    lower_ts_mm = [data_lowmm['%s_1' %(obsdata[0])],
                   data_lowmm['%s_1' %(obsdata[1])],
                   data_lowmm['%s_1' %(obsdata[2])],
                   data_lowmm['%s_2' %(obsdata[0])],
                   data_lowmm['%s_2' %(obsdata[1])],
                   data_lowmm['%s_2' %(obsdata[2])],
                   data_lowmm['%s_3' %(obsdata[0])],
                   data_lowmm['%s_3' %(obsdata[1])],
                   data_lowmm['%s_3' %(obsdata[2])],
                   data_lowmm['%s_4' %(obsdata[0])],
                   data_lowmm['%s_4' %(obsdata[1])],
                   data_lowmm['%s_4' %(obsdata[2])]]
    upper_ts_mm = [data_upmm['%s_1' %(obsdata[0])],
                   data_upmm['%s_1' %(obsdata[1])],
                   data_upmm['%s_1' %(obsdata[2])],
                   data_upmm['%s_2' %(obsdata[0])],
                   data_upmm['%s_2' %(obsdata[1])],
                   data_upmm['%s_2' %(obsdata[2])],
                   data_upmm['%s_3' %(obsdata[0])],
                   data_upmm['%s_3' %(obsdata[1])],
                   data_upmm['%s_3' %(obsdata[2])],
                   data_upmm['%s_4' %(obsdata[0])],
                   data_upmm['%s_4' %(obsdata[1])],
                   data_upmm['%s_4' %(obsdata[2])]]
    avg_ts_wmm = [data_wmm['%s_1' %(obsdata[0])],
                  data_wmm['%s_1' %(obsdata[1])],
                  data_wmm['%s_1' %(obsdata[2])],
                  data_wmm['%s_2' %(obsdata[0])],
                  data_wmm['%s_2' %(obsdata[1])],
                  data_wmm['%s_2' %(obsdata[2])],
                  data_wmm['%s_3' %(obsdata[0])],
                  data_wmm['%s_3' %(obsdata[1])],
                  data_wmm['%s_3' %(obsdata[2])],
                  data_wmm['%s_4' %(obsdata[0])],
                  data_wmm['%s_4' %(obsdata[1])],
                  data_wmm['%s_4' %(obsdata[2])]]
    upper_ts_wmm = [data_upwmm['%s_1' %(obsdata[0])],
                    data_upwmm['%s_1' %(obsdata[1])],
                    data_upwmm['%s_1' %(obsdata[2])],
                    data_upwmm['%s_2' %(obsdata[0])],
                    data_upwmm['%s_2' %(obsdata[1])],
                    data_upwmm['%s_2' %(obsdata[2])],
                    data_upwmm['%s_3' %(obsdata[0])],
                    data_upwmm['%s_3' %(obsdata[1])],
                    data_upwmm['%s_3' %(obsdata[2])],
                    data_upwmm['%s_4' %(obsdata[0])],
                    data_upwmm['%s_4' %(obsdata[1])],
                    data_upwmm['%s_4' %(obsdata[2])]]
    lower_ts_wmm = [data_lowwmm['%s_1' %(obsdata[0])],
                    data_lowwmm['%s_1' %(obsdata[1])],
                    data_lowwmm['%s_1' %(obsdata[2])],
                    data_lowwmm['%s_2' %(obsdata[0])],
                    data_lowwmm['%s_2' %(obsdata[1])],
                    data_lowwmm['%s_2' %(obsdata[2])],
                    data_lowwmm['%s_3' %(obsdata[0])],
                    data_lowwmm['%s_3' %(obsdata[1])],
                    data_lowwmm['%s_3' %(obsdata[2])],
                    data_lowwmm['%s_4' %(obsdata[0])],
                    data_lowwmm['%s_4' %(obsdata[1])],
                    data_lowwmm['%s_4' %(obsdata[2])]]
    titles = ['(a) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[0])], unit),
              '(b) spread change = %s %s]' %(d_spread_change['%s_1' %(obsdata[1])], unit),
              '(c) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[2])], unit),
              '(d) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[0])], unit),
              '(e) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[1])], unit),
              '(f) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[2])], unit),
              '(g) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[0])], unit),
              '(h) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[1])], unit),
              '(i) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[2])], unit),
              '(j) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[0])], unit),
              '(k) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[1])], unit),
              '(l) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[2])], unit)]
    fig = plt.figure(figsize = (17, 10))
    heightratios = [0.1, 1, 1, 1, 1]
elif (len(diagnum) == 5):
    obs_ts_areaavg = [data_obs['%s_1' %(obsdata[0])],
                      data_obs['%s_1' %(obsdata[1])],
                      data_obs['%s_1' %(obsdata[2])],
                      data_obs['%s_2' %(obsdata[0])],
                      data_obs['%s_2' %(obsdata[1])],
                      data_obs['%s_2' %(obsdata[2])],
                      data_obs['%s_3' %(obsdata[0])],
                      data_obs['%s_3' %(obsdata[1])],
                      data_obs['%s_3' %(obsdata[2])],
                      data_obs['%s_4' %(obsdata[0])],
                      data_obs['%s_4' %(obsdata[1])],
                      data_obs['%s_4' %(obsdata[2])],
                      data_obs['%s_5' %(obsdata[0])],
                      data_obs['%s_5' %(obsdata[1])],
                      data_obs['%s_5' %(obsdata[2])]]    
    mm_ts_areaavg = [data_mm['%s_1' %(obsdata[0])],
                     data_mm['%s_1' %(obsdata[1])],
                     data_mm['%s_1' %(obsdata[2])],
                     data_mm['%s_2' %(obsdata[0])],
                     data_mm['%s_2' %(obsdata[1])],
                     data_mm['%s_2' %(obsdata[2])],
                     data_mm['%s_3' %(obsdata[0])],
                     data_mm['%s_3' %(obsdata[1])],
                     data_mm['%s_3' %(obsdata[2])],
                     data_mm['%s_4' %(obsdata[0])],
                     data_mm['%s_4' %(obsdata[1])],
                     data_mm['%s_4' %(obsdata[2])],
                     data_mm['%s_5' %(obsdata[0])],
                     data_mm['%s_5' %(obsdata[1])],
                     data_mm['%s_5' %(obsdata[2])]]
    lower_ts_mm = [data_lowmm['%s_1' %(obsdata[0])],
                   data_lowmm['%s_1' %(obsdata[1])],
                   data_lowmm['%s_1' %(obsdata[2])],
                   data_lowmm['%s_2' %(obsdata[0])],
                   data_lowmm['%s_2' %(obsdata[1])],
                   data_lowmm['%s_2' %(obsdata[2])],
                   data_lowmm['%s_3' %(obsdata[0])],
                   data_lowmm['%s_3' %(obsdata[1])],
                   data_lowmm['%s_3' %(obsdata[2])],
                   data_lowmm['%s_4' %(obsdata[0])],
                   data_lowmm['%s_4' %(obsdata[1])],
                   data_lowmm['%s_4' %(obsdata[2])],
                   data_lowmm['%s_5' %(obsdata[0])],
                   data_lowmm['%s_5' %(obsdata[1])],
                   data_lowmm['%s_5' %(obsdata[2])]]    
    upper_ts_mm = [data_upmm['%s_1' %(obsdata[0])],
                   data_upmm['%s_1' %(obsdata[1])],
                   data_upmm['%s_1' %(obsdata[2])],
                   data_upmm['%s_2' %(obsdata[0])],
                   data_upmm['%s_2' %(obsdata[1])],
                   data_upmm['%s_2' %(obsdata[2])],
                   data_upmm['%s_3' %(obsdata[0])],
                   data_upmm['%s_3' %(obsdata[1])],
                   data_upmm['%s_3' %(obsdata[2])],
                   data_upmm['%s_4' %(obsdata[0])],
                   data_upmm['%s_4' %(obsdata[1])],
                   data_upmm['%s_4' %(obsdata[2])],
                   data_upmm['%s_5' %(obsdata[0])],
                   data_upmm['%s_5' %(obsdata[1])],
                   data_upmm['%s_5' %(obsdata[2])]]    
    avg_ts_wmm = [data_wmm['%s_1' %(obsdata[0])],
                  data_wmm['%s_1' %(obsdata[1])],
                  data_wmm['%s_1' %(obsdata[2])],
                  data_wmm['%s_2' %(obsdata[0])],
                  data_wmm['%s_2' %(obsdata[1])],
                  data_wmm['%s_2' %(obsdata[2])],
                  data_wmm['%s_3' %(obsdata[0])],
                  data_wmm['%s_3' %(obsdata[1])],
                  data_wmm['%s_3' %(obsdata[2])],
                  data_wmm['%s_4' %(obsdata[0])],
                  data_wmm['%s_4' %(obsdata[1])],
                  data_wmm['%s_4' %(obsdata[2])],
                  data_wmm['%s_5' %(obsdata[0])],
                  data_wmm['%s_5' %(obsdata[1])],
                  data_wmm['%s_5' %(obsdata[2])]]
    upper_ts_wmm = [data_upwmm['%s_1' %(obsdata[0])],
                    data_upwmm['%s_1' %(obsdata[1])],
                    data_upwmm['%s_1' %(obsdata[2])],
                    data_upwmm['%s_2' %(obsdata[0])],
                    data_upwmm['%s_2' %(obsdata[1])],
                    data_upwmm['%s_2' %(obsdata[2])],
                    data_upwmm['%s_3' %(obsdata[0])],
                    data_upwmm['%s_3' %(obsdata[1])],
                    data_upwmm['%s_3' %(obsdata[2])],
                    data_upwmm['%s_4' %(obsdata[0])],
                    data_upwmm['%s_4' %(obsdata[1])],
                    data_upwmm['%s_4' %(obsdata[2])],
                    data_upwmm['%s_5' %(obsdata[0])],
                    data_upwmm['%s_5' %(obsdata[1])],
                    data_upwmm['%s_5' %(obsdata[2])]]    
    lower_ts_wmm = [data_lowwmm['%s_1' %(obsdata[0])],
                    data_lowwmm['%s_1' %(obsdata[1])],
                    data_lowwmm['%s_1' %(obsdata[2])],
                    data_lowwmm['%s_2' %(obsdata[0])],
                    data_lowwmm['%s_2' %(obsdata[1])],
                    data_lowwmm['%s_2' %(obsdata[2])],
                    data_lowwmm['%s_3' %(obsdata[0])],
                    data_lowwmm['%s_3' %(obsdata[1])],
                    data_lowwmm['%s_3' %(obsdata[2])],
                    data_lowwmm['%s_4' %(obsdata[0])],
                    data_lowwmm['%s_4' %(obsdata[1])],
                    data_lowwmm['%s_4' %(obsdata[2])],
                    data_lowwmm['%s_5' %(obsdata[0])],
                    data_lowwmm['%s_5' %(obsdata[1])],
                    data_lowwmm['%s_5' %(obsdata[2])]]
    titles = ['(a) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[0])], unit),
              '(b) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[1])], unit),
              '(c) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[2])], unit),
              '(d) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[0])], unit),
              '(e) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[1])], unit),
              '(f) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[2])], unit),
              '(g) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[0])], unit),
              '(h) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[1])], unit),
              '(i) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[2])], unit),
              '(j) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[0])], unit),
              '(k) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[1])], unit),
              '(l) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[2])], unit),
              '(m) spread change = %s %s' %(d_spread_change['%s_5' %(obsdata[0])], unit),
              '(n) spread change = %s %s' %(d_spread_change['%s_5' %(obsdata[1])], unit),
              '(o) spread change = %s %s' %(d_spread_change['%s_5' %(obsdata[2])], unit)]
    fig = plt.figure(figsize = (17, 12))
    heightratios = [0.1, 1, 1, 1, 1, 1]
elif (len(diagnum) == 6):
    obs_ts_areaavg = [data_obs['%s_1' %(obsdata[0])],
                      data_obs['%s_1' %(obsdata[1])],
                      data_obs['%s_1' %(obsdata[2])],
                      data_obs['%s_2' %(obsdata[0])],
                      data_obs['%s_2' %(obsdata[1])],
                      data_obs['%s_2' %(obsdata[2])],
                      data_obs['%s_3' %(obsdata[0])],
                      data_obs['%s_3' %(obsdata[1])],
                      data_obs['%s_3' %(obsdata[2])],
                      data_obs['%s_4' %(obsdata[0])],
                      data_obs['%s_4' %(obsdata[1])],
                      data_obs['%s_4' %(obsdata[2])],
                      data_obs['%s_5' %(obsdata[0])],
                      data_obs['%s_5' %(obsdata[1])],
                      data_obs['%s_5' %(obsdata[2])],
                      data_obs['%s_6' %(obsdata[0])],
                      data_obs['%s_6' %(obsdata[1])],
                      data_obs['%s_6' %(obsdata[2])]]    
    mm_ts_areaavg = [data_mm['%s_1' %(obsdata[0])],
                     data_mm['%s_1' %(obsdata[1])],
                     data_mm['%s_1' %(obsdata[2])],
                     data_mm['%s_2' %(obsdata[0])],
                     data_mm['%s_2' %(obsdata[1])],
                     data_mm['%s_2' %(obsdata[2])],
                     data_mm['%s_3' %(obsdata[0])],
                     data_mm['%s_3' %(obsdata[1])],
                     data_mm['%s_3' %(obsdata[2])],
                     data_mm['%s_4' %(obsdata[0])],
                     data_mm['%s_4' %(obsdata[1])],
                     data_mm['%s_4' %(obsdata[2])],
                     data_mm['%s_5' %(obsdata[0])],
                     data_mm['%s_5' %(obsdata[1])],
                     data_mm['%s_5' %(obsdata[2])],
                     data_mm['%s_6' %(obsdata[0])],
                     data_mm['%s_6' %(obsdata[1])],
                     data_mm['%s_6' %(obsdata[2])]]
    lower_ts_mm = [data_lowmm['%s_1' %(obsdata[0])],
                   data_lowmm['%s_1' %(obsdata[1])],
                   data_lowmm['%s_1' %(obsdata[2])],
                   data_lowmm['%s_2' %(obsdata[0])],
                   data_lowmm['%s_2' %(obsdata[1])],
                   data_lowmm['%s_2' %(obsdata[2])],
                   data_lowmm['%s_3' %(obsdata[0])],
                   data_lowmm['%s_3' %(obsdata[1])],
                   data_lowmm['%s_3' %(obsdata[2])],
                   data_lowmm['%s_4' %(obsdata[0])],
                   data_lowmm['%s_4' %(obsdata[1])],
                   data_lowmm['%s_4' %(obsdata[2])],
                   data_lowmm['%s_5' %(obsdata[0])],
                   data_lowmm['%s_5' %(obsdata[1])],
                   data_lowmm['%s_5' %(obsdata[2])],
                   data_lowmm['%s_6' %(obsdata[0])],
                   data_lowmm['%s_6' %(obsdata[1])],
                   data_lowmm['%s_6' %(obsdata[2])]]    
    upper_ts_mm = [data_upmm['%s_1' %(obsdata[0])],
                   data_upmm['%s_1' %(obsdata[1])],
                   data_upmm['%s_1' %(obsdata[2])],
                   data_upmm['%s_2' %(obsdata[0])],
                   data_upmm['%s_2' %(obsdata[1])],
                   data_upmm['%s_2' %(obsdata[2])],
                   data_upmm['%s_3' %(obsdata[0])],
                   data_upmm['%s_3' %(obsdata[1])],
                   data_upmm['%s_3' %(obsdata[2])],
                   data_upmm['%s_4' %(obsdata[0])],
                   data_upmm['%s_4' %(obsdata[1])],
                   data_upmm['%s_4' %(obsdata[2])],
                   data_upmm['%s_5' %(obsdata[0])],
                   data_upmm['%s_5' %(obsdata[1])],
                   data_upmm['%s_5' %(obsdata[2])],
                   data_upmm['%s_6' %(obsdata[0])],
                   data_upmm['%s_6' %(obsdata[1])],
                   data_upmm['%s_6' %(obsdata[2])]]    
    avg_ts_wmm = [data_wmm['%s_1' %(obsdata[0])],
                  data_wmm['%s_1' %(obsdata[1])],
                  data_wmm['%s_1' %(obsdata[2])],
                  data_wmm['%s_2' %(obsdata[0])],
                  data_wmm['%s_2' %(obsdata[1])],
                  data_wmm['%s_2' %(obsdata[2])],
                  data_wmm['%s_3' %(obsdata[0])],
                  data_wmm['%s_3' %(obsdata[1])],
                  data_wmm['%s_3' %(obsdata[2])],
                  data_wmm['%s_4' %(obsdata[0])],
                  data_wmm['%s_4' %(obsdata[1])],
                  data_wmm['%s_4' %(obsdata[2])],
                  data_wmm['%s_5' %(obsdata[0])],
                  data_wmm['%s_5' %(obsdata[1])],
                  data_wmm['%s_5' %(obsdata[2])],
                  data_wmm['%s_6' %(obsdata[0])],
                  data_wmm['%s_6' %(obsdata[1])],
                  data_wmm['%s_6' %(obsdata[2])]]
    upper_ts_wmm = [data_upwmm['%s_1' %(obsdata[0])],
                    data_upwmm['%s_1' %(obsdata[1])],
                    data_upwmm['%s_1' %(obsdata[2])],
                    data_upwmm['%s_2' %(obsdata[0])],
                    data_upwmm['%s_2' %(obsdata[1])],
                    data_upwmm['%s_2' %(obsdata[2])],
                    data_upwmm['%s_3' %(obsdata[0])],
                    data_upwmm['%s_3' %(obsdata[1])],
                    data_upwmm['%s_3' %(obsdata[2])],
                    data_upwmm['%s_4' %(obsdata[0])],
                    data_upwmm['%s_4' %(obsdata[1])],
                    data_upwmm['%s_4' %(obsdata[2])],
                    data_upwmm['%s_5' %(obsdata[0])],
                    data_upwmm['%s_5' %(obsdata[1])],
                    data_upwmm['%s_5' %(obsdata[2])],
                    data_upwmm['%s_6' %(obsdata[0])],
                    data_upwmm['%s_6' %(obsdata[1])],
                    data_upwmm['%s_6' %(obsdata[2])]]    
    lower_ts_wmm = [data_lowwmm['%s_1' %(obsdata[0])],
                    data_lowwmm['%s_1' %(obsdata[1])],
                    data_lowwmm['%s_1' %(obsdata[2])],
                    data_lowwmm['%s_2' %(obsdata[0])],
                    data_lowwmm['%s_2' %(obsdata[1])],
                    data_lowwmm['%s_2' %(obsdata[2])],
                    data_lowwmm['%s_3' %(obsdata[0])],
                    data_lowwmm['%s_3' %(obsdata[1])],
                    data_lowwmm['%s_3' %(obsdata[2])],
                    data_lowwmm['%s_4' %(obsdata[0])],
                    data_lowwmm['%s_4' %(obsdata[1])],
                    data_lowwmm['%s_4' %(obsdata[2])],
                    data_lowwmm['%s_5' %(obsdata[0])],
                    data_lowwmm['%s_5' %(obsdata[1])],
                    data_lowwmm['%s_5' %(obsdata[2])],
                    data_lowwmm['%s_6' %(obsdata[0])],
                    data_lowwmm['%s_6' %(obsdata[1])],
                    data_lowwmm['%s_6' %(obsdata[2])]]
    titles = ['(a) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[0])], unit),
              '(b) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[1])], unit),
              '(c) spread change = %s %s' %(d_spread_change['%s_1' %(obsdata[2])], unit),
              '(d) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[0])], unit),
              '(e) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[1])], unit),
              '(f) spread change = %s %s' %(d_spread_change['%s_2' %(obsdata[2])], unit),
              '(g) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[0])], unit),
              '(h) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[1])], unit),
              '(i) spread change = %s %s' %(d_spread_change['%s_3' %(obsdata[2])], unit),
              '(j) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[0])], unit),
              '(k) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[1])], unit),
              '(l) spread change = %s %s' %(d_spread_change['%s_4' %(obsdata[2])], unit),
              '(m) spread change = %s %s' %(d_spread_change['%s_5' %(obsdata[0])], unit),
              '(n) spread change = %s %s' %(d_spread_change['%s_5' %(obsdata[1])], unit),
              '(o) spread change = %s %s' %(d_spread_change['%s_5' %(obsdata[2])], unit),
              '(p) spread change = %s %s' %(d_spread_change['%s_6' %(obsdata[0])], unit),
              '(q) spread change = %s %s' %(d_spread_change['%s_6' %(obsdata[1])], unit),
              '(r) spread change = %s %s' %(d_spread_change['%s_6' %(obsdata[2])], unit)]
    fig = plt.figure(figsize = (20, 22))
    heightratios = [0.1, 1, 1, 1, 1, 1, 1]
    
## save spread and mean change to csv
with open('mean_change_%s_%s_%s_%s_%s.csv' %(variable, diag, region, wu, wq),
          'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in d_mean_change.items():
       writer.writerow([key, value])
with open('spread_change_%s_%s_%s_%s_%s.csv' %(variable, diag, region, wu, wq),
          'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in d_spread_change.items():
       writer.writerow([key, value])

## plot timeseries
plotname = '%s/panel_ts_weighted_%s_%s_%s_%s_%s_%sby%s' %(
    outdir, variable, diag, region, wu, wq, nrow, ncol)
gs = gridspec.GridSpec(nrow + 1, ncol + 1, height_ratios = heightratios,
                       width_ratios = [0.12, 1, 1, 1], hspace = 0.3,
                       wspace = 0.1, top = 0.9, right = 0.95, left = 0.0,
                       bottom = 0.05)

matplotlib.rcParams.update({'font.size': 15})

for obs in range(0, len(obsdata)):
    ax = plt.subplot(gs[0, obs + 1], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    textobs = obsname[obs]
    ax.text(0., 0, textobs)

ax = plt.subplot(gs[1, 0], frameon = False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)
text1 = 'tasmaxCLIM'
ax.text(0., 0.75, text1, rotation = 90, transform = ax.transAxes)

ax = plt.subplot(gs[2, 0], frameon = False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)
text1 = 'tasmaxCLIM,\n rsdsTREND'
ax.text(0., .7, text1, rotation = 90, transform = ax.transAxes)

ax = plt.subplot(gs[3, 0], frameon = False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)
text1 = 'tasmaxCLIM,\n rsdsTREND,\n prCLIM'
ax.text(0., 0.5, text1, rotation = 90, transform = ax.transAxes)

ax = plt.subplot(gs[4, 0], frameon = False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)
text1 = 'tasmaxCLIM,\n rsdsTREND,\n prCLIM, tosSTD'
ax.text(0., 0.6, text1, rotation = 90, transform = ax.transAxes)

if (len(diagnum) > 4):
    ax = plt.subplot(gs[5, 0], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    text1 = 'tasmaxCLIM,\n rsdsTREND,\n prCLIM, tosSTD,\n tasmaxSTD'
    ax.text(-0.15, 0.5, text1, rotation = 90,
            transform = ax.transAxes)
if (len(diagnum) > 5):
    ax = plt.subplot(gs[6, 0], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    text1 = 'tasmaxCLIM,\n rsdsTREND, prCLIM,\n tosSTD, tasmaxSTD,\n tasmaxTREND'
    ax.text(-0.15, 0.7, text1, rotation = 90,
            transform = ax.transAxes)

pltno = 0
for row in range(1, nrow + 1):
    for col in range(1, ncol + 1):
        #print pltno
        ax = plt.subplot(gs[row, col])

        plt.plot(obsyears, obs_ts_areaavg[pltno], "mediumblue",
                 label = "MERRA2/ERAint/HadGHCND",
                 linewidth = 2.0)
        plt.plot(years, mm_ts_areaavg[pltno], 'black',
                 label = "non-weighted MMM", linewidth = 2.0)
        plt.plot(years, lower_ts_mm[pltno], color = "grey")
        plt.plot(years, upper_ts_mm[pltno], color = "grey")
        plt.fill_between(years, lower_ts_mm[pltno], upper_ts_mm[pltno],
                         facecolor = 'grey',
                         alpha = 0.4)
        plt.plot(years, avg_ts_wmm[pltno], color = "crimson", linestyle = '-',
                 label = "weighted MMM", linewidth = 2.0)
        plt.plot(years, lower_ts_wmm[pltno], color = "crimson")
        plt.plot(years, upper_ts_wmm[pltno], color = "crimson")
        plt.fill_between(years, lower_ts_wmm[pltno], upper_ts_wmm[pltno],
                         facecolor = 'crimson',
                         alpha = 0.4)
        #ax.text(0.13, 0.7, 'spread change: %.1f K' %(spread_change[pltno]),
        #        ha = 'center', va = 'center', transform = ax.transAxes)
        plt.grid(False)
        plt.title(titles[pltno], ha = 'left', x = 0)
        plt.xlim([1950, 2100])
        plt.ylim([ymin, ymax])
        #ax.axes.get_xaxis().set_visible(False)
        #ax.axes.get_xaxis().set_ticklabel_visible(False)
        #ax.axes.get_ticklabels().set_ticklabel_visible(False)
        if (row == nrow):
            plt.xlabel('Year')
        if (pltno%ncol == 0):
            plt.ylabel('%s %s JJA [%s]' %(region, variable, unit))
        
        if (pltno == 0 and region == 'NAM'):
            leg = plt.legend(loc = 'upper left', fontsize = 15)  # leg defines legend -> can be modified
            leg.draw_frame(False)

        pltno = pltno + 1

gs.tight_layout(fig)
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(plotname + plottype, dpi = fig.dpi)
if (plottype == '.eps'):
    call("epstopdf " + plotname + plottype, shell = True)
call("pdfcrop " + plotname + ".pdf " + plotname + ".pdf", shell = True)

## plot spread and mean
plotname = '%s/mean_spread_change_weighted_%s_%s_%s_%s_%s_%sdiags' %(
    outdir, variable, diag, region, wu, wq, nrow)
fig = plt.figure(figsize = (8, 6), dpi = 300)
ax = fig.add_subplot(1,1,1)

plt.axhline(y = 0.0, color = 'k' ,linewidth = 1)
plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
xax_range = range(1, 7, 1)
color = ['k', 'seagreen', 'purple']
lines = []
for obs in range(0, len(obsdata)):
    plot_spread = list()
    plot_mean = list()
    for i in range(0, len(diagnum)):
        plot_spread.append(d_spread_change['%s_%s' %(obsdata[obs], diagnum[i])])
        plot_mean.append(d_mean_change['%s_%s' %(obsdata[obs], diagnum[i])])
    lines += ax.plot(xax_range, plot_mean, color = color[obs], linewidth = 3,
                      linestyle = '-', label = obsname[obs])
    lines += ax.plot(xax_range, plot_spread, color = color[obs],
                      linewidth = 3, linestyle = ':')
plt.ylim(-1.8, 1.8)

leg1 = plt.legend(loc = 'lower right', fontsize = 18, frameon = False)

from matplotlib.legend import Legend
leg2 = Legend(ax, lines[0:2], ['mean', 'spread'],
             loc = 'lower center', frameon = False)
ax.add_artist(leg2)

plt.title('Mean and spread change for region %s' %region, fontsize = 18)
plt.xlabel('Number of diagnostics included to inform weighting', fontsize = 18)
plt.ylabel('Change in mean and spread [%s]' %(unit), fontsize = 18)

#leg = plt.legend(loc = 'lower right', fontsize = 18)
#leg1.draw_frame(False)
#leg2.draw_frame(False)
plt.savefig(plotname + plottype)

