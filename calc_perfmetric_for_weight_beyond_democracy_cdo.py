#!/usr/bin/python
'''
File Name : calc_perfmetric_for_weight_beyond_democracy_cdo.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 07-02-2017
Modified: Thu 15 Mar 2018 06:07:33 PM CET
Purpose: Script calculating rmse for further use in
         weight_mm_beyond_democracy_cmip5-ng_cdo.py,
	 diagnostics precalculated using cdo in
	 calc_diag_cmip5_cdo.py and calc_diag_obs.py
'''
import netCDF4 as nc # to work with NetCDF files
import numpy as np
#from scipy import signal
import os # operating system interface
from os.path import expanduser
home = expanduser("~") # Get users home directory
import glob
import sys
sys.path.insert(0, home+'/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
from func_calc_rmse import func_calc_rmse
from calc_RMSE_obs_mod_3D import rmse_3D
from perkins_skill import perkins_skill

import datetime as dt
from netcdftime import utime
from func_write_netcdf import func_write_netcdf

###
# Define input
###
## calculate variables separately,
## weighting can then be done based on multiple variables
diag_var = ['tasmax', 'tasmax', 'tasmax']
## climatology:clim, variability:std, trend:trnd
var_file = ['CLIM', 'STD', 'TREND']
## kind is cyc: annual cycle, mon: monthly means, seas: seasonal means
var_kind = ['seas', 'seas', 'seas']
## res_name is annual (ANN) or seasonal name (DJF, MAM, JJA, SON)
res_name = ['JJA', 'JJA', 'JJA']
res_time = ['MEAN', 'MEAN', 'MEAN']
masko = ['maskT', 'maskT', 'maskT']
## cut data over region?        
region = 'NAM'
obslist = ['ERAint', 'MERRA2', 'Obs']
if region:
    area = region
else:
    area = 'GLOBAL'
experiment = 'rcp85'
freq = 'mon'

archive = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'
outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/%s/%s/' %(
    diag_var[0], freq)

syear = 1980
eyear = 2014
nyears = eyear - syear + 1

nvar = len(diag_var)

if (os.access(outdir, os.F_OK) == False):
        os.makedirs(outdir)

###read model data
print "Read model data"
##find all matching files in archive, loop over all of them
#first count matching files in folder
for v in xrange(len(diag_var)):
    model_names = list()
    if region:
        name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s_%s.nc' %(
            archive, diag_var[v], freq, masko[v], diag_var[v], freq,
            experiment, syear, eyear, res_name[v], res_time[v], var_file[v],
            region)
    else:
        name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s.nc' %(
            archive, diag_var[v], freq, masko[v], diag_var[v], freq,
            experiment, syear, eyear, res_name[v], res_time[v], var_file[v])
    data = dict()
    for filename in glob.glob(name):
        #print "Read "+filename+" data"
        fh = nc.Dataset(filename, mode = 'r')
        temp_mod = fh.variables[diag_var[v]][:] # global data, time, lat, lon
        tmp = fh.variables[diag_var[v]]
        unit = tmp.units
        lon = fh.variables['lon'][:]
        lat = fh.variables['lat'][:]
        try:
            Fill = tmp._FillValue
        except AttributeError:
            Fill = 1e+20
        fh.close()
    
        model = filename.split('_')[4]
        if model == 'ACCESS1.3':
            model = 'ACCESS1-3'
        elif model == 'FGOALS_g2':
            model = 'FGOALS-g2'
        ens = filename.split('_')[6]
        model_names.append(model + '_' + ens)

        if isinstance(temp_mod, np.ma.core.MaskedArray):
            #print type(temp_mod), temp_mod.shape
            temp_mod = temp_mod.filled(np.NaN)

        if (diag_var[v] == 'sic') and  (model == "EC-EARTH"):
            with np.errstate(invalid = 'ignore'):
                temp_mod[temp_mod < 0.0] = np.NaN
    
        data[model + '_' + ens] = temp_mod

    if obslist:
        model_names.append("Obs")
        ###read obs data
        for obsdata in obslist:
            print "Read %s data" %obsdata
            if ((obsdata == 'ERAint') and (diag_var[v] == 'huss')):
                continue
            else:
                if region:
                    name = '%s/%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
                        archive, diag_var[v], freq, masko[v], diag_var[v], freq,
                        obsdata, syear, eyear, res_name[v], res_time[v],
                        var_file[v], region)
                else:
                    name = '%s/%s/%s/%s_%s_%s_%s-%s_%s%s_%s.nc' %(
                        archive, diag_var[v], freq, masko[v], diag_var[v], freq,
                        obsdata, syear, eyear, res_name[v], res_time[v],
                        var_file[v])
            fh = nc.Dataset(name, mode = 'r')
            lon = fh.variables['lon'][:]
            lat = fh.variables['lat'][:]
            temp_obs = fh.variables[diag_var[v]][:] # global data,time,lat,lon
            fh.close()

            # mask model data where no obs
            # create mask based on obs
            if isinstance(temp_obs, np.ma.core.MaskedArray):
                for key, value in data.iteritems():
                    data[key] = np.ma.array(value, mask = temp_obs.mask)
                    data[key] = data[key].filled(np.NaN)
                temp_obs = temp_obs.filled(np.NaN)
                #print type(temp_obs), temp_obs.shape

            rmse_all = func_calc_rmse(data, temp_obs, lat, lon,
                                      model_names[:-1], var_kind[v])

            print "Save deltas to netcdf"
            fout = nc.Dataset('%s%s/%s_%s_%s_all_%s_%s-%s_%s_%s_RMSE.nc' %(
                outdir, masko[v], diag_var[v], var_file[v], res_name[v], 
                experiment, syear, eyear, area, obsdata), mode = 'w')
            fout.createDimension('x', len(model_names))
            fout.createDimension('y', len(model_names))

            xout = fout.createVariable('x', 'f8', ('x'), fill_value = 1e20)
            setattr(xout, 'Longname', 'ModelNames')
            yout = fout.createVariable('y', 'f8', ('y'), fill_value = 1e20)
            setattr(yout, 'Longname', 'ModelNames')

            rmseout = fout.createVariable('rmse', 'f8', ('x', 'y'),
                                          fill_value = 1e20)
            setattr(rmseout, 'Longname', 'Root Mean Squared Error')
            setattr(rmseout, 'units', '-')
            setattr(rmseout, 'description',
                    'RMSE between models and models and reference') 

            xout[:] = range(len(model_names))[:]
            yout[:] = range(len(model_names))[:]
            rmseout[:] = rmse_all[:]

            # Set global attributes
            setattr(fout, "author", "Ruth Lorenz @IAC, ETH Zurich, Switzerland")
            setattr(fout, "contact", "ruth.lorenz@env.ethz.ch")
            setattr(fout, "creation date", dt.datetime.today().strftime('%Y-%m-%d'))
            setattr(fout, "comment", "")

            setattr(fout, "Script", "calc_rmse_for_weight_beyond_democracy_cdo.py")
            setattr(fout, "Input files located in:", archive)
            fout.close()
    else:
        rmse_all = np.empty((len(model_names), len(model_names)), dtype = float)
        for j in xrange(len(model_names)):
            compare = model_names[j]
            for jj in xrange(len(model_names)):
                ref = model_names[jj]
                if (j == jj):
                    rmse_all[j, jj] = 0.0
                else:
                    rmse_all[j, jj] = rmse_3D(data[compare], data[ref], lat,
                                              lon, tp = var_kind[v])
        print "Save deltas to netcdf"
        fout = nc.Dataset('%s%s/%s_%s_%s_all_%s_%s-%s_%s_NoObs_RMSE.nc' %(
            outdir, masko[v], diag_var[v], var_file[v], res_name[v], 
            experiment, syear, eyear, area), mode = 'w')
        fout.createDimension('x', len(model_names))
        fout.createDimension('y', len(model_names))

        xout = fout.createVariable('x', 'f8', ('x'), fill_value = 1e20)
        setattr(xout, 'Longname', 'ModelNames')
        yout = fout.createVariable('y', 'f8', ('y'), fill_value = 1e20)
        setattr(yout, 'Longname', 'ModelNames')

        rmseout = fout.createVariable('rmse', 'f8', ('x', 'y'),
                                      fill_value = 1e20)
        setattr(rmseout, 'Longname', 'Root Mean Squared Error')
        setattr(rmseout, 'units', '-')
        setattr(rmseout, 'description',
                'RMSE between models and models and reference') 

        xout[:] = range(len(model_names))[:]
        yout[:] = range(len(model_names))[:]
        rmseout[:] = rmse_all[:]

        # Set global attributes
        setattr(fout, "author", "Ruth Lorenz @IAC, ETH Zurich, Switzerland")
        setattr(fout, "contact", "ruth.lorenz@env.ethz.ch")
        setattr(fout, "creation date", dt.datetime.today().strftime('%Y-%m-%d'))
        setattr(fout, "comment", "")

        setattr(fout, "Script", "calc_rmse_for_weight_beyond_democracy_cdo.py")
        setattr(fout, "Input files located in:", archive)
        fout.close()

    #save model names into separate text file
    with open('%s%s/%s_%s_%s_all_%s_%s-%s.txt' %(
        outdir, masko[v], diag_var[v], var_file[v], res_name[v], experiment,
        syear, eyear), "w") as text_file:
        for m in range(len(model_names)):
            text_file.write(model_names[m] + "\n")
