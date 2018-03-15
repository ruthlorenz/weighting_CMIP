#!/usr/bin/python
'''
File Name : calc_diag_cmip5_cdo.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 06-01-2017
Modified: Thu 15 Mar 2018 06:04:13 PM CET
Purpose: Script calculating diagnostics from cmip5 data for further use
         e.g. in Mult_Diag_Lin_Reg

'''
import netCDF4 as nc # to work with NetCDF files
import numpy as np
from cdo import *   # load cdo functionality
cdo = Cdo()
import os
from os.path import expanduser
home = expanduser("~") # Get users home directory
import glob

###
# Define input
###
varnames = ['tasmax', 'pr', 'rlus', 'rsds',
            'huss', 'psl', 'hfls']
var_kind = 'seas'    #kind is ann: annual, mon: monthly, seas: seasonal
seasons = ['JJA']      #choose data for particular month or season?
region = ['NAM', 'CNA']      #cut data over region?
syear = 1950
eyear = 2100
nyears = eyear - syear + 1

experiment = 'rcp85'
grid = 'g025'
masko = True

archive = '/net/atmos/data/cmip5-ng'
pathtrop = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'

for variable in varnames:
    workdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/%s/work/' %(
        variable)

    if masko:
        oceanmask = "%s/scripts/plot_scripts/areas_txt/seamask_%s.nc" %(home,
                                                                        grid)
        outdir = '%s/%s/mon/maskT/' %(pathtrop, variable)
    else:
        outdir = '%s/%s/mon/maskF/' %(pathtrop, variable)

    if (os.access(workdir,os.F_OK) == False):
        os.makedirs(workdir)
    if (os.access(outdir, os.F_OK) == False):
        os.makedirs(outdir)

    ### read model data
    print "Read model data"
    print 'Processing variable %s %s' %(variable, var_kind)
    ## find all matching files in archive, loop over all of them
    # first count matching files in folder:
    infile = variable + '_mon'
    name = '%s/%s/%s_*_%s_*_%s.nc' %(archive, variable, infile, experiment,
                                     grid)
    nfiles = len(glob.glob(name))
    model_names = []

    print str(nfiles) + ' matching files found'
    for filename in glob.glob(name):
        print "Processing " + filename
        fh = nc.Dataset(filename, mode = 'r')
        unit = fh.variables[variable].units
        model = fh.source_model
        if model == 'ACCESS1.3':
            model = 'ACCESS1-3'
        elif model == 'FGOALS_g2':
            model = 'FGOALS-g2'
        ens = fh.source_ensemble
        model_names.append(model + '_' + ens)
        fh.close()

        for res in seasons:
            tmpfile = '%s/%s_%s_%s_%s_%s-%s_%s' %(workdir, infile, model,
                                                  experiment, ens, syear, eyear,
                                                  res)
            outfile = '%s/%s_%s_%s_%s_%s-%s_%s' %(outdir, infile, model,
                                                  experiment, ens, syear, eyear,
                                                  res)
            if (var_kind == 'seas'):
                cdo.selseas(res, input = 
                            "-seldate,%s-01-01,%s-12-31 -sellonlatbox,-180,180,-90,90 %s"
                            %(syear, eyear, filename),
                            output = '%s.nc' %(tmpfile))
            else:
                cdo.seldate('%s-01-01,%s-12-31' %(syear, eyear),
                            input = "-sellonlatbox,-180,180,-90,90 %s"
                            %(filename), output = '%s.nc' %(tmpfile))
            if (variable == 'pr') and  (unit == 'kg m-2 s-1'):
                unitfile = cdo.mulc(24 * 60 * 60, input = '%s.nc' %(tmpfile))
                cdo.mulc(24 * 60 * 60, input = '%s.nc' %(tmpfile),
                         output = unitfile)
                newunit = "mm/day"
                cdo.chunit('"kg m-2 s-1",%s' %(newunit), input = unitfile,
                           output = '%s.nc' %(tmpfile))
            if (variable == 'huss') and  ((unit == 'kg/kg') or 
                                          (unit == 'kg kg-1') or (unit == '1')):
                unitfile = cdo.mulc(1000, options = '-b 64',
                                    input = '%s.nc' %(tmpfile))
                newunit = "g/kg"
                if (unit == 'kg/kg'):
                    cdo.chunit('"kg/kg",%s' %(newunit), input = unitfile,
                               output = '%s.nc' %(tmpfile))
                if (unit == 'kg kg-1'):
                    cdo.chunit('"kg kg-1",%s' %(newunit), input = unitfile,
                               output = '%s.nc' %(tmpfile))
                if (unit == '1'):
                    cdo.chunit('"1",%s' %(newunit), input = unitfile,
                               output = '%s.nc' %(tmpfile))
            if (((variable == 'tas') or (variable == 'tasmax') or
                (variable == 'tasmin') or (variable == 'tos')) and 
                (unit == 'K')):
                unitfile = cdo.subc(273.15, input = '%s.nc' %(tmpfile))
                newunit = "degC"
                cdo.chunit('"K",%s' %(newunit), input = unitfile,
                           output = '%s.nc' %(tmpfile))

            if masko:
                maskfile = '%s_masko.nc' %(tmpfile)
                cdo.setmissval(0, input = "-mul -eqc,1 %s %s.nc" %(oceanmask,
                                                                   tmpfile),
                               output = '%s' %(maskfile))
                os.system("mv %s %s.nc" %(maskfile, tmpfile))
            if (variable == 'tos'):
                rangefile = '%s_vrange.nc' %(tmpfile)
                cdo.setvrange('0,40', input = '%s.nc' %tmpfile,
                              output = '%s' %rangefile)
                os.system("mv %s %s.nc" %(rangefile, tmpfile))
            if (var_kind == 'seas'):
                cdo.seasmean(input = '%s.nc' %(tmpfile),
                             output = '%sMEAN.nc' %(outfile))
                cdo.yseasmean(input = '%s.nc' %(tmpfile),
                              output = '%sMEAN_CLIM.nc' %(outfile))
            elif (var_kind == 'ann'):
                cdo.yearmean(input = '%s.nc' %(tmpfile),
                             output = '%sMEAN.nc' %(outfile))
                cdo.timmean(input = '%s.nc' %(tmpfile),
                               output = '%sMEAN_CLIM.nc' %(outfile))
            else:
                cdo.ymonmean(input = '%s.nc' %(tmpfile),
                              output = '%sMEAN_CLIM.nc' %(outfile))
            cdo.detrend(input = '%sMEAN.nc' %(outfile),
                        output = '%sMEAN_DETREND.nc' %(tmpfile))
            cdo.regres(input = '%sMEAN.nc' %(outfile),
                       output = '%sMEAN_TREND.nc' %(outfile))
            cdo.timstd(input = '%sMEAN_DETREND.nc' %(tmpfile),
                       output = '%sMEAN_STD.nc' %(outfile))

            if region:
                # loop over regions
                for reg in region:
                    print 'Region is %s' %(reg)
                    area = reg
                    mask = np.loadtxt(
                        '%s/scripts/plot_scripts/areas_txt/%s.txt' %(
                            home, reg))

                    lonmax = np.max(mask[:, 0])
                    lonmin = np.min(mask[:, 0])
                    latmax = np.max(mask[:, 1]) 
                    latmin = np.min(mask[:, 1])
                    cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,
                                     input = '%sMEAN_CLIM.nc' %(outfile),
                                     output = '%sMEAN_CLIM_%s.nc' %(outfile,
                                                                    reg))
                    cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,
                                     input = '%sMEAN_STD.nc' %(outfile),
                                     output = '%sMEAN_STD_%s.nc' %(outfile,
                                                                   reg))
                    cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,
                                     input = '%sMEAN_TREND.nc' %(outfile),
                                     output = '%sMEAN_TREND_%s.nc' %(outfile,
                                                                     reg))
                    cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,
                                     input = '%sMEAN.nc' %(outfile),
                                     output = '%sMEAN_%s.nc' %(outfile, reg))

    #save model names into separate text file
    with open('%s%s_%s_all_%s_%s-%s.txt' %(outdir, variable, 
                                           res, experiment,
                                           syear, eyear), "w") as text_file:
        for m in range(len(model_names)):
            text_file.write(model_names[m] + "\n")

    # clean up workdir
    filelist = glob.glob(workdir + '*')
    for f in filelist:
        os.remove(f)
