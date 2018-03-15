#!/usr/bin/python
'''
File Name : calc_diag_obs_cdo.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 18-01-2017
Modified: Wed 22 Nov 2017 06:34:49 PM CET
Purpose: calculate diagnostic from obs and save for further use

'''
import netCDF4 as nc # to work with NetCDF files
import numpy as np
from cdo import *   # load cdo functionality
cdo = Cdo()
from os.path import expanduser
home = expanduser("~") # Get users home directory
import glob

###
# Define input
###
#varnames = ['tas', 'tasmax', 'tasmin', 'pr', 'rlus', 'rsds', 'rnet',
#            'huss', 'psl', 'hfls']
varnames = ['hurs']
var_kind = 'seas'    #kind is ann: annual, mon: monthly, seas: seasonal
seasons = ['JJA', 'MAM', 'SON', 'DJF']           # choose data for particular month or season?
region = ['EUR', 'CNEU']           # cut data over region?
obs = ['Obs']  # 'ERAint', 'MERRA2', 'Obs'

syear = 1980
eyear = 2014
nyears = eyear - syear + 1
grid = 'g025'
masko = False

pathtrop = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'

for variable in varnames:
    print 'Processing variable %s %s' %(variable, var_kind)
    workdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/%s/work/' %(
        variable)

    if masko:
        outdir = '%s/%s/mon/maskT/' %(pathtrop, variable)
        oceanmask = "%s/scripts/plot_scripts/areas_txt/seamask_%s.nc" %(
            home, grid)
    else:
        outdir = '%s/%s/mon/maskF/' %(pathtrop, variable)

    if (os.access(workdir,os.F_OK) == False):
        os.makedirs(workdir)
    if (os.access(outdir, os.F_OK) == False):
            os.makedirs(outdir)

    #read obs data
    for dataset in obs:
        if dataset == 'ERAint':
            print "Read ERAint data"
            path = '/net/tropo/climphys/rlorenz/Datasets/ERAint/ERAint_mon'
            if (variable == 'tas'):
                varname = 'T2M'
                filename = '%s/%s_monthly_ERAint_197901-201601_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'pr'):
                varname = 'tp'
                filename = '%s/%s_0-12_monthly_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'sic'):
                varname = 'ci'
                filename = '%s/%s_monthly_ERAint_197901-201601_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'hfls'):
                varname = 'slhf'
                filename = '%s/%s_monthly_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'tasmax'):
                varname = 'mx2t'
                filename = '%s/%s_monmean_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'tasmin'):
                varname = 'mn2t'
                filename = '%s/%s_monmean_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'hus'):
                varname = 'q'
                filename = '%s/%s_monthly_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'rsds'):
                varname = 'ssrd'
                filename = '%s/%s_monthly_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'rsus'):
                varname = 'ssru'
                filename = '%s/%s_monthly_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'rlus'):
                varname = 'stru'
                filename = '%s/%s_monthly_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'psl'):
                varname = 'SLP'
                filename = '%s/%s_monthly_ERAint_197901-201412_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'rnet'):
                varname = 'rnet'
                filename = '%s/%s_monthly_ERAint_197901-201512_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'tos'):
                varname = 'sst'
                filename = '%s/%s_monthly_ERAint_197901-201612_%s.nc' %(
                    path, varname, grid)
            elif (variable == 'clt'):
                varname = 'tcc'
                filename = '%s/cld_monthly_ERAint_197901-201612_%s.nc' %(
                    path, grid)
            else:
                print'Variable not available in ERAint, continue'
                continue
        elif dataset == 'MERRA2':
            print "Read MERRA2 data"
            path = '/net/tropo/climphys/rlorenz/Datasets/MERRA2/monthly'
            if (variable == 'tas'):
                varname = 'T2MMEAN'
                lev = 'slv'
            elif (variable == 'tasmax'):
                varname = 'T2MMAX'
                lev = 'slv'
            elif (variable == 'tasmin'):
                varname = 'T2MMIN'
                lev = 'slv'
            elif (variable == 'pr'):
                varname = 'PRECTOTCORR'
                lev = 'flx'
                filename = '%s/%s.tavgmon_2d_%s_Nx.%s.1980-2015_remapcon2_%s.nc' %(
                    path, dataset, lev, varname, grid)
            elif (variable == 'hfls'):
                varname = 'EFLUX'
                lev = 'flx'
            elif (variable == 'psl'):
                varname = 'SLP'
                lev = 'slv'
            elif (variable == 'rsds'):
                varname = 'SWGDN'
                lev = 'rad'
            elif (variable == 'rlus'):
                varname = 'LWGEM'
                lev = 'rad'
            elif (variable == 'huss'):
                varname = 'QV2M'
                lev = 'slv'
            elif (variable == 'rnet'):
                varname = 'RNET'
                lev = 'rad'
            elif (variable == 'clt'):
                varname = 'CLDTOT'
                lev = 'rad' 
            elif (variable == 'tos'):
                varname = 'TSKINWTR'
                lev = 'ocn' 
                filename = '%s/MERRA2.tavgM_2d_ocn_Nx.TSKINWTR.1980-2016_remapbil_%s.nc' %(path, grid)
            else:
                print'Variable not available in MERRA2, continue'
                continue
            if (variable != 'tos') and (variable != 'pr'):
                filename = '%s/%s.tavgmon_2d_%s_Nx.%s.1980-2015_remapbil_%s.nc' %(
                    path, dataset, lev, varname, grid)
        elif (dataset == 'Obs'):
            print "Read Obs data"
            path = '/net/tropo/climphys/rlorenz/Datasets/'
            if (variable == 'tas'):
                if ((region[0] == 'EUR') or (region[0] == 'CNEU') or
                    (region[0] == 'SOEU')):
                    filename = '%s/E-OBS/tg_mon_1950-2017_reg_v16.0_%s.nc' %(
                        path, grid)
                    varname = 'tg'
                else:
                    filename = '%s/BerkeleyEarth/Complete_TAVG_abs_LatLong_011750-122015_%s.nc' %(path, grid)
                    varname = 'temperature'
            elif (variable == 'tasmax'):
                if ((region[0] == 'EUR') or (region[0] == 'CNEU') or
                    (region[0] == 'SOEU')):
                    filename = '%s/E-OBS/tx_mon_1950-2017_reg_v16.0_%s' %(path,
                                                                          grid)
                    varname = 'tx'
                else:
                    filename = '%s/HadGHCND/monthly/HadGHCND_TXTN_avg_monthly_acts_195001-201412_remapbil_%s.nc' %(path, grid)
                    varname = 'tmax'
            elif (variable == 'tasmin'):
                if ((region[0] == 'EUR') or (region[0] == 'CNEU') or
                    (region[0] == 'SOEU')):
                    filename = '%s/E-OBS/tn_mon_1950-2017_reg_v16.0_%s' %(
                        path, grid)
                    varname = 'tn'
                else:
                    filename = '%s/HadGHCND/monthly/HadGHCND_TXTN_avg_monthly_acts_195001-201412_remapbil_%s.nc' %(path, grid)
                    varname = 'tmin'
            elif (variable == 'pr'):
                if ((region[0] == 'EUR') or (region[0] == 'CNEU') or
                    (region[0] == 'SOEU')):
                    filename = '%s/E-OBS/rr_mon_1950-2017_reg_v16.0_%s.nc' %(
                        path, grid)
                    varname = 'rr'
                else:
                    filename = '%s/GPCP/precip.mon.mean.197901-201607.v2.3_g025.nc' %(path)
                    varname = 'precip' 
            elif (variable == 'hfls'):
                filename = '%s/GLEAM/monthly/GLEAM_E_monthly_v3.0a_1980-2014_%s.nc' %(path, grid)
                varname = 'E'
            elif (variable == 'psl'):
                filename = '%s/HadSLP2/slp.mnmean.real.185001-201701_%s.nc'%(
                    path, grid)
                varname = 'slp'
            elif (variable == 'huss'):
                filename = '%s/HadISDH/HadISDH.landq.3.0.0.2016p_FLATgridIDPHA5by5_anoms7605_JAN2017_cf_remapcon2_%s.nc' %(path, grid)
                varname = 'q_abs'
            elif (variable == 'hurs'):
                filename = '%s/HadISDH/hurs-land_HadISDH_HadOBS_v3-0-0-2016p_19730101-20161231_remapbil_%s.nc' %(path, grid)
                varname = 'hurs'
            elif (variable == 'ef'):
                print "Warning: no obs data for this variable, exiting"
                sys.exit
            elif (variable == 'rsus'):
                filename = '%s/CERES/CERES_EBAF-sfc_sw_up_all_mon_Ed2.8_Subset_200003-201511_%s.nc' %(path, grid)
                varname = 'sfc_sw_up_all_mon'
                syear = 2000
            elif (variable == 'rsds'):
                filename = '%s/CERES/CERES_EBAF-sfc_sw_down_all_mon_Ed2.8_Subset_200003-201511_%s.nc' %(path, grid)
                varname = 'sfc_sw_down_all_mon'
                syear = 2000
            elif (variable == 'rlus'):
                filename = '%s/CERES/CERES_EBAF-sfc_lw_up_all_mon_Ed2.8_Subset_200003-201511_%s.nc' %(path, grid)
                varname = 'sfc_lw_up_all_mon'
                syear = 2000
            elif (variable == 'rnet'):
                filename = '%s/CERES/CERES_EBAF-sfc_net_tot_all_mon_Ed2.8_Subset_200003-201511_%s.nc' %(path, grid)
                varname = 'sfc_net_tot_all_mon'
                syear = 2000
            elif (variable == 'clt'):
                filename = '%s/CERES/CERES_SYN1deg-Month_Terra-Aqua-MODIS_Ed4A_Subset_200003-201708_%s.nc' %(path, grid)
                varname = 'cldarea_total_mon'
                syear = 2000
            elif (variable == 'tos'):
                filename = '%s/HadISST/HadISST_sst_187001-201705_%s.nc' %(
                    path, grid)
                varname = 'sst'
            else:
                print'Variable not available in obsdata, continue'
                continue

        fh = nc.Dataset(filename, mode = 'r')
        try:
            unit = fh.variables[varname].units
        except AttributeError:
            unit = "unknown"
        fh.close()

        infile = '%s_mon' %(variable)

        for res in seasons:
            tmpfile = '%s/%s_%s_%s-%s_%s' %(
                workdir, infile, dataset, syear, eyear, res)

            outfile = '%s/%s_%s_%s-%s_%s' %(
                outdir, infile, dataset, syear, eyear, res)

            datefile = cdo.seldate('%s-01-01,%s-12-31' %(syear, eyear),
                                   input = '%s' %(filename))
            boxfile = cdo.sellonlatbox(-180,180,-90,90, input = datefile)
            varfile = cdo.selname(varname, input = boxfile)
            namefile = cdo.chname(varname,variable, input = varfile)
            cdo.selseas(res, input = namefile, output = '%s.nc' %(tmpfile))
            #try:
            os.system("ncatted -h -O -a valid_min,,d,, %s.nc" %(tmpfile))
            os.system("ncatted -h -O -a valid_max,,d,, %s.nc" %(tmpfile))
            #except:

            if (variable == 'pr') and  (unit == 'kg m-2 s-1'):
                unitfile = cdo.mulc(24 * 60 * 60, input = '%s.nc' %(tmpfile))
                newunit = "mm/day"
                cdo.chunit('"kg m-2 s-1",%s' %(newunit), input = unitfile,
                           output = '%s.nc' %(tmpfile))
            if (variable == 'pr') and  (unit == 'm'):
                unitfile = cdo.mulc(1000, input = '-divdpm %s.nc' %(tmpfile))
                newunit = "mm/day"
                cdo.chunit('m,%s' %(newunit), input = unitfile,
                           output = '%s.nc' %(tmpfile))
            if (variable == 'psl') and  (unit == 'mb'):
                unitfile = cdo.mulc(100, options = '-b 64',
                                    input = '%s.nc' %(tmpfile))
                newunit = "Pa"
                cdo.chunit('"mb",%s' %(newunit), input = unitfile,
                           output = '%s.nc' %(tmpfile))
            if (variable == 'psl') and (dataset == 'ERAint'):
                unitfile = cdo.mulc(100, input = '%s.nc' %(tmpfile))
                newunit = "Pa"
                cdo.setunit('%s' %(newunit), input = unitfile,
                            output = '%s.nc' %(tmpfile))
            if (variable == 'huss') and  ((unit == 'kg/kg') or 
                                          (unit == 'kg kg-1')):
                unitfile = cdo.mulc(1000, options = '-b 64',
                                    input = '%s.nc' %(tmpfile))
                newunit = "g/kg"
                if (unit == 'kg/kg'):
                    cdo.chunit('"kg/kg",%s' %(newunit), input = unitfile,
                               output = '%s.nc' %(tmpfile))
                if (unit == 'kg kg-1'):
                    cdo.chunit('"kg kg-1",%s' %(newunit), input = unitfile,
                               output = '%s.nc' %(tmpfile))
            if ((variable == 'tasmax') or (variable == 'tas') or
                (variable == 'tasmin')) and (unit == 'K'):
                unitfile = cdo.subc(273.15, options = '-b 64',
                                    input = '%s.nc' %(tmpfile))
                cdo.chunit('"K","degC"', input = unitfile,
                           output = '%s.nc' %(tmpfile))
            if (variable == 'hfls') and (dataset == 'ERAint'):
                chsignfile = cdo.mulc(-1., input = '%s.nc' %(tmpfile))
                os.system("mv %s %s.nc" %(chsignfile, tmpfile))
            if (variable == 'hfls') and ((unit == 'mm/d') or
                                         (unit == 'mm/day')):
                unitfile = cdo.mulc(2.5 * 10 ** 6 / 86400,
                                    input =  '%s.nc' %(tmpfile))
                if (unit == 'mm/d'):
                    cdo.chunit('"mm/d","W m-2"', input = unitfile,
                               output = '%s.nc' %(tmpfile))
                elif (unit == 'mm/day'):
                    cdo.chunit('"mm/day","W m-2"', input = unitfile,
                               output = '%s.nc' %(tmpfile))
            if ((variable == 'tos') and (unit == 'K')):
                unitfile = cdo.subc(273.15, options = '-b 64',
                                    input = '%s.nc' %(tmpfile))
                cdo.chunit('"K","degC"', input = unitfile,
                           output = '%s.nc' %(tmpfile))
            if ((variable == 'clt') and ((unit == '1') or (unit == '(0 - 1)'))):
                unitfile = cdo.mulc(100, options = '-b 64',
                                    input = '%s.nc' %(tmpfile))
                if (unit == '1'):
                    cdo.chunit('"1","%"', input = unitfile,
                               output = '%s.nc' %(tmpfile))
                elif (unit == '(0 - 1)'):
                    cdo.chunit('"(0 - 1)","%"', input = unitfile,
                               output = '%s.nc' %(tmpfile))
            if masko:
                maskfile = '%s_masko.nc' %(tmpfile)
                cdo.setmissval(0, input = "-mul -eqc,1 %s %s.nc" %(
                    oceanmask, tmpfile),
                               output = '%s' %(maskfile))
                os.system("mv %s %s.nc" %(maskfile, tmpfile))

            cdo.seasmean(input = '%s.nc' %(tmpfile),
                         output = '%sMEAN.nc' %(outfile))
            cdo.yseasmean(input = '%s.nc' %(tmpfile),
                          output = '%sMEAN_CLIM.nc' %(outfile))
            cdo.detrend(input = '%sMEAN.nc' %(outfile),
                        output = '%sMEAN_DETREND.nc' %(tmpfile),
                        options =  '-b 64')
            cdo.regres(input = '%sMEAN.nc' %(outfile),
                       output = '%sMEAN_TREND.nc' %(outfile),
                       options =  '-b 64')
            cdo.timstd(input = '%sMEAN_DETREND.nc' %(tmpfile),
                       output = '%sMEAN_STD.nc' %(outfile))

            if region != None:
                # loop over regions
                for reg in region:
                    print 'Region is %s' %(reg)
                    area = reg
                    mask = np.loadtxt(
                        '/home/rlorenz/scripts/plot_scripts/areas_txt/%s.txt' %(
                            reg))

                    lonmax = np.max(mask[:, 0])
                    lonmin = np.min(mask[:, 0])
                    latmax = np.max(mask[:, 1]) 
                    latmin = np.min(mask[:, 1])
                    cdo.sellonlatbox(lonmin,lonmax,latmin,latmax,
                                     input = '%sMEAN_CLIM.nc' %(outfile),
                                     output = '%sMEAN_CLIM_%s.nc' %(
                                         outfile, reg))
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

    # clean up workdir
    filelist = glob.glob(workdir + '*')
    for f in filelist:
        os.remove(f)
