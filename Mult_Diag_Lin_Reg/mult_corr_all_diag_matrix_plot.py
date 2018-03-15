#!/usr/bin/python
'''
File Name : mult_corr_all_diag_matrix_plot.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 14-02-2017
Modified: Tue 24 Oct 2017 04:26:54 PM CEST
Purpose: calculate and plot cross correlations of all diagnostics
         to see which diagnosticcs are highly correlated and
         e.g. need to be excluded from random forest analysis
'''
import numpy as np
import netCDF4 as nc
import glob as glob
import os
from os.path import expanduser
home = expanduser("~") # Get users home directory
import sys
import math
from scipy import signal, stats
sys.path.insert(0,home+'/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
from func_write_netcdf import func_write_netcdf
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import rc
###
# Define input & output
###
experiment = 'rcp85'
archive = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'
freq = 'mon'
res_name_target = 'JJA'

#NAM
#target_var = 'tasmax'
#target_file = 'CLIM'
#target_mask = 'maskT'
#diag_var = ['tasmax', 'pr', 'rlus', 'rsds', 'huss', 'psl', 'hfls', 'tos', 
#            'tasmax', 'pr', 'rlus', 'rsds', 'huss', 'psl', 'hfls', 'tos',
#            'tasmax', 'pr', 'rlus', 'rsds', 'huss', 'psl', 'hfls', 'tos']
#var_file = ['CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM','CLIM',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD','STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND']
#res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA','JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA','JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA']
#masko = [ 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT','maskF',
#          'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT','maskF',
#          'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT','maskF']

#target_var = 'TXx'
#target_file = 'CLIM'
#target_mask = 'maskT'
#res_name_target = 'ANN'
#target_time = 'MAX'
#freq = 'ann'

#diag_var = ['TXx', 'TXx', 'TXx',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tasclt', 'tashuss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tasclt', 'tashuss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tasclt', 'tashuss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tas', 'tasmax', 'tasmin', 'dtr', 'tos', 'pr',
#            'rnet', 'rlus', 'rsds', 'psl', 'ef', 'hfls', 'huss',
#            'tasclt', 'tashuss']
#var_file = ['CLIM', 'STD', 'TREND',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'CORR', 'CORR',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'CORR', 'CORR',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'CORR', 'CORR',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'CORR', 'CORR']
#res_name = ['ANN', 'ANN', 'ANN',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
#            'JJA', 'JJA',
#            'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM',
#            'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM',
#            'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM',
#            'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM',
#            'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM',
#            'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM', 'MAM',
#            'MAM', 'MAM',
#            'SON', 'SON', 'SON', 'SON', 'SON', 'SON',
#            'SON', 'SON', 'SON', 'SON', 'SON', 'SON', 'SON', 
#            'SON', 'SON', 'SON', 'SON', 'SON', 'SON',
#            'SON', 'SON', 'SON', 'SON', 'SON', 'SON', 'SON', 
#            'SON', 'SON', 'SON', 'SON', 'SON', 'SON',
#            'SON', 'SON', 'SON', 'SON', 'SON', 'SON', 'SON', 
#            'SON', 'SON',
#            'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 
#            'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF',
#            'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 
#            'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF',
#            'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 
#            'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF', 'DJF',
#            'DJF', 'DJF']
#masko = ['maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT' ,'maskT',
#         'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT' ,'maskT',
#         'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT' ,'maskT',
#         'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT' ,'maskT',
#         'maskT', 'maskT']
#res_time =['MAX', 'MAX', 'MAX',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',
#           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN',
#           'MEAN',  'MEAN']
#freq_v = ['ann', 'ann', 'ann', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
#          'mon', 'mon']

#target_var = 'tasmax'
#target_file = 'CLIM'
#target_mask = 'maskT'
#diag_var = ['tasmax', 'tasmin', 'dtr', 'pr', 'rlus', 'rsds', 'rnet', 'huss', 'psl', 'hfls', 'tos',
#            'tasmax', 'tasmin', 'dtr', 'pr', 'rlus', 'rsds', 'rnet', 'huss', 'psl', 'hfls', 'tos',
#            'tasmax', 'tasmin', 'dtr', 'pr', 'rlus', 'rsds', 'rnet', 'huss', 'psl', 'hfls', 'tos',
#            'tasclt', 'tashuss']
#var_file = ['CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM','CLIM', 'CLIM', 'CLIM',
#            'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD','STD', 'STD', 'STD',
#            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
#            'CORR', 'CORR']
#res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA','JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA','JJA',
#            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA','JJA',
#            'JJA', 'JJA']
#masko = ['maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF',
#         'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF',
#         'maskT', 'maskT']

target_var = 'hurs'
target_file = 'CLIM'
target_mask = 'maskF'
target_time = 'MEAN'
diag_var = ['hurs', 'tas', 'tasmax', 'tasmin', 'pr', 'rlus', 'rsds', 'rnet', 'huss', 'clt', 'psl', 'hfls', 'ef', 'tos',
            'hurs', 'tas', 'tasmax', 'tasmin', 'pr', 'rlus', 'rsds', 'rnet', 'huss', 'clt', 'psl', 'hfls', 'ef', 'tos',
            'hurs', 'tas', 'tasmax', 'tasmin', 'pr', 'rlus', 'rsds', 'rnet', 'huss', 'clt', 'psl', 'hfls', 'ef', 'tos',
            'tasclt', 'tashuss']
var_file = ['CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM', 'CLIM',
            'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD', 'STD',
            'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND', 'TREND',
            'CORR', 'CORR']
res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA',
            'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA']
res_time = ['MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN', 'MEAN',
           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN', 'MEAN',
           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN', 'MEAN',
           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN', 'MEAN',
           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN', 'MEAN',
           'MEAN',  'MEAN',  'MEAN',  'MEAN',  'MEAN', 'MEAN', 'MEAN',
           'MEAN',  'MEAN']
freq_v = ['mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
          'mon', 'mon', 'mon', 'mon', 'mon', 'mon', 'mon',
          'mon', 'mon']
masko = [ 'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT', 'maskT', 'maskF',
          'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT', 'maskT', 'maskF',
          'maskF', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT', 'maskT', 'maskF',
          'maskT', 'maskT']

nvar = len(diag_var)
outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/%s/Mult_Var_Lin_Reg/' %(target_var)
region = 'CNEU'         #cut data over region?

syear_hist = 1980
eyear_hist = 2014
syear_fut = 2065
eyear_fut = 2099

low_corr_thres_pval = 0.1
high_corr_thres = 0.8

nyears = eyear_hist - syear_hist + 1
grid = 'g025'

if (os.access(outdir, os.F_OK) == False):
        os.makedirs(outdir)

###read model data
print "Read model data"
##find all matching files in archive, loop over all of them
#first count matching files in folder
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
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v], freq_v[v],
            experiment, syear_hist, eyear_hist, res_name[v], res_time[v],
            var_file[v], region)
    else:
        name_v = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s.nc' %(
            archive, diag_var[v], freq_v[v], masko[v], diag_var[v], freq_v[v],
            experiment, syear_hist, eyear_hist, res_name[v], res_time[v],
            var_file[v])
    for filename in glob.glob(name_v):
        models_v.append(filename.split('_')[4] + ' ' + filename.split('_')[6])
        #find overlapping files for all variables
    overlap = list(set(models_v) & set(overlap))
    nfiles = len(overlap)
    #print str(nfiles)
    if nfiles == 0:
        break
    #print v, len(models_v)
#    del models_v
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
        if (target_file != 'SCALE') and (target_file != 'VARSCALE'):
            if region:
                filename_hist = '%s/%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(
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
            with np.errstate(invalid = 'ignore'):
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
feature = list()
data_unit = list()
for v in xrange(len(diag_var)):
    if region:
        name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s_%s.nc' %(
                archive, diag_var[v], freq_v[v], masko[v], diag_var[v],
                freq_v[v], experiment, syear_hist, eyear_hist, res_name[v],
                res_time[v], var_file[v], region)
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
            #check that time axis and grid is identical for models
            if f != 0:
                if temp_mod0.shape != temp_mod.shape:
                    print('Warning: Dimension for model0 and modelX' +
                          ' is different!')
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
    feature.append(diag_var[v] + '_' + var_file[v] + '_' + res_name[v])

if not region:
    region = 'GLOBAL'
corr_hist = np.zeros((nvar, nvar), dtype = float)
spear_corr_hist = np.zeros((nvar, nvar), dtype = float)
p_val_corr_hist = np.zeros((nvar, nvar), dtype = float)

my_xticks = list()
my_yticks = list()
my_yticks.append('$\Delta$' + target_var + target_file)
ma_change_target_areaavg = np.ma.masked_array(target, np.isnan(target))
for v in range(nvar):
    ma_data_areaavg = np.ma.masked_array(data[:, v], np.isnan(data[:, v]))
    corr = np.ma.corrcoef(ma_data_areaavg, ma_change_target_areaavg)
    corr_hist[0, v] = corr[0, 1]
    spear = stats.spearmanr(data[:, v], target, nan_policy = 'omit')
    spear_corr_hist[0, v] = spear[0]
    p_val_corr_hist[0, v] = spear[1]
    print('Spearman Rank Corrcoeff %s %s %s and delta %s %s: %s' %(
        diag_var[v], var_file[v], res_name[v], target_var, target_file,
        round(spear[0], 2)))
    my_xticks.append(diag_var[v] + var_file[v])
    my_yticks.append(diag_var[v] + var_file[v])
    del corr
    del spear
    for x in range(nvar - (v + 1)):
        ma_datav_areaavg = np.ma.masked_array(data[:, v],
                                              np.isnan(data[:, v]))
        ma_data_areaavg = np.ma.masked_array(data[:, x + v + 1],
                                             np.isnan(data[:, x + v + 1]))
        corr = np.ma.corrcoef(ma_data_areaavg, ma_datav_areaavg)
        corr_hist[v + 1, v + x + 1] = corr[0, 1]

        spear = stats.spearmanr(data[:, x + v + 1], data[:, v],
                                nan_policy = 'omit')
        spear_corr_hist[v + 1, v + x + 1] = spear[0]
        p_val_corr_hist[v + 1, v + x + 1] = spear[1]
        if (diag_var[v] == target_var) and ((var_file[v] == target_file)):
            print('Spearman Rank Corrcoeff %s %s %s and %s %s: %s' %(
                diag_var[x + v + 1], var_file[x + v + 1], res_name[x + v + 1],
                diag_var[v], var_file[v], round(spear[0], 2)))

        del corr
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
plt.xticks(range(nvar), my_xticks)
plt.yticks(range(nvar), my_yticks)
locs, labels = plt.xticks()
plt.setp(labels, rotation = 90)
cmap = plt.get_cmap("PuOr")
bounds = np.linspace(-0.95, 0.95, 20)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
CORR = ax.imshow(corr_hist, interpolation = 'nearest', cmap = cmap,
                 vmin = -1, vmax = 1, norm = norm)
#plt.plot([7, 7], [locs[0], locs[-1]], 'k-', lw = 2)
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size = "5%", pad = 0.05)
fig.colorbar(CORR, cax = cax1)
plt.savefig('%scorr_matrix_change_%s_%s_%s_%s.pdf' %(outdir,
                                                     target_var,
                                                     target_file,
                                                     res_name_target,
                                                     region))

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
plt.xticks(range(nvar), my_xticks)
plt.yticks(range(nvar), my_yticks)
locs, labels = plt.xticks()
plt.setp(labels, rotation = 90)
cmap = plt.get_cmap("PuOr")
bounds = np.linspace(-0.95, 0.95, 20)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

CORR_SPEAR = ax.imshow(spear_corr_hist, interpolation = 'nearest', cmap = cmap,
                       vmin = -1, vmax = 1, norm = norm)
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size = "5%", pad = 0.05)
fig.colorbar(CORR_SPEAR, cax = cax1)
plt.savefig('%scorr_spear_matrix_change_%s_%s_%s_%s.pdf' %(outdir,
                                                           target_var,
                                                           target_file,
                                                           res_name_target,
                                                           region))

# create masked array for data array, corr array and feature names
data_mask = np.ma.array(data, mask = False)
corr_hist_mask = np.ma.array(spear_corr_hist, mask = False)
feature_mask = np.ma.array(feature, mask = False)

# find diagnostics which have insignificant correlation with target variable
# exclude uncorrelated variables
ind_corr = np.where(p_val_corr_hist[0, :] >= low_corr_thres_pval)[0]
data_mask.mask[:, ind_corr] = True
corr_hist_mask.mask[:, ind_corr] = True
feature_mask.mask[ind_corr] = True

print(feature_mask.compressed())

# find variables with very high correlation between each other,
# mask second one
for r in range(len(spear_corr_hist)):
    for c in range(len(spear_corr_hist)):
        if round(spear_corr_hist[r, c], 2) >= high_corr_thres:
            data_mask.mask[:, c] = True
            corr_hist_mask.mask[:, c] = True
            feature_mask.mask[c] = True

# calculate variance inflation factor to check if still collinear features exist
# if vif > 10 -> exclude feature for further analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas
v = 0
diag = list()
for item in diag_var:
    diag.append(item + var_file[v] + ' ' + res_name[v])
    v = v +  1
df = pandas.DataFrame({'Diagnostic' : diag, diag[0] : spear_corr_hist[1, :], 
                       diag[0] + 'pval' : p_val_corr_hist[1, :], 
                       'delta' + target_var + target_file : spear_corr_hist[0, :],
                       'delta' + target_var + target_file + 'pval' : p_val_corr_hist[0, :]})
df.to_csv('%s%s_%s_corrdata.csv' %(target_var, target_file, region), sep = ',')

data_mean = np.ma.mean(data_mask, 0)
data_anom = data_mask - data_mean
data_comp = np.ma.compress_cols(data_anom)

vif_df = pandas.DataFrame()
vif_df["VIF Factor"] = [variance_inflation_factor(data_comp, i)
                        for i in range(data_comp.shape[1])]
vif_df["features"] = feature_mask.compressed()
print(vif_df.round(1))
