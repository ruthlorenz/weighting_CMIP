#!/usr/bin/python
'''
File Name : mult_dim_scale_delta_matrix.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 03-03-2017
Modified: Fri 03 Mar 2017 09:50:24 AM CET
Purpose: perform multi-dimesnional scaling on delta matrix
         to make sure distances between model-obs larger than model-model
'''
import netCDF4 as nc # to work with NetCDF files
import numpy as np
from sklearn.manifold import MDS
import os # operating system interface
from os.path import expanduser
home = expanduser("~") # Get users home directory
import glob
import sys
sys.path.insert(0, home + '/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
import matplotlib as mpl
import matplotlib.pyplot as plt
###
# Define input
###
# multiple variables possible but deltas need to be available
# and first variable determines plotting and titles in plots
diag_var = ['tasmax', 'rsds', 'pr', 'tos'] #, 'rsds', 'pr', 'tos']
#diag_var = ['tasmax', 'pr', 'rsds', 'huss'] #, 'rlus', 'rsds', 'huss'] 
# climatology:clim, variability:std, trend:trnd
var_file = ['CLIM', 'TREND', 'CLIM', 'STD'] #, 'TREND', 'CLIM', 'STD']
#var_file = ['STD', 'STD', 'TREND', 'CLIM'] #, 'STD', 'STD', 'TREND', 'CLIM']
# kind is cyc: annual cycle, mon: monthly values, seas: seasonal
res_name = ['JJA', 'JJA', 'JJA', 'JJA'] #, 'JJA', 'JJA', 'JJA', 'JJA']
masko = ['maskT', 'maskT', 'maskT', 'maskF']
# weight of individual fields, all equal weight 1 at the moment
fields_weight = [1, 1, 1, 1] #, 1, 1, 1, 1]

# choose region if required
region = 'NAM'
if region != None:
    area = region
else:
    area = 'GLOBAL'
syear_eval = [1980, 2000, 1980, 1980]
eyear_eval = [2014, 2014, 2014, 2014]
s_year = 1951
e_year = 2100

experiment = 'rcp85'
#grid = 'g025'
freq = 'mon'

obsdata = 'Obs' #ERAint, MERRA2, Obs
obsname = 'HadGHCND, CERES, GPCP, HadISST' #'ERAint', 'HadGHCND, CERES, GPCP'
err = 'RMSE' #'perkins_SS', 'RMSE'
err_var = 'rmse' #'SS', 'rmse'

indir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'
outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/Eval_Weight/%s/%s/' %(diag_var[0], area)
if (os.access(outdir, os.F_OK) == False):
    os.makedirs(outdir)

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
rmse = np.ndarray((len(var_file), len(overlap) + 1, len(overlap) + 1))
deltas = np.ndarray((len(var_file), len(overlap) + 1, len(overlap) + 1))
delta_avg = np.ndarray((len(var_file), len(overlap) + 1, len(overlap) + 1))
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
    if (os.access('%s_%s_%s_%s.nc' %(rmsefile, region, obsdata, err),
                  os.F_OK) == True):
        print err + ' already exist, read from netcdf'
        fh = nc.Dataset('%s_%s_%s_%s.nc' %(rmsefile, region, obsdata, err),
                        mode = 'r')
        rmse_all = fh.variables[err_var]
        rmse[v, : -1, : -1] = rmse_all[ind, ind]
        rmse[v, -1, 1:] = rmse_all[ - 1, ind]
        rmse[v, 1:, -1] = rmse_all[ind, - 1]
        fh.close()
    else:
        print "RMSE delta matrix does not exist yet, exiting"
        sys.exit
model_names.append(obsname)

for v in xrange(len(var_file)):
    ## normalize rmse by median
    med = np.nanmedian(rmse[v, :, :])
    deltas[v, :, :] = rmse[v, :, :] / med
## average deltas over fields,
## taking field weight into account (all 1 at the moment)
field_w_extend_u = np.reshape(np.repeat(fields_weight,
                                        len(model_names) * len(model_names)),
                              (len(fields_weight), len(model_names),
                               len(model_names)))
delta_avg = np.sqrt(np.nansum(field_w_extend_u * deltas, axis = 0)
                    / np.nansum(fields_weight))

# convert model_names into float with models from same group having
# similar values (to be plotted in similar colors)
models = list()
for m in xrange(len(model_names)):
    models.append(model_names[m][0 : 3])
mod = 0.
model_groups = np.ndarray((len(model_names)))
for i in xrange(len(models)):
    if (i == 0):
        model_groups[i] = mod
    else:
        if models[i] == models[i - 1]:
            model_groups[i] = mod
        else:
            mod = mod + 1
            model_groups[i] = mod
# multi-dimensional scaling
mds = MDS(n_components = 2, dissimilarity = "precomputed", random_state = 5)
results = mds.fit(delta_avg)
coords = results.embedding_
plt.subplots_adjust(bottom = 0.1)
plt.scatter(coords[:, 0], coords[:, 1], marker = 'o', s = 50, c = model_groups,
            cmap = plt.cm.nipy_spectral_r, edgecolors = 'none')
#for label, x, y in zip(model_names, coords[:, 0], coords[:, 1]):
#    plt.annotate(label, xy = (x, y), xytext = (-20, 20),
#                 textcoords = 'offset points', ha = 'right', va = 'bottom',
#                 arrowprops = dict(arrowstyle = '->',
#                 connectionstyle = 'arc3, rad = 0'))
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])

label = model_names[-1]
x = coords[-1, 0]
y = coords[-1, 1]
plt.annotate(label, xy = (x, y), xytext = (75, 40),
             textcoords = 'offset points', ha = 'right', va = 'bottom',
             arrowprops = dict(arrowstyle = '->',
                               connectionstyle = 'arc3, rad = 0'))

plt.title('Multi-dimensional scaling of distance matrix')
plt.savefig(outdir + 'MDS_deltas_%s_%s_%s_%s_%s_%s_%s.pdf' %(diag_var[0],
                                                             var_file[0],
                                                             res_name[0],
                                                             len(diag_var),
                                                             obsdata, area,
                                                             err))
