#!/usr/bin/python
'''
File Name : linear_reg_feature_selectionRFE_sklearn.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 14-02-2017
Modified: Mon 13 Feb 2017 09:14:59 CET
Purpose: Use randomized lasso and other feature selection methods for CMIP5 data
	 using sklearn python packages (http://scikit-learn.org)
         Scikit-learn: Machine Learning in Python, Pedregosa et al.,
                       JMLR 12, pp. 2825-2830, 2011.
         need to determine alphas for Lasso, Randomized Lasso beforehand.
         Use cross validation in LassoCV and use same alpha for Randomized Lasso
'''
import numpy as np
import netCDF4 as nc
import glob as glob
import os
import math
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, Lasso,
                                  LassoCV, 
                                  RandomizedLasso, BayesianRidge,
                                  TheilSenRegressor)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from os.path import expanduser
home = expanduser("~") # Get users home directory
import sys
sys.path.insert(0,home+'/scripts/plot_scripts/utils/')
from func_read_data import func_read_netcdf
from func_write_netcdf import func_write_netcdf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import operator
###
# Define input & output
###
experiment = 'rcp85'
archive = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/'

target_var = 'tasmax'
target_file = 'CLIM'
target_mask = 'maskT'

diag_var = ['tasmax', 'pr', 'tasmax', 'pr', 'huss', 'hfls', 'tos', 'tasmax', 'rsds']
var_file = ['CLIM', 'CLIM', 'STD', 'STD', 'STD', 'STD', 'STD', 'TREND', 'TREND']#
res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA'] #
masko = ['maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT']
#diag_var = ['tasmax', 'rlus', 'hfls', 'tasmax', 'rlus', 'tos', 'tasmax', 'rlus', 'rsds', 'tos'] ## preselected CNA
#var_file = ['CLIM', 'CLIM', 'CLIM', 'STD', 'STD', 'STD', 'TREND', 'TREND', 'TREND', 'TREND']#
#res_name = ['JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA', 'JJA'] #
#masko = ['maskT', 'maskT', 'maskT', 'maskT', 'maskT', 'maskF', 'maskT', 'maskT', 'maskT', 'maskF']


nvar = len(diag_var)

# parameters for sklearn feature selection methods 
rfe_nfeat = 7 # number of features to select in RFE method
rf_maxdepth = 6 # max depth in Random Forest method

outdir = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/%s/Mult_Var_Lin_Reg/' %(target_var)
region = 'NAM'         #cut data over region?

syear_hist = 1980
eyear_hist = 2014
syear_fut = 2065
eyear_fut = 2099

nyears = eyear_hist - syear_hist + 1
grid = 'g025'

if (os.access(outdir, os.F_OK) == False):
        os.makedirs(outdir)

names = ["%s%s %s" %(diag_var[i], var_file[i], res_name[i]) for i in range(0 , nvar)]
ranks = {}
pts_ranks = {}

def rank_to_dict(ranks, names, order = 1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def ranks_from_scores(sorted_scores):
    """sorted_scores: a list of tuples (object_id, score), sorted by score DESCENDING
       return a mapping of object IDs to ranks
    """
    ranks = {}
    previous_score = object()
    for index, (obj_id, score) in enumerate(sorted_scores):
        if score != previous_score:
            previous_score = score
            rank = index + 1
        ranks[obj_id] = rank
    return ranks

def pts_to_dict(sorted_ranks):
    rank_pts = dict()
    if (isinstance(sorted_ranks, list)):
        points = [10, 7, 5, 3, 1]
        for rank in xrange(len(sorted_ranks)):
            key = sorted_ranks[rank][0]
            if (rank < len(points)): 
                rank_pts[key] = points[rank]
            else:
                rank_pts[key] = 0
    if (isinstance(sorted_ranks, dict)):
        for key, value in sorted_ranks.items():
            if (value == 1):
                rank_pts[key] = 10.
            elif (value == 2):
                rank_pts[key] = 7.
            elif (value == 3):
                rank_pts[key] = 5.
            elif (value == 4):
                rank_pts[key] = 3.
            elif (value == 5):
                rank_pts[key] = 1.
            else:
                rank_pts[key] = 0.
    return rank_pts

###read model data
print "Read model data"
##find all matching files in archive, loop over all of them
#first count matching files in folder
models_t = list()
model_names = list()
if region:
    name = '%s/%s/%s/%s/%s_%s_*_%s_r?i?p?_%s-%s_%s%s_%s_%s.nc' %(
        archive, target_var, freq, target_mask, target_var, freq,
        experiment, syear_fut, eyear_fut, res_name_target, target_time,
        target_file, region)
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
            target_unit = tmp.units
            try:
                Fill = tmp._FillValue
            except AttributeError:
                Fill = 1e+20
            fh.close()
            if (target_file != 'SCALE'):
                if region:
                    filename_hist = '%s/%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s_%s.nc' %(archive, target_var, freq, target_mask, target_var, freq, filename.split('_')[4], experiment, filename.split('_')[6], syear_hist, eyear_hist, res_name_target, target_time, target_file, region)
                else:
                    filename_hist = '%s/%s/%s/%s/%s_%s_%s_%s_%s_%s-%s_%s%s_%s.nc' %(archive, target_var, freq, target_mask, target_var, freq, filename.split('_')[4], experiment, filename.split('_')[6], syear_hist, eyear_hist, res_name_target, target_time, target_file)
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
                            print('Warning: Dimension for model0 and modelX '
                                  'is different!')
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

lr = TheilSenRegressor(random_state = 50, copy_X = True)
lr.fit(data, target)
#ranks["OLS"] = rank_to_dict(np.abs(lr.coef_), names)

f, pval  = f_regression(data, target, center = True)
ranks["Corr."] = rank_to_dict(f, names)
sorted_scores = sorted(ranks['Corr.'].items(), key = operator.itemgetter(1),
                    reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks["Corr."] = pts_to_dict(sort_ranks)

#ridge = Ridge(alpha = 7)
#ridge.fit(data, target)
#ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

#ridgecv = RidgeCV(normalize = True)
#ridgecv.fit(data, target)
#ranks["RidgeCV"] = rank_to_dict(np.abs(ridgecv.coef_), names)

lasso = LassoCV(precompute = 'auto', cv = 5, selection = 'random',
                normalize = True)
lasso.fit(data, target)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
lasso_alpha = lasso.alpha_
sorted_scores = sorted(ranks['Lasso'].items(), key = operator.itemgetter(1),
                    reverse = True)
sort_ranks = ranks_from_scores(sorted_scores) 
pts_ranks["Lasso"] = pts_to_dict(sort_ranks)

rlasso = RandomizedLasso(alpha = lasso_alpha, normalize = True)
rlasso.fit(data, target)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
sorted_scores = sorted(ranks['Stability'].items(), key = operator.itemgetter(1),
                    reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks["Stability"] = pts_to_dict(sort_ranks)

#baysridge = linear_model.BayesianRidge(compute_score = True)
#baysridge.fit(data, target)
#ranks["BR"] = rank_to_dict(np.abs(baysridge.scores_), names)

#theilsen = linear_model.TheilSenRegressor(random_state = 50, copy_X = True)
#theilsen.fit(data, target)
#ranks["TS"] = rank_to_dict(np.abs(theilsen.coef_), names)

# stop the search when 4 (or whatever rfe_nfeat is set to) features are left
# (they will get equal scores)
rfe = RFE(lr, n_features_to_select = rfe_nfeat)
rfe.fit(data, target)
ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)
sorted_scores = sorted(ranks['RFE'].items(), key = operator.itemgetter(1),
                    reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks["RFE"] = pts_to_dict(sort_ranks)
 
rf = RandomForestRegressor(random_state = 0, n_estimators = 100,
                           max_features = 'sqrt', max_depth = rf_maxdepth)
rf.fit(data, target)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
sorted_scores = sorted(ranks['RF'].items(), key = operator.itemgetter(1),
                    reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks["RF"] = pts_to_dict(sort_ranks)

r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)

r_pts = {}
for name in names:
    r_pts[name] = round(np.sum([pts_ranks[method][name] 
                             for method in pts_ranks.keys()]), 2)
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
ranks["Points"] = r_pts
methods.append("Points")

sort_pts = sorted(ranks['Points'].items(), key = operator.itemgetter(1),
       reverse = True)
dict_ranks = dict()
for i in xrange(len(sort_pts)):
    key = sort_pts[i][0]
    dict_ranks[key] = i + 1
ranks["Rank"] = dict_ranks
methods.append("Rank")
 
print "\t\t&%s\\" % "\t\t&".join(methods)
for name in names:
    print "%s&\t\t%s\\" % (name, "\t\t&".join(map(str,
        [ranks[method][name] for method in methods])))


del lr
del lasso
#del ridge
del rlasso
del rfe
del rf
#del baysridge
#del theilsen
ranks_hist = {}
pts_ranks_hist = {}

lr = TheilSenRegressor(random_state = 50, copy_X = True)
lr.fit(data[:, 1:], data[:, 0])
#ranks_hist["OLS"] = rank_to_dict(np.abs(lr.coef_), names[1:])

f, pval  = f_regression(data[:, 1:], data[:, 0], center = True)
ranks_hist["Corr."] = rank_to_dict(f, names[1:])
sorted_scores = sorted(ranks_hist['Corr.'].items(),
                       key = operator.itemgetter(1),
                       reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks_hist["Corr."] = pts_to_dict(sort_ranks)

#ridge = Ridge(normalize = True)
#ridge.fit(data[:, 1:], data[:, 0])
#ranks_hist["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names[1:])

#ridgecv = RidgeCV(normalize = True)
#ridgecv.fit(data[:, 1:], data[:, 0])
#ranks_hist["RidgeCV"] = rank_to_dict(np.abs(ridgecv.coef_), names[1:])
                                                        
lasso = LassoCV(precompute = 'auto', cv = 5, selection = 'random',
                normalize = True)
#lasso = Lasso(alpha = lasso_alpha, normalize = True)
lasso.fit(data[:, 1:], data[:, 0])
ranks_hist["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names[1:])
lasso_alpha = lasso.alpha_
sorted_scores = sorted(ranks_hist['Lasso'].items(),
                       key = operator.itemgetter(1), reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks_hist["Lasso"] = pts_to_dict(sort_ranks)

rlasso = RandomizedLasso(alpha = lasso_alpha, normalize = True)
rlasso.fit(data[:, 1:], data[:, 0])
ranks_hist["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names[1:])
sorted_scores = sorted(ranks_hist['Stability'].items(),
                    key = operator.itemgetter(1), reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks_hist["Stability"] = pts_to_dict(sort_ranks)

#baysridge = linear_model.BayesianRidge(compute_score = True)
#baysridge.fit(data[:, 1:], data[:, 0])
#ranks_hist["BR"] = rank_to_dict(np.abs(baysridge.scores_), names[1:])

#theilsen = linear_model.TheilSenRegressor(random_state = 50, copy_X = True)
#theilsen.fit(data[:, 1:], data[:, 0])
#ranks_hist["TS"] = rank_to_dict(np.abs(theilsen.coef_), names[1:])

# stop the search when 5 (or whatever rfe_nfeat is set to) features are left
# (they will get equal scores)
rfe = RFE(lr, n_features_to_select = rfe_nfeat)
rfe.fit(data[:, 1:], data[:, 0])
ranks_hist["RFE"] = rank_to_dict(map(float, rfe.ranking_), names[1:],
                                 order = -1)
sorted_scores = sorted(ranks_hist['RFE'].items(), key = operator.itemgetter(1),
                    reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks_hist["RFE"] = pts_to_dict(sort_ranks)

rf = RandomForestRegressor(random_state = 0, n_estimators = 100,
                           max_features = 'sqrt', max_depth = rf_maxdepth)
rf.fit(data[:, 1:], data[:, 0])
ranks_hist["RF"] = rank_to_dict(rf.feature_importances_, names[1:])

sorted_scores = sorted(ranks_hist['RF'].items(), key = operator.itemgetter(1),
                       reverse = True)
sort_ranks = ranks_from_scores(sorted_scores)
pts_ranks_hist["RF"] = pts_to_dict(sort_ranks)

r_hist = {}
for name in names[1:]:
    r_hist[name] = round(np.mean([ranks_hist[method][name] 
                             for method in ranks_hist.keys()]), 2)
r_pts_hist = {}
for name in names[1:]:
    r_pts_hist[name] = round(np.sum([pts_ranks_hist[method][name] 
                             for method in pts_ranks_hist.keys()]), 2)
 
methods = sorted(ranks_hist.keys())
ranks_hist["Mean"] = r_hist
methods.append("Mean")
ranks_hist["Points"] = r_pts_hist
methods.append("Points")

sort_pts_hist = sorted(ranks_hist['Points'].items(),
                       key = operator.itemgetter(1),
                       reverse = True)
dict_ranks_hist = dict()
for i in xrange(len(sort_pts_hist)):
    key = sort_pts_hist[i][0]
    dict_ranks_hist[key] = i + 1
ranks_hist["Rank"] = dict_ranks_hist
methods.append("Rank")

print "\t\t&%s\\" % "\t\t&".join(methods)
for name in names[1:]:
    print "%s&\t\t%s\\" % (name, "\t\t&".join(map(str, 
                         [ranks_hist[method][name] for method in methods])))
