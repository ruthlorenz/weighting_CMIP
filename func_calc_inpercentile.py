#!/usr/bin/python
'''
File Name : func_calc_inpercentile.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 20-09-2016
Modified: Tue 20 Sep 2016 12:44:33 PM CEST
Purpose: calculate how many times within 5-95% percentile for all sigmas
	to determine sigma combination which is not overconfindent but
	within 5-95% percentile

'''
import numpy as np
import copy
import math

def calc_inpercentile(weights, var_val, approx = None, verbose = False):
    dim_s2 = weights.shape[0]
    dim_s1 = weights.shape[1]
    dim_models = weights.shape[2]
    ## depending on given approximation or not the values to calculate
    ## the percentile from changes, change in sigma loop if approx != None
    values = var_val
    approx_percent = np.empty((dim_s2, dim_s1))
    for s2 in xrange(dim_s2):
	for s1 in xrange(dim_s1):
            flag_array = np.empty((dim_models))
            round_weights = np.empty((dim_models))
            if approx != None:
                values = approx[s2, s1, :]
	    for m in xrange(dim_models):
                for mm in xrange(dim_models):
                    round_weights[mm] = int(round(
                        weights[s2, s1, m, mm] * dim_models * 1000))
                ind_end = np.cumsum(round_weights).astype(int)
                ind_start = copy.deepcopy(ind_end)  # init
                ind_start[0] = 0
                ind_start[1:] = copy.deepcopy(ind_end[:dim_models - 1] + 1)
                tmp = np.empty((dim_models * 1000 + 10))

                for mm in xrange(dim_models):
                    #print "%s %s %s" %(mm, ind_start[mm], ind_end[mm])
                    if ind_end[mm] > 100010:
                        ind_end[mm] = 100009
                    for kk in xrange(ind_start[mm], ind_end[mm] + 1):
                        tmp[kk] = copy.deepcopy(values[mm])

                dim = np.count_nonzero(tmp)
                tmp2 = tmp[:dim]
                tmp1 = sorted(tmp2, reverse = True)
                del tmp2
                dim = np.count_nonzero(tmp1)
                ind_10 = int(math.ceil(0.05 * dim) - 1)
                ind_90 = int(math.floor(0.95 * dim) - 1)
                if verbose:
                    print "%s %s %s %s %s %s" %(m, len(tmp), ind_10, ind_90,
                                                values.shape, flag_array.shape)
                if ind_10 != 0.:
                    if values[m] < tmp1[ind_10] and values[m] > tmp1[ind_90]:
                        flag_array[m] = 1
                    else:
                        flag_array[m] = 0
                else:
                    flag_array[m] = np.nan
                if verbose:
                    print "%s %s %s %s" %(var_val[k], tmp1[ind_10],
                                          tmp1[ind_90], flag_array[k])
                del tmp
            approx_percent[s2, s1] = np.sum(flag_array) / np.count_nonzero(~np.isnan(flag_array))

    return approx_percent

