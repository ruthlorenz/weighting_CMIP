#!/usr/bin/python
'''
File Name : func_calc_corr.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 14-03-2016
Modified: Thu 09 Feb 2017 09:54:18 AM CET
Purpose: calculate correlation between approximated and observed data

'''
import numpy as np
import scipy.stats
import copy
def calc_corr(sigma2, sigma1, approx, var_val, model_names):
    corr = np.empty((len(sigma2), len(sigma1)))
    corrcoef = np.empty((len(sigma2), len(sigma1)))
    #pval = np.empty((len(sigma2),len(sigma1),len(model_names)))
    print corr.shape
    for s2 in xrange(len(sigma2)):
        for s1 in xrange(len(sigma1)):
            tmp_app = approx[s2, s1, :, :]
            #tmp_var = var_val
            spear = scipy.stats.spearmanr(tmp_app, var_val, axis = None)
                                          #nan_policy = 'omit')
            corr[s2, s1] = copy.deepcopy(spear[0])
	    corrcoef[s2, s1] = np.corrcoef(tmp_app, var_val)[0, 1]
    return corrcoef
