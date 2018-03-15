#!/usr/bin/python
'''
File Name : func_eval_wmm_non-wmm_error_indexI.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 14-03-2016
Modified: Fri 09 Mar 2018 09:58:24 AM CET
Purpose: evaluate weighted multi-model mean versus non-weighted multi-model mean
	using error index I^2 (as proposed in Baker and Taylor, 2016, JClim)


'''
import numpy as np
from math import atan

def error_indexI(weighted_mm, nonweighted_mm, obs_clim, obs_std,
                 lat = None, lon = None, tp = None):
    if lat:
        if lat.any():
            rad = 4.0 * atan(1.0) / 180
            w_lat = np.cos(lat * rad) # weight for latitude differences in area
    e_wmm2 = 0
    e_mm2 = 0
    dims = weighted_mm.shape
    #for 3D data
    if (len(dims) == 3) and (tp == "cyc"):
        for i in xrange(len(lon)):
            for j in xrange(len(lat)):
                if not ((np.isnan(weighted_mm[:, j, i]).any()) or
                        (np.isnan(obs_clim[:, j, i]).any()) or
                        (np.isnan(obs_std[:, j, i]).any())):
                    e_wmm2 = e_wmm2 + (w_lat[j] * (weighted_mm[:, j, i] -
                                                   obs_clim[:, j, i]) ** 2 /
                                       obs_std[:, j, i] ** 2)
                if not ((np.isnan(nonweighted_mm[:, j, i]).any()) or
                        (np.isnan(obs_clim[:, j, i]).any()) or
                        (np.isnan(obs_std[:, j, i]).any())): 
                    e_mm2 = e_mm2 + (w_lat[j] * (nonweighted_mm[:, j, i] -
                                                 obs_clim[:, j, i]) ** 2 /
                                     obs_std[:, j, i] ** 2)
        e_wmm2 = np.nanmean(e_wmm2)
        e_mm2 = np.nanmean(e_mm2)
    elif (len(dims) == 3) and (tp != "cyc"):
        for i in xrange(len(lon)):
            for j in xrange(len(lat)):
                if not ((np.isnan(weighted_mm[:, j, i]).any()) or
                        (np.isnan(obs_clim[j, i]).any()) or
                        (np.isnan(obs_std[j, i]).any())):
                    e_wmm2 = e_wmm2 + (w_lat[j] * (weighted_mm[:, j, i] -
                                                   obs_clim[j, i]) ** 2 /
                                       obs_std[j,i] ** 2)
                if not ((np.isnan(nonweighted_mm[:, j, i]).any()) or
                        (np.isnan(obs_clim[j, i]).any()) or
                        (np.isnan(obs_std[j, i]).any())): 
                    e_mm2 = e_mm2 + (w_lat[j] * (nonweighted_mm[:, j, i] -
                                                 obs_clim[j, i]) ** 2 /
                                     obs_std[j, i] ** 2)
        e_wmm2 = np.nanmean(e_wmm2)
        e_mm2 = np.nanmean(e_mm2)
    elif len(dims) == 2:
        for i in xrange(len(lon)):
            for j in range(len(lat)):
                if not ((np.isnan(weighted_mm[j, i])) or
                        (np.isnan(obs_clim[j, i])) or
                        (np.isnan(obs_std[j, i])) or
			(obs_std[j, i] == 0.0)): 
                    e_wmm2 = e_wmm2 + (w_lat[j] * (weighted_mm[j, i] -
                                                   obs_clim[j, i])** 2 /
                                       obs_std[j, i] ** 2)
                if not ((np.isnan(nonweighted_mm[j, i])) or
                        (np.isnan(obs_clim[j, i])) or
                        (np.isnan(obs_std[j, i])) or
			(obs_std[j, i] == 0.0)):
                    e_mm2 = e_mm2 + (w_lat[j] * (nonweighted_mm[j, i] -
                                                 obs_clim[j, i]) ** 2 /
                                     obs_std[j, i] ** 2)
    ## timeseries at point or already averaged over area
    elif len(dims) == 1: 
        e_wmm2 = np.nansum((weighted_mm - obs_clim) ** 2 / obs_std ** 2)
        e_mm2 = np.nansum((nonweighted_mm - obs_clim) ** 2 / obs_std ** 2)
    #print e_wmm2, e_mm2
    I_2 = e_wmm2 / e_mm2

    return I_2
