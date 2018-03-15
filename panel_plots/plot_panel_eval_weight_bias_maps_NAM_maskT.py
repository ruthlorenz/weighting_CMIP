#!/usr/bin/python
'''
File Name : plot_panel_eval_weight_maps_NAM.py
Author: Ruth Lorenz (ruth.lorenz@env.ethz.ch)
Created: 26-04-2017
Modified: Wed 26 Apr 2017 11:25:44 AM CEST
Purpose: plot panel with maps of difference between weighted and unweighted
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

###
# Define input
###
variable = 'tasmax'
diag = 'CLIM'
region = 'NAM'
obsdata = ['MERRA2', 'ERAint', 'Obs']
obsname = ['MERRA2', 'ERAint', 'OBS']
diagnum = ['1', '2', '3', '4', '5', '6'] #, '2', '3', '4'
wu = '0.6'
wq = '0.5'
degree_sign = u'\N{DEGREE SIGN}'
convert = 0 # set to zero if no conversion to different unit
unit = degree_sign + "C"

path = '/net/tropo/climphys/rlorenz/processed_CMIP5_data/Eval_Weight'
indir = '%s/%s/%s/ncdf' %(path, variable, region)
outdir = '%s/%s/%s/plots' %(path, variable, region)

if (os.access('%s' %outdir, os.F_OK) == False):
    os.makedirs('%s' %outdir)
    print 'created directory %s' %outdir

ncol = len(obsdata)
nrow = len(diagnum) + 1
plottype= ".eps"

###
# read data
###
data = {}
IError = {}
# loop over diagnostics number
for d in range(0, nrow):
    dnum = str(d)
    for o in range(0, ncol):
        if d == 0:
            dnum = '0'
            path_mm = '%s/plot_bias_mm_%s%s_JJA_latlon_1_%s_%s_RMSE_%s_%s.nc' %(
                indir, variable, diag, obsdata[o], region, wu, wq)
            print path_mm
            ifile = nc.Dataset(path_mm)
            lats = ifile.variables['lat'][:]
            lons = ifile.variables['lon'][:]
            mm = ifile.variables[variable][:]
            data[obsdata[o] + '_' + dnum] = mm
        else:
            path_wmm = '%s/plot_bias_wmm_%s%s_JJA_latlon_%s_%s_%s_RMSE_%s_%s.nc' %(
                indir, variable, diag, dnum, obsdata[o], region, wu, wq)
            print path_wmm
            ifile = nc.Dataset(path_wmm)
            lats = ifile.variables['lat'][:]
            lons = ifile.variables['lon'][:]
            wmm = ifile.variables[variable][:]
            comment = str(round(float(ifile.comment), 2))

            data[obsdata[o] + '_' + dnum] = wmm
            IError[obsdata[o] + '_' + dnum] = comment
###
# plotting part using gridspec
###
plt.close('all')
if (len(diagnum) == 4):
    diff = [data['%s_0' %(obsdata[0])], data['%s_0' %(obsdata[1])],
            data['%s_0' %(obsdata[2])],
            data['%s_1' %(obsdata[0])], data['%s_1' %(obsdata[1])],
            data['%s_1' %(obsdata[2])],
            data['%s_2' %(obsdata[0])], data['%s_2' %(obsdata[1])],
            data['%s_2' %(obsdata[2])],
            data['%s_3' %(obsdata[0])], data['%s_3' %(obsdata[1])],
            data['%s_3' %(obsdata[2])],
            data['%s_4' %(obsdata[0])], data['%s_4' %(obsdata[1])],
            data['%s_4' %(obsdata[2])]]
    titles = ['(a) I$^2$ = %s' %(IError['%s_1' %(obsdata[0])]),
              '(b) I$^2$ = %s' %(IError['%s_1' %(obsdata[1])]),
              '(c) I$^2$ = %s' %(IError['%s_1' %(obsdata[2])]),
              '(d) I$^2$ = %s' %(IError['%s_2' %(obsdata[0])]),
              '(e) I$^2$ = %s' %(IError['%s_2' %(obsdata[1])]),
              '(f) I$^2$ = %s' %(IError['%s_2' %(obsdata[2])]),
              '(g) I$^2$ = %s' %(IError['%s_3' %(obsdata[0])]),
              '(h) I$^2$ = %s' %(IError['%s_3' %(obsdata[1])]),
              '(i) I$^2$ = %s' %(IError['%s_3' %(obsdata[2])]),
              '(j) I$^2$ = %s' %(IError['%s_4' %(obsdata[0])]),
              '(k) I$^2$ = %s' %(IError['%s_4' %(obsdata[1])]),
              '(l) I$^2$ = %s' %(IError['%s_4' %(obsdata[2])])]
    figheight = 10
    heightratios = [0.1, 1, 1, 1, 1, 0.1]
elif (len(diagnum) == 5):
    diff = [data['%s_1' %(obsdata[0])], data['%s_1' %(obsdata[1])],
            data['%s_1' %(obsdata[2])],
            data['%s_2' %(obsdata[0])], data['%s_2' %(obsdata[1])],
            data['%s_2' %(obsdata[2])],
            data['%s_3' %(obsdata[0])], data['%s_3' %(obsdata[1])],
            data['%s_3' %(obsdata[2])],
            data['%s_4' %(obsdata[0])], data['%s_4' %(obsdata[1])],
            data['%s_4' %(obsdata[2])],
            data['%s_5' %(obsdata[0])], data['%s_5' %(obsdata[1])],
            data['%s_5' %(obsdata[2])]]
    titles = ['(a) I$^2$ = %s' %(IError['%s_1' %(obsdata[0])]),
              '(b) I$^2$ = %s' %(IError['%s_1' %(obsdata[1])]),
              '(c) I$^2$ = %s' %(IError['%s_1' %(obsdata[2])]),
              '(d) I$^2$ = %s' %(IError['%s_2' %(obsdata[0])]),
              '(e) I$^2$ = %s' %(IError['%s_2' %(obsdata[1])]),
              '(f) I$^2$ = %s' %(IError['%s_2' %(obsdata[2])]),
              '(g) I$^2$ = %s' %(IError['%s_3' %(obsdata[0])]),
              '(h) I$^2$ = %s' %(IError['%s_3' %(obsdata[1])]),
              '(i) I$^2$ = %s' %(IError['%s_3' %(obsdata[2])]),
              '(j) I$^2$ = %s' %(IError['%s_4' %(obsdata[0])]),
              '(k) I$^2$ = %s' %(IError['%s_4' %(obsdata[1])]),
              '(l) I$^2$ = %s' %(IError['%s_4' %(obsdata[2])]),
              '(m) I$^2$ = %s' %(IError['%s_5' %(obsdata[0])]),
              '(n) I$^2$ = %s' %(IError['%s_5' %(obsdata[1])]),
              '(o) I$^2$ = %s' %(IError['%s_5' %(obsdata[2])])]
    figheight = 10.5
    heightratios = [0.1, 1, 1, 1, 1, 1, 0.1]
elif (len(diagnum) == 1):
    diff = [data['%s_1' %(obsdata[0])], data['%s_1' %(obsdata[1])],
            data['%s_1' %(obsdata[2])]]
    titles = ['(a) I$^2$ = %s' %(IError['%s_1' %(obsdata[0])]),
              '(b) I$^2$ = %s' %(IError['%s_1' %(obsdata[1])]),
              '(c) I$^2$ = %s' %(IError['%s_1' %(obsdata[2])])]
    heightratios = [0.1, 1, 0.1]
    figheight = 4
elif (len(diagnum) == 2):
    diff = [data['%s_1' %(obsdata[0])], data['%s_1' %(obsdata[1])],
            data['%s_1' %(obsdata[2])],
            data['%s_2' %(obsdata[0])], data['%s_2' %(obsdata[1])],
            data['%s_2' %(obsdata[2])]]
    titles = ['(a) I$^2$ = %s' %(IError['%s_1' %(obsdata[0])]),
              '(b) I$^2$ = %s' %(IError['%s_1' %(obsdata[1])]),
              '(c) I$^2$ = %s' %(IError['%s_1' %(obsdata[2])]),
              '(d) I$^2$ = %s' %(IError['%s_2' %(obsdata[0])]),
              '(e) I$^2$ = %s' %(IError['%s_2' %(obsdata[1])]),
              '(f) I$^2$ = %s' %(IError['%s_2' %(obsdata[2])])]
    heightratios = [0.1, 1, 1, 0.1]
    figheight = 5.5
elif (len(diagnum) == 3):
    diff = [data['%s_1' %(obsdata[0])], data['%s_1' %(obsdata[1])],
            data['%s_1' %(obsdata[2])],
            data['%s_2' %(obsdata[0])], data['%s_2' %(obsdata[1])],
            data['%s_2' %(obsdata[2])],
            data['%s_3' %(obsdata[0])], data['%s_3' %(obsdata[1])],
            data['%s_3' %(obsdata[2])]]
    titles = ['(a) I$^2$ = %s' %(IError['%s_1' %(obsdata[0])]),
              '(b) I$^2$ = %s' %(IError['%s_1' %(obsdata[1])]),
              '(c) I$^2$ = %s' %(IError['%s_1' %(obsdata[2])]),
              '(d) I$^2$ = %s' %(IError['%s_2' %(obsdata[0])]),
              '(e) I$^2$ = %s' %(IError['%s_2' %(obsdata[1])]),
              '(f) I$^2$ = %s' %(IError['%s_2' %(obsdata[2])]),
              '(g) I$^2$ = %s' %(IError['%s_3' %(obsdata[0])]),
              '(h) I$^2$ = %s' %(IError['%s_3' %(obsdata[1])]),
              '(i) I$^2$ = %s' %(IError['%s_3' %(obsdata[2])])]
    heightratios = [0.1, 1, 1, 1, 0.1]
    figheight = 7.5
elif (len(diagnum) == 6):
    diff = [data['%s_0' %(obsdata[0])], data['%s_0' %(obsdata[1])],
            data['%s_0' %(obsdata[2])],
            data['%s_1' %(obsdata[0])], data['%s_1' %(obsdata[1])],
            data['%s_1' %(obsdata[2])],
            data['%s_2' %(obsdata[0])], data['%s_2' %(obsdata[1])],
            data['%s_2' %(obsdata[2])],
            data['%s_3' %(obsdata[0])], data['%s_3' %(obsdata[1])],
            data['%s_3' %(obsdata[2])],
            data['%s_4' %(obsdata[0])], data['%s_4' %(obsdata[1])],
            data['%s_4' %(obsdata[2])],
            data['%s_5' %(obsdata[0])], data['%s_5' %(obsdata[1])],
            data['%s_5' %(obsdata[2])],
            data['%s_6' %(obsdata[0])], data['%s_6' %(obsdata[1])],
            data['%s_6' %(obsdata[2])]]
    titles = ['(a) Bias non-weighted MMM',
              '(b) Bias non-weighted MMM',
              '(c) Bias non-weighted MMM',
              '(d) I$^2$ = %s' %(IError['%s_1' %(obsdata[0])]),
              '(e) I$^2$ = %s' %(IError['%s_1' %(obsdata[1])]),
              '(f) I$^2$ = %s' %(IError['%s_1' %(obsdata[2])]),
              '(g) I$^2$ = %s' %(IError['%s_2' %(obsdata[0])]),
              '(h) I$^2$ = %s' %(IError['%s_2' %(obsdata[1])]),
              '(i) I$^2$ = %s' %(IError['%s_2' %(obsdata[2])]),
              '(j) I$^2$ = %s' %(IError['%s_3' %(obsdata[0])]),
              '(k) I$^2$ = %s' %(IError['%s_3' %(obsdata[1])]),
              '(l) I$^2$ = %s' %(IError['%s_3' %(obsdata[2])]),
              '(m) I$^2$ = %s' %(IError['%s_4' %(obsdata[0])]),
              '(n) I$^2$ = %s' %(IError['%s_4' %(obsdata[1])]),
              '(o) I$^2$ = %s' %(IError['%s_4' %(obsdata[2])]),
              '(p) I$^2$ = %s' %(IError['%s_5' %(obsdata[0])]),
              '(q) I$^2$ = %s' %(IError['%s_5' %(obsdata[1])]),
              '(r) I$^2$ = %s' %(IError['%s_5' %(obsdata[2])]),
              '(s) I$^2$ = %s' %(IError['%s_6' %(obsdata[0])]),
              '(t) I$^2$ = %s' %(IError['%s_6' %(obsdata[1])]),
              '(u) I$^2$ = %s' %(IError['%s_6' %(obsdata[2])]),
]
    figheight = 15.5
    heightratios = [0.1, 1, 1, 1, 1, 1, 1, 1, 0.1]

max_lat = np.amax(lats)
min_lat = np.amin(lats)
max_lon = -65.0 #np.amax(lons)
min_lon = -125.0 #np.amin(lons)

colbar='RdBu_r'
levels = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5,
          2.5, 3.5, 4.5, 5.5]
#levels = [-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8]

fig = plt.figure(figsize = (12, figheight))
plotname = '%s/panel_maps_bias_weighted_%s_%s_%s_%s_%s_%sby%s' %(
    outdir, variable, diag, region, wu, wq, nrow, ncol)
gs = gridspec.GridSpec(nrow + 2, ncol + 1,
                       height_ratios = heightratios,
                       width_ratios = [0.25, 1, 1, 1], hspace = 0.1,
                       wspace = 0.1, top = 0.95, right = 0.95, left = 0.05,
                       bottom = 0.05)
for obs in range(0, len(obsdata)):
    ax = plt.subplot(gs[0, obs + 1], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    textobs = obsname[obs]
    ax.text(0., 0, textobs, size = 12)

ax = plt.subplot(gs[2, 0], frameon = False)
ax.axes.get_yaxis().set_visible(False)
ax.axes.get_xaxis().set_visible(False)
text1 = '%s%s' %(variable, diag)
ax.text(0., 0.75, text1, size = 12, rotation = 90, transform=ax.transAxes)

if (len(diagnum) > 1):
    ax = plt.subplot(gs[3, 0], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if (variable == 'tasmax' and diag == 'CLIM'):
        text1 = 'tasmaxCLIM,\n rsdsTREND'
    elif (variable == 'tasmax' and diag == 'STD'):
        text1 = '%s%s,\n prTREND' %(variable, diag)
    ax.text(0., .65, text1, size = 12, rotation = 90, transform = ax.transAxes)
if (len(diagnum) > 2):
    ax = plt.subplot(gs[4, 0], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if (variable == 'tasmax' and diag == 'CLIM'):
        text1 = 'tasmaxCLIM,\n rsdsTREND,\n prCLIM'
    elif (variable == 'tasmax' and diag == 'STD'):
        text1 = '%s%s,\n prTREND, prSTD' %(variable, diag)
    ax.text(-0.15, 0.55, text1, size = 12, rotation = 90,
            transform = ax.transAxes)
if (len(diagnum) > 3):
    ax = plt.subplot(gs[5, 0], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if (variable == 'tasmax') and (diag == 'CLIM'):
        text1 = 'tasmaxCLIM,\n rsdsTREND,\n prCLIM, tosSTD'
    elif (variable == 'tasmax') and (diag == 'STD'):
        text1 = '%s%s,\n prTREND, ,' %(variable, diag)
    ax.text(-0.15, 0.6, text1, size = 12, rotation = 90,
            transform = ax.transAxes)
if (len(diagnum) > 4):
    ax = plt.subplot(gs[6, 0], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if (variable == 'tasmax') and (diag == 'CLIM'):
        text1 = 'tasmaxCLIM,\n rsdsTREND, prCLIM,\n tosSTD, tasmaxSTD'
    elif (variable == 'tasmax') and (diag == 'STD'):
        text1 = '%s%s,\n prTREND, ,' %(variable, diag)
    ax.text(-0.15, 0.75, text1, size = 12, rotation = 90,
            transform = ax.transAxes)
if (len(diagnum) > 5):
    ax = plt.subplot(gs[7, 0], frameon = False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if (variable == 'tasmax') and (diag == 'CLIM'):
        text1 = 'tasmaxCLIM,\n rsdsTREND, prCLIM,\n tosSTD, tasmaxSTD, \n tasmaxTREND'
    elif (variable == 'tasmax') and (diag == 'STD'):
        text1 = '%s%s,\n prTREND, ,' %(variable, diag)
    ax.text(-0.2, 0.7, text1, size = 12, rotation = 90, transform=ax.transAxes)

pltno = 0
for row in range(1, nrow + 1):
    for col in range(1, ncol + 1):
        #print pltno
        ax = plt.subplot(gs[row, col])

        m = Basemap(projection = 'cyl', llcrnrlat = min_lat,
                    urcrnrlat = max_lat, llcrnrlon = min_lon, 
                    urcrnrlon = max_lon, resolution = 'c')
        m.drawcoastlines()
        if (pltno%ncol == 0):
            dpl = 1
        else:
            dpl = 0
        if (row == nrow):
            dml = 1
        else:
            dml = 0
        m.drawparallels(np.arange(-90., 91., 10.), labels = [dpl, 0, 0, 0])
        m.drawmeridians(np.arange(-180., 181., 20.), labels = [0, 0, 0, dml])

        diff[pltno], lonsnew = addcyclic(diff[pltno], lons)
        lons2d, lats2d = np.meshgrid(lonsnew, lats)
        x, y = m(lons2d, lats2d)

        cmap=plt.get_cmap(colbar)
        norm=mc.BoundaryNorm(levels, cmap.N)

        cs = ax.contourf(x, y, diff[pltno], levels, cmap = cmap, norm = norm,
                         extend = 'both')

        plt.title(titles[pltno], size = 15, ha = 'left', x = 0)
        pltno = pltno + 1

axC = fig.add_subplot(gs[-1, 1 :])
cbar = fig.colorbar(cs, ax = ax, cax = axC, orientation = 'horizontal')
cbar.ax.tick_params(labelsize = 15)
cbar.ax.set_title('[%s]' %unit)

gs.tight_layout(fig)
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(plotname + plottype, dpi = fig.dpi)
if (plottype == '.eps'):
    call("epstopdf " + plotname + plottype, shell = True)
    call("pdfcrop " + plotname + ".pdf " + plotname + ".pdf", shell = True)


