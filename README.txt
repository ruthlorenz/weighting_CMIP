# README #

Scripts to evaluate CMIP5 data with multiple datasets.
Main folder contain scripts and functions to apply a simple weighting method
to CMIP5 data with the goal to decrease spread in projections. 

### What is this repository for? ###

* Evaluate CMIP5
* Version 0.1
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
1. Run calc_diag/calc_diag_cmip5_cdo.py and calc_diag/calc_diag_obs.py to
   calculate diagnostics for further use, e.g. 1980-2014, 2065-2099, and
   1950-2100 (for timeseries)
2. Calculate correlations (Figure 1) with
   Mult_Diag_Lin_Reg/mult_corr_all_diag_matrix_plot.py
   removes diagnostics not correlated with DeltaDiag and highly correlated diags
   (pre-selection), and calculates variance inflation factor.
   Saves .csv file with correlation and p-values for Table 1 in paper.
3. use most promising diagnostics in
   Mult_Diag_Lin_Reg/mult_diag_lin_reg_hist_fut_CMIP5_sklearn_cdo.py
   to determine optimum number of features to include and which ones those are
   based on highest R^2. Creates Figures 3 and 4.
   3.1 check with Mult_Diag_Lin_Reg/linear_reg_feature_selectionRFE_sklearn.py
       if machine learning methods come up with similar selection of diags,
       prints values for Table S2 and Table S3 in paper.
   3.2 check explained variance of selected diags and obtain classification
       report with Mult_Diag_Lin_Reg/
       mult_diag_lin_reg_classification_hist_fut_CMIP5_sklearn_cdo.py
4. use diagnostics from 3. in weight_mm_beyond_democracy_cmip5-ng_cdo.py
   4.1 calculate RMSE's with calc_perfmetric_for_weight_beyond_democracy_cdo.py
   4.2 estimate sigmas with calc_opt_sigmas_cdo.py
   4.3 use RMSE's and sigmas from above in
       weight_mm_beyond_democracy_delta_cdo.py
       saves netcdfs with data and single plots
       use scripts in panel_plots for panel figures (Figures 5 - 8)

* Configuration
* Dependencies
CMIP5 data archive, some utility functions (found in ../utils/)
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Ruth Lorenz: ruth.lorenz22@gmail.com
* Other community or team contact
