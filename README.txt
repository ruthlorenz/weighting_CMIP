# README #

Functions used to apply weighting CMIP5 data.
Method used in Knutti et al. 2017, GRL, Lorenz et al. 2017, JGR (sub).

### What is this repository for? ###

* Weight CMIP5
* Version 0.1
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
1. Calculate diagnostics for further use (climatologies, trends etc.).

2. Choose diagnostics for further use, e.g based on correlations with target

3. Calculate delta matrix, e.g. RMSE between all models and all models and obs,
   can use func_calc_rmse.py, uses rmse_3D from calc_RMSE_obs_mod_3D

4. Calculate optimal sigmas, needs delta_matrix
   calculate which sigmas result in values in between 10-90% percentile
   start with range of sigmas, e.g:
   tmp = np.mean(delta_matrix)
   sigma_S2 = np.linspace(tmp - 0.9 * tmp, tmp + 0.9 * tmp, 41)  # array
   sigma_D2 = np.linspace(tmp - 0.9 * tmp, tmp + 0.9 * tmp, sigma_size)  # array
   w_u = calc_wu(delta_matrix, model_names, sigma_S2)
   w_q = calc_wq(delta_matrix, model_names, sigma_D2)
   tmp_wmm_avg = calc_weights_approx(w_u, w_q, model_names, cmip5_area_avg)
   test_perc = calc_inpercentile(tmp_wmm_avg['weights'],
				 np.array(cmip5_area_avg, dtype = float))
   -> choose sigmas based on test_perc (sigma_S2_end, sigma_D2_end)

5. use delta's and sigmas from above to calculate final weighted mean
   delta_u = delta_matrix_models_normalized_by_median_if_multiple_diagnostics
   delta_q = delta_matrix_obs_normalized_by_median_if_multiple_diagnostics
   d_target = dict_or_array_with_target_diagnostic_for_all_models_in_ensemble
   target_file: string with file of target diagnostics, 'clim', 'trend', 'std'
   wu_end = calc_wu(delta_u, model_names, sigma_S2_end)
   wq_end = calc_wq(delta_q, model_names, sigma_D2_end)
   approx_wmm = calc_weights_approx(wu_end, wq_end, model_names, d_target,
                                    var_file = target_file)

(6.) evaluate weighted mean using func_eval_wmm_nonwmm_error_indexI.py, needs
     weighted multi model mean, non-weighted multi-model mean,
     climatology for observational data, variability in observational data.
     All either as timeseries (area averaged) or 3D (time, latitude, longitude),
     if 3D latitude and longitude need to be given as well.
   

* Configuration
* Dependencies
CMIP5 data archive
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Ruth Lorenz: ruth.lorenz22@gmail.com
