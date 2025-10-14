import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as regres

def load_test_data_whole_sim(this_collection, path_norm_metrics, verbose = 0, keep_pandas = False):
    """ This function will load in a whole dataset as an xarray dataframe, """
    """" unless otherwise specified using keep_pandas                      """
    fp_apply = path_norm_metrics + this_collection + '_not_yet_normalised.csv'
    df_sel = pd.read_csv(fp_apply)
    print('\033[1m' + 'You have loaded data to apply the nn to:' + '\033[0m')
    print(fp_apply)
    print('You have kept the whole dataset')
    # Convert imported data to xarray
    if keep_pandas == False:
        clean_df = df_sel.to_xarray()
        return clean_df
    else:
        return df_sel

def load_model_nn(exp_name, this_collection, seed_nb, mod_size = 'small', TS_opt = 'extrap', norm_method = 'std', verbose = 0, 
                 annual_f = ''):
    """ This function will load in a trained neural network """
    fp_model = path_model + 'model_nn_'+ mod_size + '_' +exp_name + '_' + this_collection + '_' + annual_f + \
                                    str(seed_nb).zfill(2) + '_' + TS_opt + '_' + norm_method + '.keras'
    model = keras.models.load_model(fp_model)
    if verbose == 1:
        print('\033[1m' + 'You have loaded a trained neural network:' + '\033[0m')
        print(fp_model)
    return model
    
def load_normalisation_metrics(this_collection, norm_method = 'std', verbose = 0):
    """ This function will load in the normalisation metrics associated with a trained model """
    fp_norm_metrics = path_norm_metrics + this_collection + '_metrics_norm.nc'
    norm_metrics_file = xr.open_dataset(fp_norm_metrics)
    if verbose > 0:
        print('\033[1m' + 'You have loaded normalisation metrics:' + '\033[0m')
        print(fp_norm_metrics)
    norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()
    return norm_metrics

def load_results_nn(this_collection_NN, this_collection_apply, path_model, load_ym_only = False,
                    exp_name = 'slope_front', mod_size = 'small', TS_opt = 'extrap', norm_method = 'std',
                    keep_all_runs = True, annual_only = False, annual_f = '', quick_check = False,
                    verbose = 0):
    """ If a trained neural network has already been applied to a dataset, """
    """ then this function will load in the results for that dataset       """
    if keep_all_runs == False:
        sims = 'just_mean'
    else:
        sims = 'all_sims'
    if annual_only == False:
        ano = ''
    else:
        ano = '_annual'
    ano = ano + '_special_lc_' + this_collection_apply
    fp_ref_pred = path_model + 'applied_nn' + '_' + mod_size + '_' + exp_name + '_' + this_collection_NN + '_' + annual_f + \
                                            sims + '_' + TS_opt + '_' + norm_method + ano + '.csv'
    if quick_check == True:
        df_ref_pred = pd.read_csv(fp_ref_pred, nrows = 5)
    else:
        df_ref_pred = pd.read_csv(fp_ref_pred)
    if verbose == 1:
        print('\033[1m' + 'You have loaded the results of applying the neural network to the test dataset:' + '\033[0m')
        print(fp_ref_pred)
        if quick_check == True:
            print('Warning, only first 5 rows loaded')
    for i in range(10):
        df_ref_pred['melt_'+str(i+1).zfill(2)] = df_ref_pred['melt_'+str(i+1).zfill(2)] * df_ref_pred.loc[:,'area']* 31536000 * 1/1e12
    df_ref_pred.loc[:,'melt_Gt_per_y'] = ((df_ref_pred.loc[:,'melt_m_ice_per_y'] * df_ref_pred.loc[:,'area']* 31536000) * 1/1e12)
    df_ref_pred.loc[:,'melt_pred_Gt'] = ((df_ref_pred.loc[:,'melt_pred_mean'] * df_ref_pred.loc[:,'area'] * 31536000) * 1/1e12)
    # Create an array with the 10 NN runs, and then calculate the std 
    a = []
    for i in range(10):
        a.append('melt_' + str(i+1).zfill(2))   
    df_ref_pred.loc[:,'std_Gt'] = df_ref_pred[a].std(axis = 1)
    if this_collection_apply == 'OPM026_whole_dataset':
        df_ref_pred = df_ref_pred[df_ref_pred.year > 1983]
    if verbose == 1:
        print('Calculated melt in Gt per yr')
    df_ymb_mean = df_ref_pred.groupby(['year','month','basin'], as_index = False).mean()
    df_ymb_sum = df_ref_pred.groupby(['year','month','basin'], as_index = False).sum()
    df_ymb_sum.loc[:,'std_Gt'] = df_ymb_sum[a].std(axis = 1)
    df_ym_mean = df_ref_pred.groupby(['year','month'], as_index = False).mean()
    df_ym_sum = df_ref_pred.groupby(['year','month'], as_index = False).sum()
    df_ym_sum.loc[:,'std_Gt'] = df_ym_sum[a].std(axis = 1)
    if verbose == 1:
        print('Calculated integrated volumes')        
    RMSE = np.sqrt(np.mean((df_ref_pred.melt_m_ice_per_y - df_ref_pred.melt_pred_mean)**2))/917*31536000
    if load_ym_only == True:
        return df_ym_sum
    else:
        return df_ref_pred, df_ymb_mean, df_ymb_sum, df_ym_mean, df_ym_sum, RMSE


# Linear regression model creation and fitting
def regression_analysis(ref_melt, pred_melt):
    """ This function will calculate the regression coefficient between two data arrays """
    ref_melt = ref_melt.values.reshape(-1,1)
    pred_melt = pred_melt.values.reshape(-1,1)
    model = regres()
    reg = model.fit(ref_melt, pred_melt)
    slope = reg.coef_[0][0] 
    intercept = reg.intercept_[0]
    rsq = reg.score(ref_melt, pred_melt)
    return slope, intercept, rsq

def ticks_labels(df):
    ticks = np.arange(np.min(np.unique(df.year)), np.min(np.unique(df.year)) + int(np.ceil(len(np.unique(df.year))/10))*10, 10)
    labels = np.arange(0, int(np.ceil(len(np.unique(df.year))/10))*10, 10)
    return ticks, labels

def plot_monthly_trend(df_z, color, label, ax, std = False, label_std = 'Need to specify label', flip = False):
    RMSE, R2 = RMSE_R2(df_z, flip)
    label2 = '\nRMSE : {:.0f} Gt/yr, R$^2$ : {:.2f}'.format(RMSE, R2)
    mean_std = np.mean(df_z.std_Gt.values)
    mean_std = np.round(mean_std/10**(np.floor(np.log10(mean_std))-1))*10**(np.floor(np.log10(mean_std))-1)
    if flip == True:
        ax.plot(df_z.year + df_z.month/12, df_z.melt_pred_Gt, color = color, label = label+label2)
        if std == True:
            ax.fill_between(df_z.year + df_z.month/12, df_z.melt_pred_Gt - df_z.std_Gt, \
                    df_z.melt_pred_Gt + df_z.std_Gt, color = color, alpha = 0.2, \
                            label = label_std + ', $\\sigma_{{mean}}$ : {:.0f} Gt/yr'.format(mean_std))
    else:    
        ax.plot(df_z.year + df_z.month/12, -df_z.melt_pred_Gt, color = color, label = label+label2)
        if std == True:
            ax.fill_between(df_z.year + df_z.month/12, -df_z.melt_pred_Gt - df_z.std_Gt, \
                    -df_z.melt_pred_Gt + df_z.std_Gt, color = color, alpha = 0.2, \
                            label = label_std + ', $\\sigma_{{mean}}$ : {:.0f} Gt/yr'.format(mean_std))      

def RMSE_R2(df_z, flip = False):
    if flip == False:
        RMSE = np.sqrt(np.mean((df_z.melt_Gt_per_y - df_z.melt_pred_Gt)**2))
        R2 = regression_analysis(df_z.melt_Gt_per_y, df_z.melt_pred_Gt)[2]
    elif flip == True:
        RMSE = np.sqrt(np.mean((df_z.melt_Gt_per_y + df_z.melt_pred_Gt)**2))
        R2 = regression_analysis(df_z.melt_Gt_per_y, -df_z.melt_pred_Gt)[2]
    elif flip == 'second':
        RMSE = np.sqrt(np.mean((-df_z.melt_Gt_per_y - df_z.melt_pred_Gt)**2))
        R2 = regression_analysis(-df_z.melt_Gt_per_y, df_z.melt_pred_Gt)[2]
    RMSE = np.round(RMSE/10**(np.floor(np.log10(RMSE))-1))*10**(np.floor(np.log10(RMSE))-1)
    return RMSE, R2

def std_range(df_z, color, position, total, ax, flip = False):
    fraction = (-1/(2*total))+(position/total)
    if flip == False:
        ax.scatter(fraction, -df_z.melt_pred_Gt.values[-1], color = color)
        ax.plot((fraction,fraction), \
               (-df_z.melt_pred_Gt.values[-1]+ np.mean(df_z.std_Gt), -df_z.melt_pred_Gt.values[-1]- np.mean(df_z.std_Gt)), 
               color = color)
    else:
        ax.scatter(fraction, df_z.melt_pred_Gt.values[-1], color = color)
        ax.plot((fraction,fraction), \
               (df_z.melt_pred_Gt.values[-1]+ np.mean(df_z.std_Gt), df_z.melt_pred_Gt.values[-1]- np.mean(df_z.std_Gt)), 
               color = color)
    ax.annotate('{:.0f} Gt/yr'.format(np.mean(df_z.std_Gt)), xytext = (fraction,0.2), xy = (0.5,0.5), xycoords = 'axes fraction', \
                rotation = 90, fontsize = 12, color = color)

def twosigfig(a):
    if a > 10:
        return '{:.0f}'.format(np.floor(a/(10 ** (np.floor(np.log10(a))-1))) *  (10 ** (np.floor(np.log10(a))-1)))
    elif (a>1) & (a <10):
        return '{:.1f}'.format(a)
    elif (a < 1):
        return '{:.2f}'.format(a)