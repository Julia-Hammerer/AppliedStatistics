#!/usr/bin/env python
# coding: utf-8

# # Helper functions
# 
# This file contains useful functions that can be used in multiple notebooks. 
# 
# Note: After adding a function, do not forget to save it as a **.py file**. 
# 
# 
# Authors: Julia Hammerer, Vanessa Mai <br>
# Last Change: 18.11.2018

# In[ ]:


import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

from scipy import stats
from scipy.stats import norm
from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, r2_score 
from scipy.stats import linregress
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.compat import lzip
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
# In[ ]:


# plots a confusion matrix
def plot_confusion_matrix(confmat):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
            s=confmat[i, j],va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')


# In[ ]:


def na_ratio_table(df):
    """ NA counts and percentage values for each dataframe column

    Args:
        df (pandas DataFrame): DataFrame object

    Returns:
        pandas DataFrame: original columns to rows, returns NA_COUNT and
            NA_RATIO_PERC columns
    """
    null_sum = df.isnull().sum()
    return pd.DataFrame({'NA_COUNT': null_sum,
                         'NA_RATIO_PERC': (null_sum / df.shape[0]) * 100})


# In[ ]:


def dist_quantile_plots(df, categories, cat_col='SalePrice',
                        res_col='SalePrice'):
    """ Grid of pairs of distribution plot and theoretical quantiles plot.

    Two plots for each category: distribution plot fitted to normal distribution
    and theoretical quantiles plot

    Args:
        df (pandas DataFrame): DataFrame object
        categories (list): list of categories to filter the data for plots
        cat_col (str): column name to filter the data by its unique values
            from categories to plot. 'MeasurementDescription' by default
        res_col (str): column name of values to plot. 'MeasurementResult'
            by default

    Returns:
        axes grid: grid of matplotlib pyplot seaborn distribution and
        theoretical quantiles subplots

    """
    c_categories = len(categories)

    with sns.axes_style('darkgrid'):
        fig, axes = plt.subplots(c_categories, 2,
                                 figsize=(18, 6 * c_categories))

    for i, cat in enumerate(categories):
        cat_data = df.loc[df[cat_col] == cat][res_col]
        (mean, std) = norm.fit(cat_data)
        ax_distplot = axes[i, 0]
        sns.distplot(cat_data, fit=norm, ax=ax_distplot)
        ax_distplot.legend([
            'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(
                mean, std)],
            loc='upper right', fontsize=12)
        ax_distplot.set_ylabel('Frequency', fontsize=14)
        ax_distplot.set_xlabel('MeasurementResult', fontsize=14)
        ax_distplot.set_title('{} - Distribution'.format(cat), fontsize=14)

        ax_probplot = axes[i, 1]
        stats.probplot(cat_data, plot=ax_probplot, rvalue=True)
        ax_probplot.set_title('{} - Probability Plot'.format(cat), fontsize=14)
        ax_probplot.set_ylabel('Ordered Values', fontsize=14)
        ax_probplot.set_xlabel('Theoretical quantiles', fontsize=14)

    return fig.show()


# In[ ]:


def corr_heatmap(df, vmax=1.0, center=0.2, figsize=(16, 12)):
    """ Correlation heatmap.

    Correlation heatmap triangle. Duplications and diagonal values removed.

    Args:
        df (pandas DataFrame): DataFrame object with correlations
        vmax (float): maximum value of correlation for legend, 1.0 by default
        center (float): center value for correlation pallete of colors,
            0.2 by default
        figsize (tuple): figure size, width and height, (18, 12) by default

    Returns:
        Correlation heatmap triangle

    """
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style('white'):
        _, _ = plt.subplots(figsize=figsize)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df, mask=mask, cmap=cmap, vmax=vmax, center=center, annot=True,
                fmt='0.2f',
                square=True, linewidth=.5, cbar_kws={"shrink": .5})
    plt.ylabel(df.index.name, fontsize=16)
    plt.xlabel(df.index.name, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


# In[ ]:


def _corr_func_1(x, y, **kwargs):
    nas = np.logical_or(x.isna(), y.isna())
    corr_r, p_value = pearsonr(x[~nas], y[~nas])
    ax = plt.gca()
    ax.annotate('r = {:.2f} '.format(corr_r),
                xy=(0.05, 0.9), xycoords=ax.transAxes)
    ax.annotate('p = {:.2f} '.format(p_value),
                xy=(0.05, 0.8), xycoords=ax.transAxes)


# In[ ]:


def corr_matrix_1(df, fontScale=1):
    """ Linear regressions and bivariate distributions matrix.

    Correlation Matrix - Version 1:
    on the left - bivariate kernel density estimates
    diagonal - estimated measurement distributions
    on the right - scatter plot and simple linear regression with confidence
        intervals, pearson correlations, and p-values

    Args:
        df (pandas DataFrame): DataFrame object
    Returns:
        PairGrid figure: grid of plots

    """
    sns.set(style='white', font_scale=fontScale)
    fig = sns.PairGrid(df, palette=['red'], diag_sharey=False)
    fig.map_upper(sns.regplot, line_kws={"color": "black"})
    fig.map_upper(_corr_func_1)
    fig.map_diag(sns.kdeplot, lw=3)
    fig.map_lower(sns.kdeplot, map='Reds_d')
    return fig


def rmsle(ypred, ytest) : 
    '''Return rmsle'''

    assert len(ytest) == len(ypred)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(ytest))**2))

# define function for the baseline
def  get_baseline_for_regression(X_train, y_train, X_test, y_test):
    Baseline = DummyRegressor(strategy='mean')
    Baseline.fit(X_train, y_train)
    baseline = Baseline.predict(X_test)[0]
    mse = round(mean_squared_error(Baseline.predict(X_test), y_test), 3)
    bl_rmsle = round(rmsle(Baseline.predict(X_test), y_test),3)
    
    return baseline, mse, bl_rmsle

def regression_model_metrics(y, y_predicted, X):
    ''' return mse r2 and standard error'''
    if isinstance(y_predicted, pd.Series):
        y_predicted=y_predicted.values
    if isinstance(y, pd.Series):
        y=y.values
    
    mse = round(mean_squared_error(y, y_predicted), 3)
    r2 = r2_score(y, y_predicted)
    adj_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    standard_error = np.std(y - y_predicted)
    
    return mse, r2, adj_r2, standard_error

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    std_err = results.bse
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "std_err":std_err,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff", "std_err","pvals","conf_lower","conf_higher"]]
    results_df = results_df.reset_index()
    results_df = results_df.rename(index=str, columns={"index":"attributes"})
    
    return results_df

# define function to handle outliers, values higher than 3 standard deviations
def outliers_idx(df, y_measurement, n_std=3):
    upper = df[y_measurement].mean() + df[y_measurement].std() * 3
    lower = df[y_measurement].mean() - df[y_measurement].std() * 3
    
    return df.loc[(df[y_measurement] < lower) | (df[y_measurement] > upper)].index


def lin_model(y_train, X_train, y_test, X_test, reg_model="OLS"):
    ''' builds a specified linear regression, including a iterative way of removing insignificant attributes'''
    model_function=getattr(sm, reg_model)
    
    model_summaries={}
    y_pred_name = "saleprice_predicted"
    model = model_function(y_train, X_train).fit()
    
    model_sign_attr=[]
    number_sign_attr=-1;
    stop=False
    while (results_summary_to_dataframe(model)["pvals"].apply(lambda x: x <= 0.05).any())&(stop==False):

        if  len(model_sign_attr)==number_sign_attr:
            stop=True    
        else:
            number_sign_attr=len(model_sign_attr)

        model_sign_attr=[]
        for i, e in enumerate(results_summary_to_dataframe(model)["pvals"]):
            if e <=0.05:            
                attr=results_summary_to_dataframe(model).iloc[i]["attributes"]
                model_sign_attr.append(attr)
        if len(model_sign_attr)!=0:
            model = model_function(y_train, X_train[model_sign_attr]).fit()
        else:
            model = model_function(y_train, X_train).fit()
            model_sign_attr=list(X_train)
            stop=True
    
    if len(model_sign_attr)==0:
        model_sign_attr=list(X_train)

        # Test results
    reg_test = model.predict(X_test[model_sign_attr])

    # get results
    reg_train_results_df = pd.DataFrame({"saleprice": y_train.copy()})
    reg_train_results_df[y_pred_name] = model.fittedvalues
    reg_train_results_df['set'] = 'TRAIN'

    reg_test_results_df = pd.DataFrame({"saleprice": y_test.copy()})
    reg_test_results_df[y_pred_name] = reg_test
    reg_test_results_df['set'] = 'TEST'

    # metrics
    baseline_val, baseline_mse, baseline_rmsle = get_baseline_for_regression(
        X_train=X_train,
        y_train=reg_train_results_df["saleprice"],
        X_test=X_test,
        y_test=reg_test_results_df["saleprice"])

    mse_train, r2_train, adj_r2_train, st_error_train = regression_model_metrics(
        y=reg_train_results_df["saleprice"],
        y_predicted=reg_train_results_df[y_pred_name],
        X=X_train)

    mse_test, r2_test, adj_r2_test, st_error_test = regression_model_metrics(
        y=reg_test_results_df["saleprice"],
        y_predicted=reg_test_results_df[y_pred_name],
        X=X_test)
    
    rmsle_train=rmsle(ytest=reg_train_results_df["saleprice"], 
                      ypred=reg_train_results_df[y_pred_name])
    rmsle_test=rmsle(ytest=reg_test_results_df["saleprice"], 
                     ypred=reg_test_results_df[y_pred_name])


    model_summaries = {'y_pred_name': y_pred_name,
                           'train_res_df': reg_train_results_df,
                           'test_res_df': reg_test_results_df,
                           'baseline_mse': baseline_mse,
                           'baseline_rmsle': baseline_rmsle,
                           'baseline_val': baseline_val,
                           'mse_train': mse_train,
                           'mse_test': mse_test,
                           'rmsle_train': rmsle_train,
                           'rmsle_test':rmsle_test,
                           'r2_train': r2_train,
                           'adj_r2_train': adj_r2_train,
                           'r2_test': r2_test,
                           'adj_r2_test': adj_r2_test,
                           'se_train': st_error_train,
                           'se_test': st_error_test,
                           'model_report': model.summary(),
                           'train_obs': reg_train_results_df.shape[0],
                           'test_obs': reg_test_results_df.shape[0]}
    return model, model_summaries, model_sign_attr

def get_model_results(model,y_train, X_train, y_test, X_test):
    
    model_summaries={}
    reg_test = model.predict(X_test)
    y_pred_name = "saleprice_predicted"
    
    # get results
    reg_train_results_df = pd.DataFrame({"saleprice": y_train.copy()})
    reg_train_results_df[y_pred_name] = model.predict(X_train)
    reg_train_results_df['set'] = 'TRAIN'

    reg_test_results_df = pd.DataFrame({"saleprice": y_test.copy()})
    reg_test_results_df[y_pred_name] = reg_test
    reg_test_results_df['set'] = 'TEST'

    # metrics
    baseline_val, baseline_mse, baseline_rmsle = get_baseline_for_regression(
        X_train=X_train,
        y_train=reg_train_results_df["saleprice"],
        X_test=X_test,
        y_test=reg_test_results_df["saleprice"])

    mse_train, r2_train, adj_r2_train, st_error_train = regression_model_metrics(
        y=reg_train_results_df["saleprice"],
        y_predicted=reg_train_results_df[y_pred_name],
        X=X_train)

    mse_test, r2_test, adj_r2_test, st_error_test = regression_model_metrics(
        y=reg_test_results_df["saleprice"],
        y_predicted=reg_test_results_df[y_pred_name],
        X=X_test)
    
    
    rmsle_train=rmsle(ytest=reg_train_results_df["saleprice"], 
                      ypred=reg_train_results_df[y_pred_name])
    rmsle_test=rmsle(ytest=reg_test_results_df["saleprice"], 
                     ypred=reg_test_results_df[y_pred_name])

    model_summaries = {'y_pred_name': y_pred_name,
                           'train_res_df': reg_train_results_df,
                           'test_res_df': reg_test_results_df,
                           'baseline_mse': baseline_mse,
                           'baseline_rmsle': baseline_rmsle,
                           'baseline_val': baseline_val,
                           'mse_train': mse_train,
                           'mse_test': mse_test,                      
                           'rmsle_train': rmsle_train,
                           'rmsle_test':rmsle_test,
                           'r2_train': r2_train,
                           'adj_r2_train': adj_r2_train,
                           'r2_test': r2_test,
                           'adj_r2_test': adj_r2_test,
                           'se_train': st_error_train,
                           'se_test': st_error_test,
#                            'model_report': model.summary(),
                           'train_obs': reg_train_results_df.shape[0],
                           'test_obs': reg_test_results_df.shape[0]}
    return model, model_summaries

def reg_model_results_plots(model_summaries):
                           
#     c_measurements = len(model_summaries.keys())
    
    with plt.style.context(('ggplot')):

        fig, axes = plt.subplots(1, 2,
                                 figsize=(20, 7))


    y_pred_name = model_summaries['y_pred_name']
    train_res_df = model_summaries['train_res_df']
    test_res_df = model_summaries['test_res_df']
    baseline_mse = model_summaries['baseline_mse']    
    baseline_rmsle = model_summaries['baseline_rmsle']
    baseline_val = model_summaries['baseline_val']
    mse_train = model_summaries['mse_train']                     
    mse_test = model_summaries['mse_test']
    r2_train = model_summaries['r2_train']
    adj_r2_train=model_summaries['adj_r2_train']
    adj_r2_test=model_summaries['adj_r2_test']
    rmsle_train = model_summaries['rmsle_train']                     
    rmsle_test = model_summaries['rmsle_test']
    r2_test = model_summaries['r2_test']
    se_train = model_summaries['se_train']
    se_test = model_summaries['se_test']
    train_obs = model_summaries['train_obs']
    test_obs = model_summaries['test_obs']

    line = reg_line(train_res_df["saleprice"],
                    train_res_df[y_pred_name])

    ax_1 = axes[0]
    ax_2 = axes[1]

    ax_1.scatter(train_res_df["saleprice"], 
                train_res_df[y_pred_name],
                c='blue', edgecolors='k', label='train')
    ax_1.legend(loc='upper left')

    ax_1.scatter(test_res_df["saleprice"], 
                test_res_df[y_pred_name],
                c='red', edgecolors='k', label='test')
    ax_1.legend(loc='upper left', fontsize=12)

    ax_1.plot(train_res_df["saleprice"], line, c='green', linewidth=1, label='reg line')
    ax_1.legend(loc='upper left', fontsize=12)
    
    ax_1.axhline(baseline_val, c='black', linewidth=1, linestyle='--', label='baseline')
    ax_1.legend(loc='upper left', fontsize=12)

    ax_1.set_title('{}: Actual vs. Predicted'.format("saleprice"), fontsize=16)
    ax_1.set_xlabel('Actual', fontsize=14)
    ax_1.set_ylabel('Predicted', fontsize=14)
    ax_1.tick_params(axis='both', labelsize=12)


    # add metrics info
    ax_1.annotate('MSE$_{baseline}:$ %5.3f' % baseline_mse + '\n' + \
                  'RMSLE$_{baseline}:$ %5.3f' % baseline_rmsle + '\n' + \
                  'MSE$_{train}:$ %5.3f' % mse_train + '\n' + \
                  'MSE$_{test}:$ %5.3f' % mse_test + '\n' + \
                  'RMSLE$_{train}:$ %5.3f' % rmsle_train + '\n' + \
                  'RMSLE$_{test}:$ %5.3f' % rmsle_test + '\n' + \
                  'R$^{2}_{train}:$ %5.3f' % r2_train + '\n' + \
                  'adj. R$^{2}_{train}:$ %5.3f' % adj_r2_train + '\n' + \
                  'R$^{2}_{test}:$ %5.3f' % r2_test + '\n' + \
                  'adj. R$^{2}_{test}:$ %5.3f' % adj_r2_test + '\n' + \
                  'SE$_{train}:$ %5.3f' % se_train + '\n' + \
                  'SE$_{test}:$ %5.3f' % se_test + '\n' + \
                  'N_OBS$_{train}:$ %d' % train_obs + '\n' + \
                  'N_OBS$_{test}:$ %d' % test_obs ,
                  xy=(0, 0.5),
                  xytext=(-ax_1.yaxis.labelpad - 5, 0),
                  xycoords=ax_1.yaxis.label,
                  textcoords='offset points',
                  fontsize=16,
                  weight='bold',
                  ha='right',
                  va='center')

    # results box plot per set    
    medianprops = dict(linestyle='-',
                       linewidth=2.5,
                       color='firebrick')

    bplot = ax_2.boxplot([train_res_df["saleprice"].values, 
                        train_res_df[y_pred_name].values,
                        test_res_df["saleprice"].values,
                        test_res_df[y_pred_name].values],
                        patch_artist=True, 
                        medianprops=medianprops)
    ax_2.set_title('{}: Sets Statistics'.format("saleprice"), fontsize=16)

    colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(ax_2, xticklabels=['Train Actual', 'Train Predicted', 'Test Actual', 'Test Predicted'])

    ax_2.set_xlabel('Set', fontsize=14)
    ax_2.set_ylabel('Values', fontsize=14)
    ax_2.tick_params(axis='both', labelsize=12)
    
    return fig.show()

def reg_line(y, y_predicted):
    slope, intercept, _, _, _ = linregress(y, y_predicted)
    return slope * y + intercept

def resid_plot(model, y_train):
    model_fitted_y = model.fittedvalues;
    # model residuals
    model_residuals = model.resid
    # normalized residuals
    model_norm_residuals = model.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    
    # figure size
    plot = plt.figure(1)
    plot.set_figheight(8)
    plot.set_figwidth(12)
    plot.axes[0]= sns.residplot(model_fitted_y, y_train, 
                                lowess=True, 
                                scatter_kws={'alpha': 0.5}, 
                                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    # label axes
    plot.axes[0].set_title('Residuals vs Fitted')
    plot.axes[0].set_xlabel('Fitted values')
    plot.axes[0].set_ylabel('Residuals')
    
    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    
    for i in abs_resid_top_3.index:
        plot.axes[0].annotate(i,
                              xy=(model_fitted_y[i], 
                                  model_residuals[i]));
  
    
    
    
#     # Basic plot
#     plot = sns.residplot(model_fitted_y, y_train, lowess=True, 
#                          scatter_kws={'alpha': 0.5}, 
#                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

#     plot.set_title('Residuals vs Fitted');
#     plot.set_xlabel('Fitted values');
#     plot.set_ylabel('Residuals');



# by Jan Kirenz
def cross_fold(data, response, predictors):
    mse = []
    rmsle=[]
    kf = KFold(n_splits=10, shuffle=True)
    feat = list(data.columns)
    feat.remove(response)
    for train_index, test_index in kf.split(data):
        # Train
        lm = smf.ols(response + ' ~ ' + predictors, data.iloc[train_index]).fit()
        # Test
        yp = lm.predict(data.iloc[test_index][feat])
        # Calc MSE of fold
        y_true=data.iloc[test_index][response]
        resid = data.iloc[test_index][response] - yp
        rs = resid**2
        n = len(resid)
        MSE = rs.sum() / n
        
        RMSLE= np.sqrt(np.mean((np.log1p(yp) - np.log1p(y_true))**2))
        
        

        mse.append([len(mse), MSE]) 
        rmsle.append(RMSLE) 
    df = pd.DataFrame(mse)
    rmsle = pd.Series(rmsle)
    df.columns = ['Fold', 'MSE']
    df['RMSLE']=rmsle.values
    return df