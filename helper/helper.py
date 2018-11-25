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

