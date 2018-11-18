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

