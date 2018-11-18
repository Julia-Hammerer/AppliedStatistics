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

# In[1]:


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

