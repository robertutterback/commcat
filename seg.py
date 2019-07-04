# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:34:04 2019

@author: Abhi
"""

#%%
import pickle

#%%

#segregates the given text file into individual articles 
def seg(file_name):
    file = open(file_name, "r")
    
    txt = file.read().split("##")
    
    file.close
    
    return txt

#%%

#K-means
#SVM
#MNB (multinomial naive bayes)
#