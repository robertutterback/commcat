# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:34:04 2019

@author: Abhi
"""

#%%
import pickle
from sklearn.feature_extraction.text import CountVectorizer
#%%

#segregates the given text file into individual articles 
def seg(file_name):
    file = open(file_name, "r")
    
    txt = file.read().split("##")
    
    file.close
    
    pickle.dump(foo, open(file_name+"-split.pkl", "wb"))
    
    return txt

#%%

#try:
#    foo = pickle.load(open("var.pickle", "rb"))
#except (OSError, IOError) as e:
#    foo = 3
#    pickle.dump(foo, open("var.pickle", "wb"))

#%%
#Tokenization
def token(file_name):
    
    try:
        foo = pickle.load(open(file_name+"-split.pkl", "rb"))
    except (OSError, IOError) as e:
        text = seg(file_name)
    
    for i in text:
        count_vect = CountVectorizer(stopwords='english')
        tokens = count_vect.fit_transform(text)
    
    pickle.dump(foo, open(file_name+"-split.pkl", "wb"))
    
    return tokens

#%%
#lemmatization
#def lem(file_name):

#%%
#Bag of words
#tf-idf

#%%
#word2vec / doc2vec

#%%
#GloVe

#%%
#K-means

#%%
#Naive-bayes    
        
        
        
        
        
        
        