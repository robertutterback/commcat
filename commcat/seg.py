# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:54:07 2019

@author: Abhi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:34:04 2019

@author: Abhi
"""

#%%
import pickle
import numpy
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#%%
#removing the metadata
def met(txt):
    for i in txt:
        txt[i] = txt[i].split("++")[1]
    
    return txt

#segregates the given text file into individual articles 
def seg(file_name):
    file = open(file_name, "r")
    
    txt = file.read().split("##")
                   
    file.close
    
    txt = met(txt)
    
    pickle.dump(txt, open(file_name+"-split.pkl", "wb"))
    
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
        text = pickle.load(open(file_name+"-split.pkl", "rb"))
    except (OSError, IOError):
        text = seg(file_name)
    
    tokens = numpy.empty(text.length())
    count_vect = CountVectorizer(stopwords='english')
    
    for i in text:
        tokens[i] = count_vect.fit_transform(text[i])
    
    pickle.dump(text, open(file_name+"-split.pkl", "wb"))
    
    return tokens

#%%
#lemmatization from file
def lem(file_name):
    
    try:
        text = pickle.load(open(file_name+"-split.pkl", "rb"))
    except (OSError, IOError):
        text = seg(file_name)
        
    lemma = numpy.empty(text.length())
    lem = WordNetLemmatizer()
    
    for i in text:
        lemma[i] = lem.lemmatize(text[i])
    
    pickle.dump(text, open(file_name+"-split.pkl", "wb"))
    
    return lemma    

#%%
#lemmatization from tokens
    

#%%
#Bag of words
#tf-idf
#word embedding


#%%
#word2vec / doc2vec

#%%
#GloVe

#%%
#K-means

def kmeans(file_name):
    
    text = token(file_name)
    
    km = KMeans(n_clusters=2, init = 'random')
    km.fit_predict(text)
    
    pca = PCA(n_components=2)
    text_pca = pca.fit_transform(text)
    plt.scatter(text_pca[:, 0], text_pca[:, 1], c=km.labels_, alpha=.5)
    plt.title("KMeans")
    plt.gca().set_aspect("equal")
    plt.figure()
    
    return

#%%
#Naive-bayes    


#%%
#dbscan