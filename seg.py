#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:54:07 2019

@author: Abhi
"""

#%%
import pickle, os, sys, argparse, re
import codecs # to decode the weird CP1252 files
import numpy
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#%%

# Global config
ARTICLE_SPLITTER = re.compile(r"^##$", re.MULTILINE)
METADATA_SPLITTER = re.compile(r"^\+\+$", re.MULTILINE)
DATA_DIR = './data/geopolitical'
PICKLE_DIR = './.pickled'
if not os.path.exists(PICKLE_DIR):
  os.mkdir(PICKLE_DIR)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="Print lots of information",
                    action="store_true", default=True)
parser.add_argument("basenames", nargs='+')
prog_args = parser.parse_args()

def vprint(*args, **kwargs):
  if prog_args.verbose:
    print(*args, **kwargs)

def pickle_name(basename, stepname):
  return f"{PICKLE_DIR}/{basename}-{stepname}.pkl"

def get_body(article):
  parts = re.split(METADATA_SPLITTER, article)
  assert len(parts) == 2, "\n\nArticle splitting failed!"
  return parts[1]
    
# Reads articles from a file, removing metadata and returning a list
# of articles.
def load_new_file(basename):
  filename = f"{DATA_DIR}/{basename}.txt"
  vprint(f"Processing {filename}", end='')

  assert os.path.exists(filename), f"{filename} does not exist!"

  with codecs.open(filename, 'rb', 'cp1252') as f:
    full_articles = re.split(ARTICLE_SPLITTER, f.read())[1:]

  article_bodies = [get_body(a) for a in full_articles]

  with open(pickle_name(basename, 'split'), 'wb') as f:
    pickle.dump(article_bodies, f)

  return article_bodies

def load_pickled(filename):
  vprint(f"Loading pickled data from {filename}", end='')
  with open(filename, 'rb') as f:
    data = pickle.load(f)
  assert data != None, "No data!"
  return data

# Check for pickled file and load, otherwise process from scratch.
def load_file(basename):
  filename = pickle_name(basename, 'split')
  if os.path.exists(filename): # load from pickled
    articles = load_pickled(filename)
  else: # have to process it
    articles = load_new_file(basename)

  vprint(f" ... found {len(articles)} articles.")
  return articles

#%%
#CountVecotrizer

def encoding(articles):
    
    count_vect = CountVectorizer()
    #add back stopwords
    X = count_vect.fit_transform(articles)
    
    for i, article in enumerate(articles):
        temp = article.split()
        wordCount = len(temp)
        
        X[i] = X[i]/wordCount
    
    return X
    
 
#%%
#Tokenization
def token(articles):
        
    tokens = numpy.empty(articles.len())
    count_vect = CountVectorizer(stop_words='english')
    #add back stopwords
        
    tokens = [count_vect.fit_transform(a) for a in articles]
    
    return tokens

#%%
#lemmatization from file
def lem(articles):
         
    lemma = numpy.empty(articles.len())
    lem = WordNetLemmatizer()
    
    lemma = [lem.lemmatize(a) for a in articles]
        
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
#K-Means

def kmeans(data):
    
    km = KMeans(n_clusters=2, init = 'random')
    X = km.fit_predict(data)
    
    return X

#%%
#visualization

def visualize(data, components = 2)    :
    
    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(data)
    plt.scatter(X_pca[:, 0], X_pca[:, 1],alpha=.5)
    plt.title("KMeans")
    plt.gca().set_aspect("equal")
    plt.figure()
    
    return

#%%
#Main Function

if __name__ == "__main__":
  for basename in prog_args.basenames:
    articles = load_file(basename)
    data = encoding(articles)
    X = kmeans(data)
    visualize(X)    

#%%
#Naive-bayes    


#%%
#dbscan

#%%
