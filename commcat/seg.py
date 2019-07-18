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
DATA_DIR = '../data/geopolitical'
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
  if len(parts) != 2:
    print("\n\nArticle splitting failed!")
    print(article)
    sys.exit(1)
  return parts[1]
    
# Reads articles from a file, removing metadata and returning a list
# of articles.
def load_new_file(basename):
  filename = f"{DATA_DIR}/{basename}.txt"
  vprint(f"Processing {filename}", end='')

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
  if data == None:
    print(" ... No data! Quitting.")
    sys.exit(1)
  return data

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

def countVect(file_name):
    
    try:
        text = pickle.load(open(file_name+"-split.pkl", "rb"))
    except (OSError, IOError):
        text = seg(file_name)
    
    count_vect = CountVectorizer(stopwords='english')
    
    X = count_vect.fit_transform(text)
    
    for i, article in enumerate(text):
        temp = article.split()
        wordCount = len(temp)
        
        X[i] = X[i]/wordCount
        
    pickle.dump(text, open(file_name+"-split.pkl", "wb"))
    
    return X
    
 
#%%
#Tokenization
def token(file_name):
    
    try:
        text = pickle.load(open(file_name+"-split.pkl", "rb"))
    except (OSError, IOError):
        text = seg(file_name)
    
    tokens = numpy.empty(text.len())
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
        
    lemma = numpy.empty(text.len())
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
    
    X = countVect(file_name)
    
    km = KMeans(n_clusters=2, init = 'random')
    km.fit_predict(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=km.labels_, alpha=.5)
    plt.title("KMeans")
    plt.gca().set_aspect("equal")
    plt.figure()
    
    return

if __name__ == "__main__":
  for basename in prog_args.basenames:
    load_file(basename)


#%%
#Naive-bayes    


#%%
#dbscan

#%%
