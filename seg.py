#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:54:07 2019

@author: Abhi
"""

#%%
import pickle, os, sys, argparse, re
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

#%%

# Global config
ARTICLE_SPLITTER = re.compile(r"^##$", re.MULTILINE)
METADATA_SPLITTER = re.compile(r"^\+\+$", re.MULTILINE)
DATA_DIR = './data/geopolitical'
PICKLE_DIR = './.pickled'

def vprint(*args, **kwargs):
    if prog_args.verbose:
        print(*args, **kwargs)

def pickle_name(basename, stepname):
    return f"{PICKLE_DIR}/{basename}-{stepname}.pkl"

def split_metadata(full_txt):
    parts = re.split(METADATA_SPLITTER, article)
    assert len(parts) == 2, "\n\nMetadata splitting failed!"
    return parts

def get_body(full_txt):
    return split_metadata(full_txt)[1]

def split_articles(txt):
    full_articles = re.split(ARTICLE_SPLITTER, txt)[1:]
    return full_articles

# Reads articles from a file, removing metadata and returning a list
# of articles.
def load_new_file(basename):
    filename = f"{DATA_DIR}/{basename}.txt"
    vprint(f"Processing {filename}", end='')

    assert os.path.exists(filename), f"{filename} does not exist!"
    
    with open(filename, 'r', errors='ignore') as f:
        corpus = f.read()
        
    articles = metadata_remover_article_splitter(corpus)

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

def load_multiple(basenames):
    articles = []
    for basename in prog_args.basenames:
        articles.extend(load_file(basename))
    return articles

#%%
#CountVecotrizer

def cv_encoding(articles):
    vectorizer = CountVectorizer()
    # We convert to float so that we can divide
    X = vectorizer.fit_transform(articles).astype(np.float32)
    
    for i, article in enumerate(articles):
        wordCount = len(article.split())
        X[i] /= wordCount
    
    return X
    
 
#%%

#%%
#lemmatization from file
def lem(articles):
    lemma = np.empty(articles.len()) # ???
    lem = WordNetLemmatizer()
    lemma = [lem.lemmatize(a) for a in articles]
    return lemma    

#%%
#lemmatization from tokens
    

#%%
#Bag of words
#tf-idf
#word embedding

def tfidf_encoding(articles):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(articles).astype(np.float32)    
    return X

#%%
#word2vec / doc2vec

#%%
#GloVe

#%%
#K-Means

def kmeans(X, num_clusters=3):
    km = KMeans(n_clusters=num_clusters)
    model = km.fit(X)
    dist = km.transform(X)
    return model.labels_, model.cluster_centers_, dist

#%%

def points_near_centroid(X, centers, dist, labels):
    dist = dist ** 2

    df = pd.DataFrame(dist.sum(axis=1).round(2), columns=['sqdist'])
    df['label'] = labels

    df.head()

    return

#%%
#visualization

def visualize(X, labels, centers):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=.5)

    # TODO: Also plot the cluster centers
    
    n_clusters = len(np.unique(labels))
    plt.title(f"KMeans Clustering with {n_clusters} clusters, PCA")
    plt.show()
    return

#%%
#Main Function

def main(basenames):
    if not os.path.exists(PICKLE_DIR):
        os.mkdir(PICKLE_DIR)

    articles = load_multiple(basenames)
    X = cv_encoding(articles)
    labels, centers = kmeans(X)
    X = X.toarray()



    visualize(X, labels, centers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Print lots of information",
                        action="store_true", default=True)
    parser.add_argument("basenames", nargs='+')
    prog_args = parser.parse_args()
    main(prog_args.basenames)

#%%
#Naive-bayes    


#%%
#dbscan

#%%
