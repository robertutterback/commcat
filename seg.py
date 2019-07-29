#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:54:07 2019

@author: Abhi
"""

#%%
import pickle, os, sys, argparse, re
import numpy as np
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

def get_body(article):
    parts = re.split(METADATA_SPLITTER, article)
    assert len(parts) == 2, "\n\nArticle splitting failed!"
    return parts[1]

def slice(filename):
    with open(filename, 'r', errors='ignore') as f:
        full_articles = re.split(ARTICLE_SPLITTER, f.read())[1:]

    article_bodies = [get_body(a) for a in full_articles]

    return article_bodies

# Reads articles from a file, removing metadata and returning a list
# of articles.
def load_new_file(basename):
    filename = f"{DATA_DIR}/{basename}.txt"
    vprint(f"Processing {filename}", end='')

    assert os.path.exists(filename), f"{filename} does not exist!"
    
    article_bodies = slice(filename)

    with open(pickle_name(basename, 'split'), 'wb') as f:
        pickle.dump(article_bodies, f)

    return article_bodies
    # The files are weirdly encoded as CP1252. But using the 'codecs'
    # package here has trouble with the different line endings of
    # various OSes, as it doesn't automatically convert them. This
    # will work for now, there there will be lots of '?' inserted
    # where CP1252 characters don't translate to UTF-8. If all
    # developers switch to Linux we can switch back to using codecs.
    #  - codecs.open(filename, 'rb', encoding='cp1252') as f:
    
    """ with open(filename, 'r', errors='ignore') as f:
        full_articles = re.split(ARTICLE_SPLITTER, f.read())[1:]

    article_bodies = [get_body(a) for a in full_articles]

    with open(pickle_name(basename, 'split'), 'wb') as f:
        pickle.dump(article_bodies, f)

    return article_bodies
    """

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

def kmeans(X, num_clusters=2):
    km = KMeans(n_clusters=num_clusters)
    model = km.fit(X)
    return model.labels_, model.cluster_centers_

#%%
#visualization

def visualize(X, labels, centers):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=.5)

    # TODO: Also plot the cluster centers
    
    n_clusters = np.unique(labels)
    plt.title(f"KMeans Clustering with {n_clusters} clusters, PCA")
    plt.gca().set_aspect("equal")
    plt.figure()
    
    return

#%%
#Main Function

def main(basenames):
    if not os.path.exists(PICKLE_DIR):
        os.mkdir(PICKLE_DIR)

    articles = load_multiple(basenames)
    X = cv_encoding(articles)
    labels, centers = kmeans(X)
    X = X.todense()
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
