#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle, os, sys, argparse, re
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

# @TODO: the default stop words are not very good...
stop_words = text.ENGLISH_STOP_WORDS

# Global config
ARTICLE_SPLITTER = re.compile(r"^Document \w+$", re.MULTILINE)
METADATA_SPLITTER = re.compile(r"^English$", re.MULTILINE)
DATA_DIR = './data/kashmir'
PICKLE_DIR = './.pickled'
OUTPUT_DIR = './outputK'

def vprint(*args, **kwargs):
  if prog_args.verbose:
    print(*args, **kwargs)

def pickle_name(basename, stepname):
  return f"{PICKLE_DIR}/{basename}-{stepname}.pkl"

def split_metadata(full_article):
  parts = re.split(METADATA_SPLITTER, full_article)
  if len(parts) != 2:
    print(len(parts))
    print(parts)
    print(full_article)
  assert len(parts) == 2, "\n\nMetadata splitting failed!"
  return parts

def get_body(full_txt):
  return split_metadata(full_txt)[1]

def split_articles(corpus):
  articles = re.split(ARTICLE_SPLITTER, corpus)
  return articles[:-1]

# Reads articles from a file, removing metadata and returning a list
# of articles.
def load_new_file(basename):
  filename = f"{DATA_DIR}/{basename}.txt"
  vprint(f"Processing {filename}", end='')

  assert os.path.exists(filename), f"{filename} does not exist!"

  with open(filename, 'r', errors='ignore') as f:
    corpus = f.read()

  articles = split_articles(corpus)
  bodies = [get_body(a) for a in articles]

  with open(pickle_name(basename, 'split'), 'wb') as f:
    pickle.dump(bodies, f)

  return bodies

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

def load_multiple(*basenames):
  articles = []
  for basename in basenames:
    articles.extend(load_file(basename))
  return articles

def freq_encoding(articles):
  vectorizer = CountVectorizer(stop_words=stop_words)
  # We convert to float so that we can divide
  X = vectorizer.fit_transform(articles).astype(np.float32)

  for i, article in enumerate(articles):
    wordCount = len(article.split())
    X[i] /= wordCount

  return X, vectorizer.vocabulary_

def tfidf_encoding(articles):
  vectorizer = TfidfVectorizer(stop_words=stop_words)
  X = vectorizer.fit_transform(articles).astype(np.float32)
  return X, vectorizer.vocabulary_

def kmeans(X, num_clusters=3):
  km = KMeans(n_clusters=num_clusters)
  model = km.fit(X)
  dist = km.transform(X)
  return model.labels_, model.cluster_centers_, dist

def find_nearest(articles, dist, labels, n=5):
  partitioned = np.argpartition(dist, n, axis=0)
  k = len(np.unique(labels))
  articles = np.array(articles)
  return [articles[partitioned[:n,i]] for i in range(k)]

# encode should be an encoding (vectorizing) function
# k is number of clusters
# nclosest is the number of articles closest to each cluster center to find and print
def process_basename(basename, encode, k, nclosest):
  articles = load_multiple(basename)
  X, _ = encode(articles)
  labels, centers, dist = kmeans(X, k)
  X = X.toarray()

  nearest = find_nearest(articles, dist, labels, nclosest)
  encode_str = encode.__name__.split('_')[0]
  for i in range(k):
    with open(f"{OUTPUT_DIR}/{basename}-{encode_str}-cluster{i}.txt", 'w') as f:
      f.write(f"----- Cluster {i} -----\n")
      for j in range(nclosest):
        f.write(nearest[i][j])
        f.write("\n -------------------- \n")

def main(basenames):
  for dirname in [PICKLE_DIR, OUTPUT_DIR]:
    os.makedirs(dirname, exist_ok=True)

  k = 10
  encode = tfidf_encoding
  nclosest = 3

  for b in basenames:
    process_basename(b, encode, k, nclosest)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--verbose", help="Print lots of information",
                      action="store_true", default=True)
  parser.add_argument("basenames", nargs='+')
  prog_args = parser.parse_args()
  main(prog_args.basenames)
