#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:54:07 2019

@author: Abhi Jouhal, Robert Utterback
"""

import pickle, os, sys, argparse, re, pathlib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

class CountAvgVectorizer(CountVectorizer):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_transform(articles):
        X = super().fit_transform(data).astype(np.float32)

        for i, article in enumerate(articles):
            word_count = len(article.split())
            X[i] /= word_count

        return X
    
class TextCategorizer:
    DEFAULTS = {
        # 'article_splitter': re.compile(r"^##$", re.MULTILINE),
        # 'metadata_splitter': re.compile(r"^\+\+$", re.MULTILINE),
        'article_splitter': re.compile(r"^Document \w+", re.MULTILINE),
        'metadata_splitter': re.compile(r"^English$", re.MULTILINE),
        'vectorizer': TfidfVectorizer,
        'cluster_alg': KMeans,
        'num_clusters': 3,
        'log': print,
    }

    def __init__(self, *,
                 article_splitter = DEFAULTS['article_splitter'],
                 metadata_splitter = DEFAULTS['metadata_splitter'],
                 vectorizer = DEFAULTS['vectorizer'],
                 cluster_alg = DEFAULTS['cluster_alg'],
                 num_clusters = DEFAULTS['num_clusters'],
                 log = DEFAULTS['log']):
        self.article_splitter = article_splitter
        self.metadata_splitter = metadata_splitter
        self.vectorizer = vectorizer()
        self.cluster_alg = cluster_alg
        self.num_clusters = num_clusters
        self.log = log

        self.articles = []
                
    def _split_metadata(self, full_txt):
        parts = re.split(self.metadata_splitter, full_txt)
        if len(parts) != 2:
            self.log(f"Found {len(parts)} pieces!")
            self.log(parts)
        assert len(parts) == 2, "\n\nMetadata splitting failed!"
        return parts

    def _get_body(self, full_txt):
        return self._split_metadata(full_txt)[1]

    def _split_articles(self, corpus):
        articles = re.split(self.article_splitter, corpus)[1:]
        if articles[-1].strip() == '':
            del(articles[-1])
        return articles

    def _read_file(self, path):
        """Reads articles from a file, removing metadata and returning a list of articles."""
        assert os.path.exists(path), f"{path} does not exist!"
        self.log(f"Reading articles from {path}...")

        self.log("\tLoading...", end='')
        with open(path, 'r', errors='ignore') as f:
            corpus = f.read()
        self.log("done.")

        self.log("\tSplitting...", end='')
        articles = self._split_articles(corpus)
        self.log(f"found {len(articles)} articles...", end='')
        bodies = [self._get_body(a) for a in articles]
        self.log("done.")

        return bodies

    def _read_files(self, paths):
        articles = []
        for path in paths:
            articles.extend(self._read_file(path))
        return articles

    def load_file(self, path):
        self.articles.append(self._read_file(path))

    def load_files(self, paths):
        self.articles.extend(self._read_files(paths))

    def _encode(self, articles):
        X = self.vectorizer.fit_transform(articles)
        return X, self.vectorizer.vocabulary_

    def _cluster(self, data):
        model = self.cluster_alg(n_clusters=self.num_clusters).fit(data)
        dist = model.transform(data)
        return model.labels_, model.cluster_centers_, dist

    def fit(self):
        self.log(f"Encoding articles with {self.vectorizer}...", end='')
        X, self.vocab = self._encode(self.articles)
        self.log("done.")
        
        self.log(f"Clustering...\n\tAlgorithm: {self.cluster_alg}\n\tNum clusters: {self.num_clusters}")
        self.labels, self.cluster_centers, self.dist = self._cluster(X)
        self.log("...done.")

    def nearest(self, k):
        self.log(f"Computing nearest {k} articles to each cluster center...", end='')
        partitioned = np.argpartition(self.dist, k, axis=0)
        nclusters = len(np.unique(self.labels))
        articles = np.array(self.articles)
        result = [articles[partitioned[:k,i]] for i in range(nclusters)]
        self.log('done.')
        return result

def main():
    ncluster = 3
    cat = TextCategorizer(num_clusters=ncluster)
    #paths = [f'../data/geopolitical/{x}.txt' for x in ['competition', 'influence', 'order']]
    paths = [f'../data/kashmir/{x}.txt' for x in ['india', 'pakistan']]
    cat.load_files(paths)
    cat.fit()

    k = 5
    near = cat.nearest(k)
    pathlib.Path("./output/").mkdir(parents=True, exist_ok=True)
    for i in range(ncluster):
      with open(f"output/cluster{i}.txt", 'w') as f:
        f.write(f"----- Cluster {i} -----\n")
        for j in range(k):
          f.write(near[i][j])
          f.write("\n -------------------- \n")

if __name__ == '__main__':
    main()
