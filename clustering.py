import pandas as pd
import numpy as np
import re, nltk        
from nltk.stem.porter import PorterStemmer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import *
from sklearn import cluster

test_data_df = pd.read_csv('test.txt',header = None ,delimiter="\n")

test_data_df.columns = ["Text"]

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = text.split(" ")
    #stems = stem_tokens(tokens, stemmer)
    #return stems
    return tokens
    
vectorizer = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 1000)
 
corpus_data_features = vectorizer.fit_transform(test_data_df.Text.tolist())
corpus_data_features_nd = corpus_data_features.toarray()

k_means = cluster.KMeans(n_clusters=1)
k_means.fit(corpus_data_features_nd)


order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(1):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print





