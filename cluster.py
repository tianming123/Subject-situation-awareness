import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from bs4 import BeautifulSoup
import mpld3
from nltk.stem.snowball import SnowballStemmer
from nltk import *
from nltk.corpus import stopwords
from time import time
from sklearn.datasets import load_files
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt

class Filter(object):
    def __init__(self):
        self.main_heading = []
        self.controlled_terms = []
        self.uncontrolled_terms = []
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.clean_data = []

    def tokenize_and_stem(self,text):
        stemmer = SnowballStemmer("english")
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]

    def tokenize_only(self,text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens


class Utils(object):
    def __init__(self):
        pass
    def load_csv_data(self,path):
        csv_data = pd.read_csv(path,encoding='gbk')
        return csv_data
    def parse_data(self,path,columns):
        df = self.load_csv_data(path[0])
        df = pd.DataFrame(df,columns=[columns])
        data_list = np.array(df)
        return data_list
    def save_data_txt(self,data,path):
        with open(path,'a') as file:
            for i in range(len(data)):
                file.write(str(data[i][0])+'\r')
    def word_count(self,tokens):
        fdist = FreqDist(tokens)
        return fdist

class Cluster(object):
    def kmeans(self,path,k):
        print("loading documents ...")
        t = time()
        docs = load_files(path)
        print("summary: {0} documents in {1} categories.".format(len(docs.data), len(docs.target_names)))
        print("done in {0} seconds".format(time() - t))

        # ?????????????????????
        from sklearn.feature_extraction.text import TfidfVectorizer
        max_features = 20000
        print("vectorizing documents ...")
        t = time()
        vectorizer = TfidfVectorizer(max_df=0.4, min_df=2, max_features=max_features, encoding='latin-1')
        X = vectorizer.fit_transform((d for d in docs.data))

        print("n_samples: %d, n_features: %d" % X.shape)
        print("number of non-zero features in sample [{0}]: {1}".format(docs.filenames[0], X[0].getnnz()))
        print("done in {0} seconds".format(time() - t))

        # ????????????
        from sklearn.cluster import KMeans, MiniBatchKMeans
        print("clustering documents ...")
        t = time()
        n_clusters = k

        kmean = MiniBatchKMeans(n_clusters=n_clusters, max_iter=100, tol=0.01, verbose=1, n_init=3)
        kmean.fit(X)
        print("kmean: k={}, cost={}".format(n_clusters, int(kmean.inertia_)))
        print("done in {0} seconds".format(time() - t))
        # ??????????????????
        print("???????????? = ", len(kmean.labels_))
        # ????????????1000???1009?????????
        print("1000???1009?????????????????????", kmean.labels_[1000:1010])
        # ????????????1000???1009????????????
        print("1000???1009?????????????????????", docs.filenames[1000:1010])

        label_dict = {}
        for i in kmean.labels_:
            if i in label_dict:
                label_dict[i]+=1
            else:
                label_dict[i]=1
        print("????????????????????????????????????"+str(label_dict))
        # ?????????????????????10???????????????
        print("???????????????10??????????????????")
        order_centroids = kmean.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(n_clusters):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()


class DbscanClustering(object):

    def __init__(self):
        self.vectorizer= CountVectorizer()
        self.transformer = TfidfTransformer()
        #???????????????
        self.stopwords = list(set(stopwords.words('english')))
        #self.path = path
    def preprocess_data(self,path):
        # ???????????????
        result = []
        with open(path,'r',encoding='gbk') as f:
            for line in f.readlines():
                result.append(''.join([word for word in nltk.word_tokenize(line) if word not in self.stopwords]))
        print(result)
        return result
    def get_text_tfidf_matrix(self,corpus):
        #??????tfidf??????
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))
        #???????????????????????????
        words = self.vectorizer.get_feature_names()
        print(words)
        #??????tfidf???????????????
        weights = tfidf.toarray()
        return weights
    def pca(self,weight,n_components=2):
        #PCA???????????????
        pca = PCA(n_components=n_components)
        return pca.fit_transform(weight)
    def dbscan(self,corpus_path,eps=0.1,min_sample=3,fig=True):
        """
        Parameters
        ----------
        corpus_path????????????
        eps????????????
        min_sample ????????????????????????
        fig ?????????????????????????????????????????????
        Returns
        -------
        """
        corpus = self.preprocess_data(corpus_path)
        weight = self.get_text_tfidf_matrix(corpus)

        pca_weights = self.pca(weight)

        clf = DBSCAN(eps=eps,min_samples=min_sample)
        y=clf.fit_predict(pca_weights)
        if fig:
            plt.scatter(pca_weights[:,0],pca_weights[:,1],c=y)
            plt.show()
        #????????????????????????
        result = {}
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result

if __name__ == '__main__':

    cluster = Cluster()
    # path= 'data'
    # cluster.kmeans(path,5)
    dbscan = DbscanClustering()
    result = dbscan.dbscan(corpus_path='uncontrolled_terms.txt',eps=0.05,min_sample=3)
    print(result)
    for key in result.keys():
        print(key)
        print(len(result.get(key)))




