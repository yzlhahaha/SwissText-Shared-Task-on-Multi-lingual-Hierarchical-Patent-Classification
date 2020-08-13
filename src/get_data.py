import pickle
import html
import html2text
import os
import json
import logging
import pickle
import random
import sys
import time
import warnings
import nltk
import csv
import re

import numpy as np

#from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
#from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
#from keras.models import Model
#from keras.preprocessing import text, sequence
import sklearn
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn.preprocessing import MultiLabelBinarizer  
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier  
from sklearn.svm import LinearSVC

from bs4 import BeautifulSoup
from utils import *


csv.field_size_limit(sys.maxsize)


def load_data_all():
    """
    Copied from baseline_hierarchical_sklearn_csv_release's load_data function

    read ids from csv extract and clean data from XML annotations.
    Assumes that data is in a directory named "data/".
    Needs following files: Train.csv, Dev.csv, patent_ABSTR.csv, patent_TITLE.csv, patent_description.csv
    """
    #read ids, labels
    prepath= "../data/"
    train_ids=[]
    train_labels=[]
    with open(prepath+"Train.csv", 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                labels =line[1:]
                train_ids.append(ids)
                train_labels.append(labels)
    dev_ids=[]
    dev_labels=[]
    with open(prepath+"Dev.csv", 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                dev_ids.append(ids)

    # read title, abstract, description for the ids
    x_train={}
    x_dev= {}
    TAG_RE = re.compile(r'<[^>]+>')
    files=[prepath+"patent_ABSTR.csv",prepath+"patent_TITLE.csv",prepath+"patent_description.csv"]
    for f1,fname in enumerate(files):
         with open(fname, 'r', encoding="utf-8") as f:
             reader = csv.reader(f, delimiter="\t")
             for i, line in enumerate(reader):
                 if i==0:
                     continue
                 ids = line[0]
                 for x in [(x_train,train_ids),(x_dev,dev_ids)]:
                     if ids in x[1]:
                         if f1<2:
                             if ids in x[0]:
                                 x[0][ids]+="\n\n "+html2text.html2text(html.unescape(line[1]))
                             else:
                                 x[0][ids]=html2text.html2text(html.unescape(line[1]))
                         else:
                             if ids in x[0]:
                                 x[0][ids]+="\n\n "+TAG_RE.sub('', line[1])
                             else:
                                 x[0][ids]=TAG_RE.sub('', line[1])

    x_train = [ x_train[tk] for tk  in train_ids if tk!="Labels"]
    x_dev = [ x_dev[tk] for tk  in dev_ids  if tk!="Labels"]
    
    return train_ids, x_train, train_labels, dev_ids, x_dev, dev_labels


def load_train_data():
    # copied from load_data in baseline_hierarchical_sklearn_csv_release
    # read ids, labels
    prepath= "../data/"
    train_ids=[]
    train_labels=[]
    with open(prepath+"Train.csv", 'r', encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                labels =line[1:]
                train_ids.append(ids)
                #l_list = []
                #for l in labels:
                #    l_encoded = encode_label(l)
                #    l_list.append(l_encoded)
                train_labels.append(labels)
                #train_labels.append(l_list)

    return train_ids, train_labels


'''
In train.csv, there are 
'''


def load_data_small(N):
    """
    Adapted from load_data_all.
    Only get the first N data to play with.

    The first 11 patents 
        '1237195', '1288943', '1293848', '1265194', 
        '1278129', '1273980', '1288742', '1237148',
        '1215710', '1211638', '1279927'
    in patent_TITLE.csv and patent_ABSTR.csv have no corresponding
    descriptions in patent_description.csv. Ignoring them for now. 
    """

    train_ids, train_labels = load_train_data() # load all the training labels



    patent_data = {}
    prepath= "../data/"


    # Get patent titles
    with open(prepath+"patent_TITLE.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i in range(12): next(reader) # skip the header and the first 11 items
        for i, line in enumerate(reader):
            if i > N: break # only read the first N items

            ID, title = line
            title = html2text.html2text(html.unescape(title)).split(';')
            # Some title has repeated items. We only take the first three ones, which are
            # in German, English and French.
            # For each item, we do some initial cleaning. 
            title = [re.sub(r'\n', ' ', x.rstrip()) for x in title if x != ''][:3] 
            patent_data[ID] = {}
            patent_data[ID]['title'] = title

    # Get patent abstracts
    with open(prepath+"patent_ABSTR.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i in range(12): next(reader) # skip the header and the first 11 items
        for i, line in enumerate(reader):
            if i > N: break # only read the first N items

            ID, abstract = line
            patent_data[ID]['abstract'] = html2text.html2text(html.unescape(abstract)).lstrip(';\n\n')

    # Get patent descriptions
    with open(prepath+"patent_description.csv", 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # only skip the header
        for i, line in enumerate(reader):
            # the ordering of labels in the patent_description file is different from the 
            # other two, we read more description data to make sure all the N patents
            # currently in patent_data dictionary have descriptions. 
            if i > 2 * N: break 

            ID, description = line
            if ID in patent_data.keys():
                des = html2text.html2text(html.unescape(description)).lstrip(';\n\n')
                des = re.sub('\n', ' ', re.sub('-\n', '-', des)).rstrip()
                patent_data[ID]['description'] = des 

    # Get patent labels
    for ID in patent_data.keys():
        if ID in train_ids:
            patent_data[ID]['labels'] = train_labels[train_ids.index(ID)]


    labeled_patent_data = {k:x for k, x in patent_data.items() if 'labels' in x.keys()}
    unlabeled_patent_data = {k:x for k, x in patent_data.items() if 'labels' not in x.keys()}

    return labeled_patent_data, unlabeled_patent_data



def get_train_test_data(labeled_data, percentage):
    N = len(labeled_data)
    data = list(labeled_data.items())
    test = random.sample(data, int(N * (1 - percentage)))
    train = {x[0]:x[1] for x in data if x not in test}
    test = {x[0]:x[1] for x in test}
    return train, test



#N = 50
#labeled_patent_data, unlabeled_patent_data = load_data_small(N)
#train, test = get_train_test_data(labeled_patent_data, 0.7)
