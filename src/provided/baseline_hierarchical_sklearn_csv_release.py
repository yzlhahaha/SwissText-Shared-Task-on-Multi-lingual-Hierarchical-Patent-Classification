# save all data to data/ directory:
# Dev.csv
# patent_ABSTR.csv
# patent_description.csv
# patent_TITLE.csv
# Train.csv

import pickle
import html
import html2text
import os
from bs4 import BeautifulSoup
import json
import logging
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
#from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
#from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
#from keras.models import Model
#from keras.preprocessing import text, sequence



from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.pipeline import make_pipeline

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn.preprocessing import MultiLabelBinarizer  
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.multiclass import OneVsRestClassifier  
from sklearn.svm import LinearSVC

from utils import build_hierarchy, extend_hierarchy

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import csv
import sys
import re

csv.field_size_limit(sys.maxsize)

logging.basicConfig(filename='Baseline_results.log', level=logging.DEBUG)



def build_feature_extractor():
    context_features = FeatureUnion(
        transformer_list=[
            ('word', TfidfVectorizer(
                strip_accents=None,
                lowercase=True,
                analyzer='word',
                ngram_range=(1, 1),
                max_df=1.0,
                min_df=0.0,
                binary=False,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True,
                max_features=70000,
                stop_words='english'
            )),
        ]
    )

    features = FeatureUnion(
        transformer_list=[
            ('context', Pipeline(
                steps=[('vect', context_features)]
            )),
        ]
    )

    return features

def print_results(ids, preds, mlb, fname="submission.txt"):
    f1=open(fname, "w")
    for i1,pred in enumerate(preds):
        f1.write(str(ids[i1])+"\t"+"\t".join([mlb.classes_[tk] for tk in np.where(pred)[0]])+"\n")
    f1.close()
    return

#
def load_data():
    """
    read ids from csv extract and clean data from XML annotations.
    Assumes that data is in a directory named "data/".
    Needs following files: Train.csv, Dev.csv, patent_ABSTR.csv, patent_TITLE.csv, patent_description.csv
    """
    #read ids, labels
    prepath= "../../data/"
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
    files=[prepath+"patent_ABSTR.csv",prepath+"patent_TITLE.csv"] #,prepath+"patent_description.csv"
    for f1,fname in enumerate(files):
         with open(fname, 'r', encoding="utf-8") as f:
             reader = csv.reader(f, delimiter="\t")
             for i, line in enumerate(reader):
                 if i >= 1000: break
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

    x_train_data = [ x_train[tk] for tk  in train_ids if tk!="Labels" and tk in x_train.keys()]
    x_dev_data = [ x_dev[tk] for tk  in dev_ids  if tk != "Labels" and tk in x_dev.keys()]
    train_labels = [train_labels[i] for i in range(len(train_ids)) if train_ids[i] in x_train.keys()]

    return train_ids, x_train_data, train_labels, dev_ids, x_dev_data, dev_labels

# =================================  Loading dataset  ====================================

if "x_train_str" not in globals():
    train_ids, x_train_str, y_train_str, dev_ids, x_dev_str, y_dev_str = load_data()
print('After load data')
# =================================  Preparing Labels  ====================================

## check if labels are lists of lists
y_train_str_p = []


for lab_set, label_set_p in [(y_train_str,y_train_str_p)]:
    for lab in lab_set:
        if isinstance(lab,list):
            label_set_p.append(lab)
        else:
            label_set_p.append([lab])
print('Here 1')


## split labels in parent and child in the fashion: "G04D" -> "G","G04","G04D", important for submission
y_train_raw=[list(set(sum([[j[0],j[:3],j] for j in tk],[]))) for tk in y_train_str_p]


mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_raw)

print('Here 2')

print('train/val length: %d / %d ' %(len(x_train_str), len(x_dev_str)))
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train_str)), dtype=int)))
print('Average train label cardinality: {}'.format(
    np.mean(list(map(len, y_train_raw)), dtype=int)))

vectorizer = build_feature_extractor()

#build baseline from leaf labels e.g. ["G04D", "F02A"] -> ["G":"G04","G04":"G04D","F":"F02","F02":"F02A"]
hierarchy_f=build_hierarchy([tj for tk in  y_train_str_p for tj in tk])
if "ROOT" in hierarchy_f:
    hierarchy_f[ROOT] = hierarchy_f["ROOT"]
    del hierarchy_f["ROOT"]


class_hierarchy = extend_hierarchy(hierarchy_f, y_train_str)

bclf = OneVsRestClassifier(LinearSVC())

base_estimator = make_pipeline(
    vectorizer, bclf)


clf = HierarchicalClassifier(
    base_estimator=base_estimator,
    class_hierarchy=class_hierarchy,
    algorithm="lcn", training_strategy="siblings",
    #preprocessing=True,
    mlb=mlb,
    #use_decision_function=True
)

print("training classifier")
print(len(x_train_str), len(y_train))

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(x_train_str,y_train, random_state=42, test_size=0.2)
print('X_train_s.shape', len(X_train_s), len(X_train_s[0]))
for x in X_train_s:
    print(len(x))
sys.exit(1)
print('y_train_s.shape', len(y_train_s), y_train_s[0].shape)

clf.fit(X_train_s, y_train_s)
print("predicting")

y_pred_scores_s = clf.predict_proba(X_test_s)
    
y_pred_scores_s[np.where(y_pred_scores_s==0)]=-10
y_pred_s=y_pred_scores_s>-0.25
    


print('f1 micro:',
  f1_score(y_true=y_test_s, y_pred=y_pred_s[:,:y_test_s.shape[1]], average='micro'))
print('f1 macro:',
  f1_score(y_true=y_test_s, y_pred=y_pred_s[:,:y_test_s.shape[1]], average='macro'))
print(classification_report(y_true=y_test_s, y_pred=y_pred_s[:,:y_test_s.shape[1]]))

clf.fit(x_train_str, y_train)
print("predicting")

y_pred_scores = clf.predict_proba(x_dev_str)
    
y_pred_scores[np.where(y_pred_scores==0)]=-10
y_pred=y_pred_scores>-0.25
    


print_results(dev_ids, y_pred, mlb)
