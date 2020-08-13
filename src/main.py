import random 
import numpy as np

from get_data import load_data_small
from extract_features import extract_features
#from IPC_parser import *
from utils import construct_adjacency_matrix, encode_label_hierarchical, get_all_labels
from model import classify


   

class data_set:
	def __init__(self, X, Y, A):
		self.X = X
		self.Y = Y
		self.A = A


def construct_train_test_val_datasets(data, all_labels, A):
	N = len(data)
	indices = list(range(N))
	random.shuffle(indices)

	X = [data[k]['embedding'] for k in data.keys()]
	X = np.vstack(X)

	Y = [data[k]['labels'] for k in data.keys()]
	Y = encode_label_hierarchical(Y, all_labels)


	train = indices[:int(N*0.5)] # 70% data used for training
	test = indices[int(N*0.5) : int(N*0.8)] # 20% data used for testing
	val = indices[int(N*0.8):] # 10% data used for validation

	train = data_set(X[train, :], Y[train, :], A)
	test = data_set(X[test, :], Y[test, :], A)
	val = data_set(X[val, :], Y[val, :], A)

	return train, test, val






labeled_patent_data, unlabeled_patent_data = load_data_small(10000)
label_list = [labeled_patent_data[k]['labels'] for k in labeled_patent_data.keys()]
sections, classes, subclasses = get_all_labels(label_list) # returns A, B, .. | A01, A02, ..| A01B, A01C, ..
data = extract_features(labeled_patent_data, extractor = "tfidf+glove", K=300)

A = construct_adjacency_matrix(sections, classes, subclasses)

all_labels = sections + classes + subclasses
train, test, val = construct_train_test_val_datasets(data, all_labels, A)

results = classify(train, test, val, 300, [len(sections), len(classes), len(subclasses)]) # epoch num = 500





