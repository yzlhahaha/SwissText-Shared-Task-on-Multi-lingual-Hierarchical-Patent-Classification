import numpy as np


def encode_label(label):
	'''Dead function. Ignore.'''
	assert(len(label) == 4)
	alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
	'P','Q','R','S','T','U','V','W','X','Y','Z']

	section = np.zeros((1,8))
	clas = np.zeros((1,100))
	subsection = np.zeros((1,26))
	section[0, alphabets.index(label[0])] = 1
	clas[0, int(label[1:3])] = 1
	subsection[0, alphabets.index(label[0])] = 1
	return np.squeeze(np.hstack([section, clas, subsection]))



def encode_label_hierarchical(label_list, all_labels):
	Y = []
	for labels in label_list:
		y = np.zeros((1, 1 + len(all_labels)))
		y[0,0] = 1
		for l in labels:
			y[0, all_labels.index(l[:1]) + 1] = 1 # assign section
			y[0, all_labels.index(l[:3]) + 1] = 1 # assign class
			y[0, all_labels.index(l) + 1] = 1     # assign subclass
		Y.append(y)

	Y = np.vstack(Y)

	return Y



def get_all_labels(label_list):
	labels = [l for labels in label_list for l in labels]
	subclasses = sorted(list(set(labels)))
	classes = list(set([x[:3] for x in subclasses]))
	sections = list(set([x[:1] for x in classes]))
	#print(len(sections), len(classes), len(subclasses))
	return sections, classes, subclasses




def construct_adjacency_matrix(sections, classes, subclasses):
	n = 1 + len(sections) + len(classes) + len(subclasses)
	A = np.zeros((n,n))
	# Connect sections to root
	A[1:1+len(sections),0] = 1
	for i in range(len(sections)):
		s = sections[i]
		for j in range(len(classes)):
			if classes[j][:1] == s: # j'th class belongs to section s
				A[1 + len(sections) + j, 1 + i] = 1


	for j in range(len(classes)):
		c = classes[j]
		for k in range(len(subclasses)):
			if subclasses[k][:3] == c: # k'th subclass belongs to class c
				A[1 + len(sections) + len(classes) + k, 1 + len(sections) + k] = 1

	return A

