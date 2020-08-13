import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors


def parse_text(dict_path):
	section_text = {}
	class_text = {}
	subclass_text = {}
	other_text = {}
	with open(dict_path, 'r') as f:
		symbols_dict = json.load(f)
	for symbol in symbols_dict.keys():
		if len(symbol) == 1:
			section_text[symbol] = re.split('[ ;,]', symbols_dict[symbol].lower())
		elif len(symbol) == 3:
			class_text[symbol] = re.split('[ ;,]', symbols_dict[symbol].lower())
		elif len(symbol) == 4:
			subclass_text[symbol] = re.split('[ ;,]', symbols_dict[symbol].lower())
		else:
			subclass = symbol[0:4]
			if not subclass in other_text.keys():
				other_text[subclass] = []
			other_text[subclass] = other_text[subclass] + re.split('[ ;,]', symbols_dict[symbol].lower())
	package = {
        'section': section_text,
        'class': class_text,
        'subclass': subclass_text,
        'subc_other': other_text
    }
	return package

def get_subclass_text_embedding(package, K=30):
	# load model. It might take a while.
	glove_model = KeyedVectors.load_word2vec_format(
		"../pretrained_model/glove/glove.twitter.27B/glove.twitter.27B.25d.word2vec.txt", binary=False)

	embedding = {}
	corpus = []
	subclass_list = []
	section_text = package['section'],
	class_text = package['class'],
	subclass_text = package['subclass'],
	other_text = package['subc_other']
	for subclass in subclass_text[0].keys():
		text = subclass_text[0][subclass] + class_text[0][subclass[0:3]] + section_text[0][subclass[0]]
		text = ' '.join(text)
		if text.replace(' ', '') == '':
			embedding[subclass] = None
			continue
		corpus.append(text)
		subclass_list.append(subclass)

	# tf-idf
	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)).toarray()
	# sort index in descending tfidf order
	informative_index = np.argsort(tfidf, axis=1)[:, ::-1]
	# get all the words
	bag_of_words = vectorizer.get_feature_names()

	for i in tqdm(range(len(subclass_list))):
		embedding_num = 0
		subclass_embedding = []

		for informative_word_index in informative_index[i]:
			if not bag_of_words[informative_word_index] in glove_model.wv.vocab:
				continue

			subclass_embedding.append(glove_model[bag_of_words[informative_word_index]])
			embedding_num += 1

			if embedding_num == K:
				break

		subclass_embedding = np.array(subclass_embedding)
		embedding[subclass_list[i]] = subclass_embedding
	return embedding

if __name__ == '__main__':
	package = parse_text('../data/symbol2name.json')
	embedding = get_subclass_text_embedding(package, K=30)

