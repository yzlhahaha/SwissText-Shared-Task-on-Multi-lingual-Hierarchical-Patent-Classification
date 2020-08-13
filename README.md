# Hierarchical Multilabel Classification

After git clone https://github.com/XiaonanHu/Hierarchical_multilabel_classification.git, please create a data folder inside this directory and put Train.csv, Dev.csv, patent_TITLE.csv, patent_ABSTR.csv and patent_description.csv in the folder. 

- Data loader:  
	In get_data.py file within the src directory, you can run load_data_small function which get N number of patents. It outputs two lists labeled_patent_data and unlabeled_patent_data (roughly 7:3). 

- Data format description:  
     labeled_patent_data = {patent1_ID: patent1_value, patent2_ID: patent2_value, ...}    
     
     patent
     - key: patent_ID  
     - value : {'title':..., 'abstract':..., 'description':..., 'labels':\[label1, label2, ...\]}  
     (unlabeled patents do not have 'labels')

## extract embedding
**extract_features.py** is provided to extract embeddings for the patent.

### TF-IDF
TF-IDF is used to find the most informative words for each patent in its title, abstract and description.

### Glove
Now the pretrained GloVe model [1] is utilized to generate embeddings for the informative words.

1. Please download **glove.twitter.27B.zip** [here](https://nlp.stanford.edu/projects/glove/), unzip the file, and put the **glove.twitter.27B** folder into the **pretrained_model/glove** folder. 
2. Run **pretrained_model/glove/glove2word2vec.py**.
3. **extract_features.py** is provided to extract embeddings for the patent.  
	*  Since only the pretrained model is used now, some words are not in the dictionary, and these words are simply ignored for now.

### Usage: 
```
labeled_patent_data, unlabeled_patent_data = load_data_small(1000)
feature = extract_features(unlabeled_patent_data, extractor = "tfidf+glove", K=20)
```
* parameter **K** is the number of informative words we use.


### output format:  
{patent1_ID: patent1_feature, patent2_ID: patent2_feature, ...} 
* `patent_feature.shape` is `(number of words, 25)`

## Label text embedding
```
package = parse_text('../data/symbol2name.json')
embedding = get_subclass_text_embedding(package, K=30)
```
Use tf-idf to extract K most informative words in the description of each subclass, then utilize the pretrained GloVe model to generate embeddings.

Note: some subclasses don't have any text. There embedding is `None` for now.
	
	
Reference:  
[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf)
