import random
import pandas as pd
from tqdm import tqdm
# from ..common.tools import save_pickle
# from ..common.tools import logger
# from ..callback.progressbar import ProgressBar
from pybert.common.tools import save_pickle
from pybert.common.tools import logger
from pybert.callback.progressbar import ProgressBar
from pybert.configs.basic_config import config
from pybert.preprocessing.preprocessor import EnglishPreProcessor
import html
import html2text
from pybert.configs.basic_config import config
import json
import csv
import sys
import re
import numpy as np

csv.field_size_limit(sys.maxsize)

class TaskData(object):
    def __init__(self):
        pass
    def save_pickle(self,X, y,data_name = None,data_dir = None, is_train=True):
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        data = []
        for step,(data_x, data_y) in enumerate(zip(X, y)):
            data.append((data_x, data_y))
            pbar(step=step)
        del X, y
        if is_train:
            save_path = data_dir / f"{data_name}.train.pkl"
        else:
            save_path = data_dir / f"{data_name}.valid.pkl"
        save_pickle(data=data,file_path=save_path)
        return data

    # def train_val_split(self,X, y,valid_size,stratify=False,shuffle=True,save = True,
    #                     seed = None,data_name = None,data_dir = None):
    #     pbar = ProgressBar(n_total=len(X),desc='bucket')
    #     logger.info('split raw data into train and valid')
    #     if stratify:
    #         num_classes = len(list(set(y)))
    #         train, valid = [], []
    #         bucket = [[] for _ in range(num_classes)]
    #         for step,(data_x, data_y) in enumerate(zip(X, y)):
    #             bucket[int(data_y)].append((data_x, data_y))
    #             pbar(step=step)
    #         del X, y
    #         for bt in tqdm(bucket, desc='split'):
    #             N = len(bt)
    #             if N == 0:
    #                 continue
    #             test_size = int(N * valid_size)
    #             if shuffle:
    #                 random.seed(seed)
    #                 random.shuffle(bt)
    #             valid.extend(bt[:test_size])
    #             train.extend(bt[test_size:])
    #         if shuffle:
    #             random.seed(seed)
    #             random.shuffle(train)
    #     else:
    #         data = []
    #         for step,(data_x, data_y) in enumerate(zip(X, y)):
    #             data.append((data_x, data_y))
    #             pbar(step=step)
    #         del X, y
    #         N = len(data)
    #         test_size = int(N * valid_size)
    #         if shuffle:
    #             random.seed(seed)
    #             random.shuffle(data)
    #         valid = data[:test_size]
    #         train = data[test_size:]
    #         # 混洗train数据集
    #         if shuffle:
    #             random.seed(seed)
    #             random.shuffle(train)
    #     if save:
    #         train_path = data_dir / f"{data_name}.train.pkl"
    #         valid_path = data_dir / f"{data_name}.valid.pkl"
    #         save_pickle(data=train,file_path=train_path)
    #         save_pickle(data = valid,file_path=valid_path)
    #     return train, valid
    #
    # def read_data(self,raw_data_path,preprocessor = None,is_train=True):
    #     '''
    #     :param raw_data_path:
    #     :param skip_header:
    #     :param preprocessor:
    #     :return:
    #     '''
    #     targets, sentences = [], []
    #     data = pd.read_csv(raw_data_path)
    #     for row in data.values:
    #         if is_train:
    #             target = row[2:]
    #         else:
    #             target = [-1,-1,-1,-1,-1,-1]
    #         sentence = str(row[1])
    #         if preprocessor:
    #             sentence = preprocessor(sentence)
    #         if sentence:
    #             targets.append(target)
    #             sentences.append(sentence)
    #     return targets,sentences


    def read_data(self, config, raw_data_path, preprocessor = None, is_train=False):
        '''
        :param config
        :param raw_data_path:
        :param preprocessor:
        :return:
        '''
        train_targets, train_sentences = [], []
        val_targets, val_sentences = [], []

        with open(config['subclass_list']) as f:
            subclass_list = json.load(f)

        with open(raw_data_path, 'r') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                # is_train = line[1]
                ID = line[0]
                title = line[1]
                title = html2text.html2text(html.unescape(title)).split(';')
                title = [re.sub(r'\n', ' ', x.rstrip()) for x in title if x != ''][:3]
                title = ';'.join(title)
                abstract = line[2]
                abstract = html2text.html2text(html.unescape(abstract)).lstrip(';\n\n')
                description = line[3]
                description = html2text.html2text(html.unescape(description)).lstrip(';\n\n')
                input = title + ';' + abstract + ';' + description
                labels = line[4:]
                onehot_label = np.zeros(645)
                for label in labels:
                    for i, subclass in enumerate(subclass_list):
                        if subclass == label:
                            onehot_label[i] = 1
                            break
                if preprocessor:
                    input = preprocessor(input)
                if is_train:
                    train_sentences.append(input)
                    train_targets.append(onehot_label)
                else:
                    val_sentences.append(input)
                    val_targets.append(onehot_label)
        return train_targets, train_sentences, val_targets, val_sentences

if __name__ == "__main__":
    data = TaskData()
    train_targets, train_sentences, val_targets, val_sentences = data.read_data(config,
                                                                                raw_data_path="/Users/xiaohan/Desktop/bert_HMC/data/summary/summary_1574.csv",
                                                                                preprocessor=EnglishPreProcessor())