#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np

X = np.load('dict_features.npy', allow_pickle = True)
X = X.item()
arr_X = np.load('arr_features.npy', allow_pickle = True)

keys = []
for thing in X.keys():
    keys.append(thing)

subkeys = []
for thing in X[keys[0]].keys():
    subkeys.append(thing)
    
#example
print(X[keys[0]][subkeys[0]])


# In[56]:


X = np.load('dict_features_f.npy', allow_pickle = True)
X = X.item()
X_f = []
for key in keys:
    X_f.append(X[key].reshape(8750))
X_f = np.array(X_f)
print(X_f.shape)


# In[27]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def find_std_x(arr_data):
    shape = []
    for i in range(len(arr_data)):
        for j in range(len(arr_data[0])):
            shape.append(arr_data[i][j].shape[0])
    shape = np.array(shape)
    return np.max(shape)

def padding(arr_data, std_x, std_y):
    result = []
    for i in range(len(arr_data)):
        temp = []
        for j in range(len(arr_data[0])):
            sub_arr = arr_data[i][j].ravel()
            full_size = np.zeros((std_x, std_y)).ravel()
            output = sub_arr.copy()
            output.resize(full_size.shape)
            output = output.reshape(std_x, std_y)
            temp.append(output)
        result.append(np.array(temp))
    return np.array(result)

def scaling(dict_data, scaler):
    if scaler == "StandardScaler":
        scaler = StandardScaler()
        scaler.fit_transform(dict_data)
        return dict_data

def encoding(dict_data, list_keys, list_subkeys, encoder):
    if encoder == "pca":
        features_out = []
        for key in range(len(list_keys)):
            temp = []
            for subkey in range(len(subkeys)):
                pca = PCA()
                temp_re = pca.fit_transform(dict_data[keys[key]][subkeys[subkey]].reshape(-1, 1))
                temp.append(temp_re)
            features_out.append(temp)
        return features_out


result_f = padding(arr_X, 217, 25)
print(result_f.shape)



from sklearn import preprocessing
from sklearn import pipeline

scaler = preprocessing.StandardScaler()
pca = PCA()
pipe =pipeline.Pipeline([('encode',pca),('scaler',scaler)])
output = pipe.fit_transform(X_f)
print(output.shape)
np.save('encoded', output)

