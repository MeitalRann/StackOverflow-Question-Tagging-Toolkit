import sys
import os, getopt
from os import path
import pandas as pd
import numpy as np
import scipy.sparse
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import csv
import random
import pickle
import datetime
import argparse


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


import lda_stats as lda_stats
import get_pred_tags as get_tags
import manage_data as manage_data
import manage_data as manage_data

# set seed:
seed_value = 1
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
from tensorflow.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=2,
                        help="method for text embedding")
    parser.add_argument("-s", type=int, default=0,
                        help="starting stage")
    parser.add_argument("-e", type=int, default=3,
                        help="ending stage")
    parser.add_argument("--prj_dir",
                        help="working directory")
    args = parser.parse_args()

    method = args.m
    stage = args.s
    exit = args.e
    prj_dir = args.prj_dir

    batch_size = 512
    n_tags = 100  # number of tags in the data
    gamma = 30  # parameter for relative importance of lda

    # data_prep:
    if stage <= 0 and exit >= 0:
        print('Data preparation...')
        # get questions:
        print('Get questions...')
        ques_data = pd.read_csv("Questions.csv", engine='python')
        ques_data['Body'] = ques_data['Body'].str.slice_replace(start=0, stop=3, repl=': ')
        ques_data['Text'] = ques_data.Title.str.cat(ques_data.Body)
        # get tags:
        print('Get tags...')
        tag_data = pd.read_csv("Tags.csv")
        # keep only the n_tags most frequent tags:
        tag_data['freq'] = tag_data.groupby('Tag')['Tag'].transform('count')
        tmp = tag_data['Tag'].value_counts().nlargest(n_tags)[-1]
        tag_top_freq = tag_data.groupby('Tag').filter(lambda x: len(x) >= tmp)
        n_tags = tag_top_freq.Tag.nunique() #count number of tags
        tag_grouped = tag_top_freq.groupby('Id')['Tag'].apply(list).reset_index(name='Tag_list')
        #tag_grouped.head(1)

        print('save data to csv...')
        if not os.path.exists(prj_dir + r'\data'):
            os.makedirs(prj_dir + r'\data')
        data = ques_data.set_index('Id').join(tag_grouped.set_index('Id'))
        data = data.filter(['Id', 'Text', 'Tag_list']).dropna()
        data.to_csv(prj_dir+r'\\data\\'+'filtered_data.csv', index=False)

    # topic embedding (LDA/BERT/LDA+BERT):
    if stage <= 1 and exit >= 1:
        print('Get vector representations of the questions and tags...')
        # apply lda analysis to the questions (import data if not exists in variables):
        try:
            corpus = data['Text']
        except:
            data = pd.read_csv("filtered_data.csv", engine='python')
            corpus = data['Text']

        if method == 0 or method == 2:  # topic embedding using LDA
            print('Question embedding using LDA...')
            lda = LDA_stats.main(corpus, lda_n_topics=10)
            # save data:
            manage_data.save_data(lda, method, prj_dir)
            print('LDA data was saved!')
        elif method == 1 or method == 2: # topic embedding using BERT
            print('Question embedding using BERT...')
            model = SentenceTransformer('bert-base-nli-max-tokens')
            bert = np.array(model.encode(corpus, show_progress_bar=True))
            # save data:
            manage_data.save_data(bert, method, prj_dir)
            print('BERT data was saved!')


        # make tags into labels:
        print('Tags binarization...')
        tags_corpus = [i.replace('[','').replace(']','').replace(' ','').split(',') for i in data['Tag_list']]
        mlb = MultiLabelBinarizer(sparse_output=True)
        labels = mlb.fit_transform(tags_corpus)
        # save labels:
        scipy.sparse.save_npz(prj_dir+r'\\data\\labels.npz', labels)
        print('Labels were saved!')

    # visualize data:
    if stage <= 2 and exit >= 2:
        print('Start data visualization...')
        # map to lower dimension
        try:
            data = manage_data.choose_data(method, prj_dir, gamma, data)
        except:
            data = manage_data.choose_data(method, prj_dir, gamma)
        sub_data = data[:1000]

        print('Apply T-sne to lower the dimension to 2D...')
        data_embedded = TSNE(n_components=2).fit_transform(sub_data)
        print('T-sne done!')
        plt.scatter(data_embedded[:, 0], data_embedded[:, 1])
        plt.ylabel('Embedded data')
        plt.show()

    # classifier:
    if stage <= 3 and exit >= 3:
        # put data in the right format:
        print("Start DNN training and testing stage")
        try:
            data, name = manage_data.choose_data(method, prj_dir, gamma, data)
        except:
            data, name = manage_data.choose_data(method, prj_dir, gamma)
        # data = data[:200000]  # debug

        # get labels:
        print("Get labels")
        try:
            labels = labels[:len(data)]
            n_files = len(labels)
        except:
            labels = scipy.sparse.load_npz(prj_dir+r'\\data\\labels.npz')
            labels = labels.todense()[:len(data)]
            n_files = len(labels)



        print("Divide to train and test data")
        train_input, test_input, train_labels, test_labels = train_test_split(
            data, labels, test_size = 0.2, random_state=seed_value)
        del data  # delete for memory reasons
        train_input = train_input.to_numpy()
        test_input = test_input.to_numpy()
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        input_shape = train_input[0].shape  # depends on LDA/BERT

        # normalize:
        print('Normalize data...')
        mean = np.mean(train_input, axis=0)
        std = np.std(train_input, axis=0)
        train_input = stats.zscore(train_input, axis=0)
        test_input = (test_input-mean)/std
        # data = normalize(scipy.sparse.csr_matrix(data), axis=0)
        # data = norm_data.toarray()

        # create log folder if not exists:
        if not os.path.exists(prj_dir+r'\\logs\\'+name):
            os.makedirs(prj_dir+r'\\logs\\'+name)
        # create model folder if not exists:
        if not os.path.exists(prj_dir+r'\\model\\'+name):
            os.makedirs(prj_dir+r'\\model\\'+name)

        # neural network model:
        print('Initialize model...')
        model = keras.Sequential([layers.Dense(units=256, input_shape=input_shape, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dropout(rate=0.2),
                                  layers.Dense(units=256, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dropout(rate=0.2),
                                  layers.Dense(units=256, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dropout(rate=0.2),
                                  layers.Dense(units=n_tags, activation='sigmoid')])
        # see model's info:
        model.summary()
        # compile model: (get it readdy for training)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
        log_dir = prj_dir+r"logs\\"+name+r'\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        my_callbacks = [
            keras.callbacks.EarlyStopping(patience=8),
            keras.callbacks.ModelCheckpoint(filepath=prj_dir+r".\\model\\"+name+r"\\model.{epoch:02d}-{val_loss:.2f}.h5"),
            keras.callbacks.TensorBoard(log_dir=log_dir)]
        epochs = 100
        # train model:
        # verbose can be 0/1/2 (0=no output, 2=max output)
        # shuffle happens after the validation split!!
        # validation set is set to be 5% of train data
        print('Train model...')
        model.fit(x=train_input,y=train_labels,validation_split=0.05, batch_size=batch_size,epochs=epochs,shuffle=True,
                  verbose=2, callbacks=my_callbacks)
        # test:
        print('Test model...')

        pred = model.predict(x=test_input,batch_size=batch_size, verbose=0)
        # choose tags: (between 1 to 5 tags per question)
        pred_bin = get_tags.get_tags(pred, n_tags, max_tags=5, th=0.9)
        # Jaccard similarity measure:
        print('Compute Jaccard similarity measure...')
        s_labels = scipy.sparse.csr_matrix(test_labels)
        s_pred = scipy.sparse.csr_matrix(pred_bin)
        J_score = jaccard_score(s_labels, s_pred, average='samples')
        print(r'Jaccard similarity measure: ' + str(J_score))
        print('Complete!')


