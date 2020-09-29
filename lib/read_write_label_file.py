import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import csv
import scipy.sparse
import pickle



# data = pd.read_csv("filtered_data.csv", engine='python')
# tags_corpus = [i.replace('[', '').replace(']', '').replace(' ', '').split(',') for i in data['Tag_list']]
# mlb = MultiLabelBinarizer(sparse_output=True)
# labels = mlb.fit_transform(tags_corpus)
#
# scipy.sparse.save_npz('labels_dbug.npz', labels)
# labels = scipy.sparse.load_npz('labels_dbug.npz')
#
# labels = labels.todense()

pickle_in = open("bert.pickle","rb")
data = pickle.load(pickle_in)
n = len(data)
third = n//3
part1 = data[:third]
pickle_out = open("bert1.pickle","wb")
pickle.dump(part1, pickle_out)
pickle_out.close()

part2 = data[third:2*third]
pickle_out = open("bert2.pickle","wb")
pickle.dump(part2, pickle_out)
pickle_out.close()

part3 = data[2*third:]
pickle_out = open("bert3.pickle","wb")
pickle.dump(part3, pickle_out)
pickle_out.close()

# pickle_in = open("lda_p.pickle","rb")
# example_dict = pickle.load(pickle_in)
# print('yay')
