import pickle
import pandas as pd
import os


def save_data(data, method, prj_dir):
    if not os.path.exists(prj_dir + r'\data'):
        os.makedirs(prj_dir + r'\data')

    if method == 0:
        if not os.path.exists(prj_dir + r'\data\lda'):
            os.makedirs(prj_dir + r'\data\lda')
        name = prj_dir + r'\data\lda'+r"lda.pickle"
        pickle_out = open(name, "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        print("LDA file saved!")
    else:
        if not os.path.exists(prj_dir + r'\data\bert'):
            os.makedirs(prj_dir + r'\data\bert')
        n = len(data)
        third = n // 3
        part1 = data[:third]
        pickle_out = open(prj_dir + r'\data\bert'+r"bert1.pickle", "wb")
        pickle.dump(part1, pickle_out)
        pickle_out.close()

        part2 = data[third:2 * third]
        pickle_out = open(prj_dir + r'\data\bert'+r"bert2.pickle", "wb")
        pickle.dump(part2, pickle_out)
        pickle_out.close()

        part3 = data[2 * third:]
        pickle_out = open(prj_dir + r'\data\bert'+r"bert3.pickle", "wb")
        pickle.dump(part3, pickle_out)
        pickle_out.close()
        print("BERT files saved!")

def load_data(method, prj_dir):
    if method == 0:
        name = prj_dir+r"\\data\\lda\\lda.pickle"
        pickle_in = open(name, "rb")
        data = pickle.load(pickle_in)
        print('Uploaded LDA file- success!')
    else:
        names = ["bert1.pickle", "bert2.pickle"]#, "bert3.pickle"]
        chunk_list = []
        for i in range(len(names)):
            pickle_in = open(prj_dir+r'\\data\\bert\\'+names[i], "rb")
            chunk = pd.DataFrame(pickle.load(pickle_in))
            chunk_list.append(chunk)
        data = pd.concat(chunk_list, axis=0)
        print('Uploaded BERT files- success!')
    return data

def choose_data(method, prj_dir, gamma=15, data=None):
    if method == 0:
        name="lda"
    elif method == 1:
        name = "bert"
    else:
        name = "lda_bert"

    if data is not None:
        return data, name
    if method == 0 or method == 1:
        data = load_data(method, prj_dir)
    else:
        data = load_data(1, prj_dir).reset_index(drop=True)
        lda = load_data(0, prj_dir)[:len(data)].reset_index(drop=True)
        data = pd.concat([lda.multiply(gamma), data], axis=1, ignore_index=True)
    return data, name

