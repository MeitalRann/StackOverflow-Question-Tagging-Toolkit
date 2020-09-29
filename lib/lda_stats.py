# This code follows the tutorial in https://www.pluralsight.com/guides/topic-identification-nlp
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('wordnet')      #download if using this module for the first time


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')    #download if using this module for the first time


#For Gensim
import gensim
import string
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

def clean(document, stopwords, exclude, lemma):
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stopwords])
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())
    return normalized

def preproc_text(compileddoc):
    # Text preprocessing:
    global stopwords
    stopwords = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    final_doc = [clean(document, stopwords, exclude, lemma).split() for document in compileddoc]
    #final_doc = clean(document, stopwords, exclude, lemma).split()
    return final_doc

def cr_lda(final_doc, lda_n_topics=5):
    # Preparing Document-Term Matrix, create LDA object and train LDA:
    # 1) convert the corpus into a matrix representation:
    dictionary = corpora.Dictionary(final_doc)  # create the term dictionary of the corpus, where every unique term is assigned an index
    DT_matrix = [dictionary.doc2bow(doc) for doc in final_doc]
    #DT_matrix = dictionary.doc2bow(final_doc)  # convert the corpus into a Document-Term Matrix using the dictionary
    # 2) Create the object for the LDA model
    Lda_object = gensim.models.ldamodel.LdaModel
    # 3) Train LDA model on the DT_matrix
    lda_model = Lda_object(DT_matrix, num_topics=lda_n_topics, id2word=dictionary)
    return lda_model, DT_matrix


def main(corpus, lda_n_topics=10):

    final_doc = preproc_text(corpus)  # preprocessing
    lda_model, DT_matrix = cr_lda(final_doc, lda_n_topics)  # make LDA model
    n_doc = len(corpus)
    vec_lda = np.zeros((n_doc, lda_n_topics))
    for i in range(n_doc):
        # get the distribution for the i-th document in corpus
        for tmp in lda_model.get_document_topics(DT_matrix[i]):
            topic, prob = tmp
            vec_lda[i, topic] = prob
    return vec_lda

