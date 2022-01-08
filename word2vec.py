from pathlib import Path
#from sentence_transformers import SentenceTransformer
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import re
#from gensim.utils import simple_preprocess
#from gensim.parsing.preprocessing import STOPWORDS
#from gensim import corpora, models
#from gensim.models.coherencemodel import CoherenceModel
#from nltk.stem import WordNetLemmatizer, SnowballStemmer
#from nltk.stem.porter import *
#import spacy
#import nltk
#import pandas as pd
#import numpy as np
#import umap
#from sklearn.feature_extraction.text import CountVectorizer
#import hdbscan
#np.random.seed(2018)



def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text)

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def word2vec():

    all_txt_files = []
    for file in Path("Madagascar").glob('**/*.txt'):
        all_txt_files.append(file.parent / file.name)
    n_files = len(all_txt_files)  # Should print the number of total files

    all_docs = []
    for txt_file in all_txt_files:
        with open(txt_file, encoding="utf8") as f:
            txt_file_as_string = f.read()
            pat = re.compile(r'[^a-zA-Z ]+')
            txt_file_as_string = re.sub(pat, '', txt_file_as_string).lower()
        all_docs.append(txt_file_as_string)

    model = Word2Vec(sentences=all_docs, vector_size=100, window=5, min_count=1, workers=4)

    print(model.wv.most_similar('community', topn=10))




all_txt_files = []
for file in Path("Madagascar").glob('**/*.txt'):
    all_txt_files.append(file.parent / file.name)
n_files = len(all_txt_files)


all_docs = []
for txt_file in all_txt_files:
    with open(txt_file, encoding="utf8") as f:
        txt_file_as_string = f.read()
        pat = re.compile(r'[^a-zA-Z ]+')
        txt_file_as_string = re.sub(pat, '', txt_file_as_string).lower()
    all_docs.append(txt_file_as_string)


print(all_docs[1])



