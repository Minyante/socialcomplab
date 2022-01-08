from pathlib import Path
#from sentence_transformers import SentenceTransformer
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
#from gensim.utils import simple_preprocess
#from gensim.parsing.preprocessing import STOPWORDS
#from gensim import corpora, models
#from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
#from nltk.stem.porter import *
#import spacy
#import nltk
#import pandas as pd
#import numpy as np
#import umap
#from sklearn.feature_extraction.text import CountVectorizer
#import hdbscan
#np.random.seed(2018)

def tfidf():
    all_txt_files = []
    for file in Path("Madagascar").glob('**/*.txt'):
        all_txt_files.append(file.parent / file.name)
    n_files = len(all_txt_files)
    print(n_files) #Should print the number of total files

    all_docs = []
    for txt_file in all_txt_files:
        with open(txt_file, encoding="utf8") as f:
            txt_file_as_string = f.read()
            pat = re.compile(r'[^a-zA-Z ]+')
            txt_file_as_string = re.sub(pat, '', txt_file_as_string).lower()
        all_docs.append(txt_file_as_string)
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=5, stop_words='english', use_idf=True, norm=None)

    transformed_documents = vectorizer.fit_transform(all_docs)
    transformed_documents_as_array = transformed_documents.toarray()
    # use this line of code to verify that the numpy array represents the same number of documents that we have in the file list
    print(len(transformed_documents_as_array))

    Path("./tf_idf_output").mkdir(parents=True, exist_ok=True)

    # construct a list of output file paths using the previous list of text files the relative path for tf_idf_output
    output_filename = 'tf_idf_output/output.csv'
    tuple_list = []
    # loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
    for counter, doc in enumerate(transformed_documents_as_array):
        # construct a dataframe
        tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
        tuple_list.extend(tf_idf_tuples)
        # output to a csv using the enumerated value for the filename
    one_doc_as_df = pd.DataFrame.from_records(tuple_list, columns=['term', 'score']).sort_values(by='score',ascending=False).reset_index(drop=True)
    one_doc_as_df.to_csv(output_filename)


def bert():
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

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(all_docs, show_progress_bar=True)
    umap_embeddings = umap.UMAP(n_neighbors=10, n_components=2, metric='cosine').fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)

    docs_df = pd.DataFrame(all_docs, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(all_docs))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df);
    topic_sizes.head(10)
    print(topic_sizes)
    print(top_n_words)

def c_tf_idf(documents, m, ngram_range=(1, 1)):
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

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

    tokenized_words = []
    IUCN_words = [
        "land",
        "water",
        "specie",
        "awareness",
        "law",
        "livelihood",
        "economic",
        "moral",
        "conservation",
        "legal",
        "policy",
        "research",
        "monitor",
        "education",
        "training",
        "institutional",
        "development"
    ]
    proposal_similarity_score = 0
    proposal_words = [
        "program",
        "train",
        "goal",
        "coordinate",
        "activity",
        "assist",
        "collected",
        "method",
        "teaching",
        "establishing",
        "improving",
        "building",
        "managing",
        "consolidated",
        "organize",
        "building",
        "create",
        "ecological",
        'monitoring',
        "landscape",
        "strength",
        "management",
        "local",
        "conduct",
        "outreach",
        "public",
        "increasing",
        "provide",
        "guidance",
        "purchase",
        "satellite",
        "course",
        "implementation",
        "support",
        "production",
        "distribution",
        "understand",
        "quantification",
        "prioritize",
        "capacity",
        "coral",
        "vulnerability",
        "identification",
        "designing",
        "planning",
        "forestry",
        "school",
        "reserve",
        "participatory",
        "analysis",
        "infrastructure",
        "protection",
        "biological",
        "restore",
        "original",
        "repair",
        "active",
        "basic",
        "educational",
        "national",
        "feedback",
        "interest",
        "fanamby",
        "adoption",
        "technical",
        "sustainably",
        "marine",
        "area",
        "fishery",
        "generating",
        "wildfire",
        "fish",
        "stock",
        "coastal",
        "conservation",
        "robust",
        "alleviation",
        "contribute",
        "biodiversity",
        "engagement",
        "region",
        "threat",
        "reduction",
        "lemur",
        "specie",
        "malagasy",
        "scientist",
        "module",
        "technique",
        "illegal",
        "logging",
        "molecular",
        "tool",
        "regional",
        "professional",
        "open",
        "access",
        "online",
        "learning",
        "threatened",
        "inventory",
        "revolving",
        "energy",
        "fund",
        "book",
        "need",
        "initial",
        "lasting",
        "impact",
        "effectiveness",
        "efficiency",
        "produce",
        "endowment",
        "evidence",
        "field",
        "systematic",
        "tropical",
        "plant",
        "formal",
        "compile",
        "framework",
        "cultural",
        "centered",
        "multidisciplinary",
        "agricultural"
    ]
    report_similarity_score = 0
    report_words = [
        "outcome",
        "accomplish",
        "development",
        "completed",
        "trained",
        "worked",
        "mapping",
        "improve",
        "enhance",
        "extend",
        "public",
        "raising",
        "supported",
        "outreach",
        "management",
        "analysis",
        "consistent",
        "awareness",
        "governance",
        "workshop",
        "training",
        "public",
        "donor",
        "conservation",
        "strategy",
        "community",
        "association",
        "fundraising",
        "assemble",
        "benefit",
        "direct",
        "successfully",
        "tested",
        "validate",
        "coordination",
        "restoration",
        "strengthen",
        "presenting",
        "degradation",
        "decrease",
        "reduction",
        "strategy",
        "interaction",
        "systematic",
        "evaluation",
        "upgrading",
        "maintain",
        "documentation",
        "promote",
        "ecotourism",
        "scientific",
        "capacity",
        "representing",
        "plant",
        "subspecies",
        "museum",
        "build",
        "stakeholder",
        "ownership",
        "long",
        "term",
        "amphibian",
        "scope",
        "reach",
        "cross",
        "institutional",
        "collaboration",
        "natural",
        "resource",
        "ecosystem",
        "service",
        "climate",
        "change",
        "different",
        "stage",
        "transfer",
        "contract",
        "attitude",
        "provided",
        "mangrove",
        "healthy",
        "operational",
        "enforcement",
        "committee",
        "member",
        "reinforced",
        "valuing",
        "build",
        "adapting",
        "mitigation",
        "regional",
        "advancing",
        "vulnerability",
        "coral",
        "strategic",
        "producing",
        "model",
        "refining",
        "partnership",
        "facility",
        "curriculum",
        "botanist",
        "accomplished",
        "comprehensive",
        "evolution",
        "consolidate",
        "founding",
        "assembly",
        "momentum",
        "innovate",
        "field",
        "nature",
        "study",
        "redesign",
        "papua",
        "new",
        "guinea",
        "rainforest",
        "staff",
        "respectively",
        "reserve",
        "index",
        "consultant"
    ]

    for i in all_docs:
        lemmed = []
        tok = word_tokenize(i)
        for j in tok:
            lemmed.append(lemmatize_stemming(j))
        tokenized_words.append(lemmed)

    model = Word2Vec(sentences=tokenized_words, vector_size=100, window=5, min_count=5, workers=4)

    for IUCN in IUCN_words:
        for proposal in proposal_words:
            proposal_similarity_score += model.wv.similarity(proposal, IUCN)

        for report in report_words:
            report_similarity_score += model.wv.similarity(report, IUCN)

        print(
            str(IUCN) + "'s cosine similarity score to proposal: " +
            str(proposal_similarity_score) +
            ", report: " +
            str(report_similarity_score))

        if(report_similarity_score > proposal_similarity_score):
            print(IUCN + " is closer to report")
        else:
            print(IUCN + " is closer to proposal")

        report_similarity_score = 0
        proposal_similarity_score = 0
