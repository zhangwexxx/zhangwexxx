import re
import time
import pandas as pd
from datetime import datetime
import gensim.corpora as corpora
from nltk.corpus import stopwords
from gensim.models import phrases
from gensim.utils import simple_preprocess
from utils.coo_311_utils import status_message

def clean_text(df, text_column=None):
    if text_column is None:
        text_series = df.to_frame()
    else:
        text_series = df[text_column]
    text_series = text_series.fillna('')
    # Remove Punctuation
    df['clean_text'] = text_series.str.replace('[^a-zA-Z#]', ' ')
    # df['text_processed'] = df['text_processed'].apply(lambda x: simple_preprocess(x, deacc=True))
    # Lower
    df['clean_text'] = df['clean_text'].map(lambda x: x.lower())
    # Stop Words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', ''])
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split(' ') if word not in stop_words]))
    return df


def create_corpus(df, text_column):
    start_time = time.time()
    status_message("Cleaned Text")
    data_words = df.text_processed.values.tolist()
    data_words = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]
    status_message("Creating Dictionary...")
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]
    status_message("Returned Corpus in {:.2f}s".format(time.time() - start_time))
    return id2word, corpus


def create_trigram_corpus(initial_corpus):
    # Create Bigram
    bigram = phrases.Phrases(initial_corpus, min_count = 3, threshold = 10)
    # Create Trigram
    trigram = phrases.Phrases(bigram[initial_corpus], threshold=10)
    return trigram[initial_corpus]

