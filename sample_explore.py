#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:28:59 2024

@author: anna
"""

from torchtext import datasets
import random
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt

vectorizer_uni = CountVectorizer()
vectorizer_bi = CountVectorizer(ngram_range = (2,2))
tokenizer = vectorizer_uni.build_tokenizer()

train, val, test = datasets.PennTreebank()
t,v,t2 = datasets.WikiText2()

data = [train,val,test,t,v,t2]
all_data = [i for ds in data for i in ds if len(tokenizer(i))>3 and len(tokenizer(i))<50]

sample_size = 5000
sample_sentences = []

while len(sample_sentences) < sample_size:
    if len(sample_sentences)%100 == 0:
        print("sequence", len(sample_sentences))
    ds = random.choice(data)
    ds = ds.shuffle()
    sent = list(ds)[0]
    length = len(tokenizer(sent))
    if length>3 and length<50:
        sample_sentences.append(sent)
  #%%  
#explore sentence length distribution

all_tokens = [tokenizer(s) for s in all_data]
all_lengths = [len(s) for s in all_tokens]

tokens = [tokenizer(s) for s in sample_sentences]
lengths = [len(s) for s in tokens]

plt.hist(all_lengths, bins = 46, label = "All data")
plt.hist(lengths, bins = 46, label = "Sample data")
plt.title("Sentence Length Distribution")
plt.legend()
plt.show()


all_hist = np.histogram(all_lengths, bins = 46)
all_hist_perc = [l/sum(all_hist[0])*100 for l in all_hist[0]]

hist = np.histogram(lengths, bins = 46)
hist_perc = [l/sum(hist[0])*100 for l in hist[0]]

plt.plot(list(range(46)), hist_perc, label = "Sample data")
plt.plot(list(range(46)), all_hist_perc, label = "All data")
plt.title("Sentence Length Distribution")
plt.xlabel("Sentence length")
plt.ylabel("Percent of sentences")
plt.legend()
plt.show()

#%%
#explore word frequency--unigram and bigram?

dtm_uni = vectorizer_uni.fit_transform(sample_sentences)
dtm_bi = vectorizer_bi.fit_transform(sample_sentences)

uni_freqs = np.array(np.sort(np.sum(dtm_uni, axis = 0))).squeeze()[::-1]
bi_freqs = np.array(np.sort(np.sum(dtm_bi, axis = 0))).squeeze()[::-1]

uni_perc = [i/np.sum(uni_freqs)*100 for i in uni_freqs[:50]]
bi_perc = [i/np.sum(bi_freqs)*100 for i in bi_freqs[:50]]

all_dtm_uni = vectorizer_uni.fit_transform(all_data)
all_dtm_bi = vectorizer_bi.fit_transform(all_data)

all_uni_freqs = np.array(np.sort(np.sum(all_dtm_uni, axis = 0))).squeeze()[::-1]
all_bi_freqs = np.array(np.sort(np.sum(all_dtm_bi, axis = 0))).squeeze()[::-1]

all_uni_perc = [i/np.sum(all_uni_freqs)*100 for i in all_uni_freqs[:50]]
all_bi_perc = [i/np.sum(all_bi_freqs)*100 for i in all_bi_freqs[:50]]

plt.plot(list(range(50)), uni_perc, label = "Sample data")
plt.plot(list(range(50)), all_uni_perc, label = "All_data")
plt.title("Unigram word frequency distribution")
plt.xlabel("Index")
plt.ylabel("% of word frequency")
plt.legend()
plt.show()

plt.plot(list(range(50)), bi_perc, label = "Sample data")
plt.plot(list(range(50)), all_bi_perc, label = "All data")
plt.title("Bigram word frequency distribution")
plt.xlabel("Index")
plt.ylabel("% of word frequency")
plt.legend()
plt.show()
