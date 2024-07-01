#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:28:59 2024

@author: anna
"""

#from torchtext import datasets
from datasets import load_dataset
import random
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

vectorizer_uni = CountVectorizer()
vectorizer_bi = CountVectorizer(ngram_range = (2,2))
tokenizer = vectorizer_uni.build_tokenizer()

wikitext = load_dataset("wikitext", "wikitext-2-v1")
ptb = load_dataset("ptb_text_only")
#torchtext deprecated, oh joy!
# train, val, test = datasets.PennTreebank()
# t,v,t2 = datasets.WikiText2()

data = [[i["sentence"] for i in ptb["train"]],
        [i["sentence"] for i in ptb["validation"]],
        [i["sentence"] for i in ptb["test"]],
        [i["text"] for i in wikitext["train"]],
        [i["text"] for i in wikitext["validation"]],
        [i["text"] for i in wikitext["test"]]]
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
# plt.hist(all_lengths, bins = 46, label = "All data")
# plt.hist(lengths, bins = 46, label = "Sample data")
# plt.title("Sentence Length Distribution")
# plt.legend()
# plt.show()

sample_loc = "/media/anna/Samsung_T5/manifolds/sample_sequences/"

all_tokens = [tokenizer(s) for s in all_data]
all_lengths = [len(s) for s in all_tokens]

all_hist = np.histogram(all_lengths, bins = range(4,50))
all_hist_perc = [l/sum(all_hist[0])*100 for l in all_hist[0]]



plt.subplot(2,2,1)
for sample in os.listdir(sample_loc):
    if sample[-4:]==".pkl":
        sample_count = sample[:-4]
        sample_sentences = pickle.load(open(sample_loc+sample, "rb"))
        tokens = [tokenizer(s) for s in sample_sentences]
        lengths = [len(s) for s in tokens]
        hist = np.histogram(lengths, bins = range(4,50))
        hist_perc = [l/sum(hist[0])*100 for l in hist[0]]
        
        plt.plot(all_hist[1][:-1], hist_perc, label = f"Sample_{sample_count}")

plt.plot(all_hist[1][:-1], all_hist_perc, label = "All data", c = "black", linewidth = 2)
plt.title("Sentence Length Distribution")
plt.xlabel("Sentence length")
plt.ylabel("Percent of sentences")
# plt.legend()
# plt.savefig("sentence_length_compare.pdf", format = "pdf")
# plt.show()


#%%
#explore word frequency--unigram and bigram?
all_dtm_uni = vectorizer_uni.fit_transform(all_data)
all_dtm_bi = vectorizer_bi.fit_transform(all_data)

all_uni_freqs = np.array(np.sort(np.sum(all_dtm_uni, axis = 0))).squeeze()[::-1]
all_bi_freqs = np.array(np.sort(np.sum(all_dtm_bi, axis = 0))).squeeze()[::-1]

all_uni_perc = [i/np.sum(all_uni_freqs)*100 for i in all_uni_freqs[:50]]
all_bi_perc = [i/np.sum(all_bi_freqs)*100 for i in all_bi_freqs[:50]]


for sample in os.listdir(sample_loc):
    if sample[-4:]==".pkl":
        sample_count = sample[:-4]
        sample_sentences = pickle.load(open(sample_loc+sample, "rb"))
        dtm_uni = vectorizer_uni.fit_transform(sample_sentences)
        uni_freqs = np.array(np.sort(np.sum(dtm_uni, axis = 0))).squeeze()[::-1]
        uni_perc = [i/np.sum(uni_freqs)*100 for i in uni_freqs[:50]]
        plt.plot(list(range(50)), uni_perc, label = f"Sample_{sample_count}")


plt.plot(list(range(50)), all_uni_perc, label = "All_data", c = "black", linewidth = 2)
plt.title("Unigram word frequency distribution")
plt.xlabel("Index")
plt.ylabel("% of word frequency")
plt.legend()
plt.savefig("unigram_freq_compare.pdf", format = "pdf")
plt.show()


for sample in os.listdir(sample_loc):
    if sample[-4:]==".pkl":
        sample_count = sample[:-4]
        sample_sentences = pickle.load(open(sample_loc+sample, "rb"))
        dtm_bi = vectorizer_bi.fit_transform(sample_sentences)
        bi_freqs = np.array(np.sort(np.sum(dtm_bi, axis = 0))).squeeze()[::-1]
        bi_perc = [i/np.sum(bi_freqs)*100 for i in bi_freqs[:50]]

        plt.plot(list(range(50)), bi_perc, label = f"Sample_{sample_count}")

plt.plot(list(range(50)), all_bi_perc, label = "All data", c = "black", linewidth = 2)
plt.title("Bigram word frequency distribution")
plt.xlabel("Index")
plt.ylabel("% of word frequency")
plt.legend()
plt.savefig("bigram_freq_compare.pdf", format = "pdf")
plt.show()

#%%
#import and compare metric results on sample spaces

metric_start = "/media/anna/Samsung_T5/manifolds/sample_sequences/metrics"

sample_metrics = pd.DataFrame(columns = ["p-error", "p-centroids", "p-point_counts", "p-point_dist", "p-patchiness",
                                  "a-error", "a-centroids", "a-point_counts", "a-point_dist", "a-patchiness",
                                  "EEE", "VRM", "IsoScore"])

for space in os.listdir(metric_start):
        for filename in os.listdir(metric_start+"/"+space):
            if filename == "metrics.json":
                fileloc = metric_start+"/"+space+"/"+filename
                with open(fileloc, "r") as f:
                    r = json.load(f)
                    df1 = pd.DataFrame(r["product"], index = [space])
                    df1 = df1.rename(columns = {"error":"p-error",
                                          "centroids":"p-centroids",
                                          "point_counts":"p-point_counts",
                                          "point_dist":"p-point_dist",
                                          "patchiness":"p-patchiness"})
                    df2 = pd.DataFrame(r["additive"], index = [space])
                    df2 = df2.rename(columns = {"error":"a-error",
                                          "centroids":"a-centroids",
                                          "point_counts":"a-point_counts",
                                          "point_dist":"a-point_dist",
                                          "patchiness":"a-patchiness"})
                    df3 = pd.DataFrame(r["spread"], index = [space])
                    merged = pd.concat([df1,df2,df3], axis = 1)
                    sample_metrics = pd.concat([sample_metrics, merged])
                    

sample_metrics[["p-count_var", "p-count_kl"]] = pd.DataFrame(sample_metrics['p-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = sample_metrics.index)
sample_metrics[["a-count_var", "a-count_kl"]] = pd.DataFrame(sample_metrics['a-point_counts'].str[1:].str[:-1].str.split(",", expand = True), index = sample_metrics.index)
sample_metrics["IsoScore"] = sample_metrics["IsoScore"].str.replace("tensor(","").str[:-1]
sample_metrics = sample_metrics.drop(["p-point_counts","a-point_counts"] , axis = 1)
sample_metrics = sample_metrics.astype(float)

pretty_names = {
           "p-error":"PQ Reconstruction Error",
           "p-count_var": "PQ Point Count Variance",
           "p-count_kl": "PQ Point Count KL-Divergence",
           "p-centroids":"PQ Centroid Distance Variance",
           "p-point_dist": "PQ Point Distribution EEE",
           "p-patchiness": "PQ Patchiness",
           "a-error": "AQ Reconstruction Error",
           "a-count_var": "AQ Point Count Variance",
           "a-count_kl": "AQ Point Count KL-Divergence",
           "a-centroids": "AQ Centroid Distance Variance",
           "a-point_dist": "AQ Point Distribution EEE",
           "a-patchiness": "AQ Patchiness",
           "EEE":"EEE",
           "VRM":"VRM",
           "IsoScore":"IsoScore",}

sample_metrics.columns = list(pretty_names.values())
sample_metrics.plot(kind='box', subplots = True, layout = (3,6),
                    sharey=False,
                    figsize = (16,8),
                    #rot = 20,
                    title = "Sample Latent Space Metric Distributions"
                    )

plt.savefig("sample_metric_compare.pdf", format = "pdf", bbox_inches = "tight")

#%%
#clean up freq viz

sample_loc = "/media/anna/Samsung_T5/manifolds/sample_sequences/"

#sentence lengths for entire dataset
all_tokens = [tokenizer(s) for s in all_data]
all_lengths = [len(s) for s in all_tokens]

all_hist = np.histogram(all_lengths, bins = range(4,50))
all_hist_perc = [l/sum(all_hist[0])*100 for l in all_hist[0]]

#unigram/bigram frequencies for entire dataset
all_dtm_uni = vectorizer_uni.fit_transform(all_data)
all_dtm_bi = vectorizer_bi.fit_transform(all_data)

all_uni_freqs = np.array(np.sort(np.sum(all_dtm_uni, axis = 0))).squeeze()[::-1]
all_bi_freqs = np.array(np.sort(np.sum(all_dtm_bi, axis = 0))).squeeze()[::-1]

all_uni_perc = [i/np.sum(all_uni_freqs)*100 for i in all_uni_freqs[:50]]
all_bi_perc = [i/np.sum(all_bi_freqs)*100 for i in all_bi_freqs[:50]]


sent_hist = []
uni_hist = []
bi_hist = []

for sample in os.listdir(sample_loc):
    if sample[-4:]==".pkl":
        sample_count = sample[:-4]
        sample_sentences = pickle.load(open(sample_loc+sample, "rb"))
        
        #sentence length
        tokens = [tokenizer(s) for s in sample_sentences]
        lengths = [len(s) for s in tokens]
        hist = np.histogram(lengths, bins = range(4,50))
        hist_perc = [l/sum(hist[0])*100 for l in hist[0]]
        sent_hist.append(hist_perc)
        
        #unigram frequencies
        dtm_uni = vectorizer_uni.fit_transform(sample_sentences)
        uni_freqs = np.array(np.sort(np.sum(dtm_uni, axis = 0))).squeeze()[::-1]
        uni_perc = [i/np.sum(uni_freqs)*100 for i in uni_freqs[:50]]
        uni_hist.append(uni_perc)
        
        #bigram frequencies
        dtm_bi = vectorizer_bi.fit_transform(sample_sentences)
        bi_freqs = np.array(np.sort(np.sum(dtm_bi, axis = 0))).squeeze()[::-1]
        bi_perc = [i/np.sum(bi_freqs)*100 for i in bi_freqs[:50]]
        bi_hist.append(bi_perc)

#%%
f, axes = plt.subplot_mosaic('AB;CC', constrained_layout = True)
#plt.subplots_adjust(hspace = 0.2)
ax1 = axes["A"]
ax2 = axes["B"]
ax3 = axes["C"]

for s in uni_hist:
    ax1.plot(list(range(50)), s, alpha = 0.5)
ax1.plot(list(range(50)), all_uni_perc, label = "All_data", c = "black", linewidth = 1.2)
ax1.set_title("Unigram Frequency Distribution")
ax1.set_ylabel("% of word frequency")
ax1.set_xlabel("\n\n")

for s in bi_hist:
    ax2.plot(list(range(50)), bi_perc, alpha = 0.5)
ax2.plot(list(range(50)), all_bi_perc, label = "All data", c = "black", linewidth = 1.2)
ax2.set_title("Bigram Frequency Distribution")


for s in sent_hist:
    ax3.plot(all_hist[1][:-1], s, alpha = 0.5)
ax3.plot(all_hist[1][:-1], all_hist_perc, label = "All data", c = "black", linewidth = 2)
ax3.set_title("Sentence Length Distribution")
ax3.set_xlabel("Sentence length")
ax3.set_ylabel("% of sentences")

plt.savefig("sample_text_compare.pdf", format = "pdf")

