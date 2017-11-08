from __future__ import division
import sys,json,math
import os
import numpy as np

def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.
    
    w2vec={}
    with open(filename,"r") as f_in:
        for line in f_in:
            line_split=line.replace("\n","").split()
            w=line_split[0]
            vec=np.array([float(x) for x in line_split[1:]])
            w2vec[w]=vec
    return w2vec

def load_contexts(filename):
    # Returns a dict containing a {word: contextcount} mapping.
    # It loads everything into memory.

    data = {}
    for word,ccdict in stream_contexts(filename):
        data[word] = ccdict
    print "file %s has contexts for %s words" % (filename, len(data))
    return data

def stream_contexts(filename):
    # Streams through (word, countextcount) pairs.
    # Does NOT load everything at once.
    # This is a Python generator, not a normal function.
    for line in open(filename):
        word, n, ccdict = line.split("\t")
        n = int(n)
        ccdict = json.loads(ccdict)
        yield word, ccdict

def cosine_similarity(word_dict1, word_dict2):
    norm1 = np.sqrt(sum([word_dict1[word]**2 for word in word_dict1]))
    norm2 = np.sqrt(sum([word_dict2[word]**2 for word in word_dict2]))

    numerator = 0
    if len(word_dict1) > len(word_dict2):
        for word in word_dict2:
            if word in word_dict1:
                numerator += word_dict1[word]*word_dict2[word]
    else:
        for word in word_dict1:
            if word in word_dict2:
                numerator += word_dict1[word]*word_dict2[word]

    return(numerator*1.0)/(norm1*norm2)



