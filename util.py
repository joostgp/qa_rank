# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:40:34 2017

@author: joostbloom
"""
import numpy as np

from itertools import combinations

import spacy

from sklearn.metrics.pairwise import cosine_similarity

# Initialize Spacy
nlp = spacy.load('en')


def tokenize(s):
    ''' Returns lemmatized tokenized list of string s with stopwords and
    punctuation removed. Plural words are converted to singular version.
    '''
    doc = nlp(s)

    l = []

    for t in doc:
        if not t.is_stop and not t.is_punct:
            if is_plural(t.lemma_):
                l.append(t.lemma_[:-1])
            else:
                l.append(t.lemma_)
    return l


def first_sentence(s):
    ''' Return first sentence of string s '''
    return nlp(s).sents.next().text


def mean_word_vector(s):
    ''' Returns mean word vector of string s '''

    doc = nlp(s)
    return sum(w.vector for w in doc) / len(doc)


def dist_matrix_to_rank_k(X, k=20):
    ''' Returns the k highest values of array of lists X
    '''
    ranks = np.argsort(X)[:, -1:-k:-1]

    # Mask zero similarity items in rank
    for ir, r in enumerate(ranks):
        for ijr, jr in enumerate(r):
            if X[ir, jr] == 0:
                ranks[ir][ijr] = -1

    return ranks


def is_plural(s):
    ''' Optimistic check if string is plural'''
    return s[-1] == 's'


def tokens_in_common(q, r):
    ''' Counts the number of tokens in common between list of strings q and r
    relative to number of tokens in string q.
    '''
    return float(len(np.intersect1d(q, r))) / len(q)


def cosine_dist(q, r):
    ''' Calculate cosine dist between vectors q and r '''
    return cosine_similarity(q, r)


def get_root(s):
    ''' Get root of string s '''
    doc = nlp(s)

    for w in doc:
        if w.dep_ == 'ROOT':
            return w.lemma_


def noun_chunks_match(q, r):
    ''' Calculates sum of maximum noun_chunks of q length found in r
    Different weights are applied depending on syntactic relations.

    See https://spacy.io/docs/usage/dependency-parse
    '''

    docQ = nlp(q)

    chunks = []
    weights = []

    # In first loop get chuncks
    for nc in docQ.noun_chunks:
        # Give noun chunck higher weight if it is object
        if nc.root.dep_ == 'dobj' and nc.root.head.dep_ == 'ROOT':
            w = 10
        else:
            w = 1

        # Remove stop words or punct from chunck
        if nc[0].is_stop or nc[0].is_punct:
            if len(list(nc)) > 1:
                chunks.append([p.lemma_ for p in nc[1:]])
                weights.append(w)
        else:
            chunks.append([p.lemma_ for p in nc])
            weights.append(w)

    cnt = 0

    # Loop over chunks and calculate final score
    for chunk, w in zip(chunks, weights):
        if ' '.join(chunk) in r:
            cnt += len(chunk) * w
        else:
            # Find smaller chunks iteratively
            for nc in range(1, len(chunk))[-1::]:
                for c in combinations(chunk, nc):
                    if ' '.join(c) in r:
                        cnt += nc * w
                        break

    return cnt

# ---------
# Metrics
# ---------


def reciprocal_rank(y_true, y_predict):
    ''' Calculates reciprocal rank for value y_true in list y_predict '''
    if y_true not in y_predict:
        return 0

    return 1 - float((np.where(y_predict == y_true)[0][0]))/len(y_predict)


def mean_rank(y_true, y_predict, k):
    ''' Calculates mean reciprocal rank for values y_true in list of lists
    y_predict considering first k elements '''
    n = len(y_predict[0])

    assert k <= n

    out = []

    for y_tr, y_pr in zip(y_true, y_predict):
        assert len(y_pr) == n

        out.append(reciprocal_rank(y_tr, y_pr[:k]))

    return sum(out)/len(out)


def mean_recall(y_true, y_predict, k):
    ''' Calculates mean recall for values y_true in list of lists
    y_predict considering first k elements '''
    n = len(y_predict[0])

    assert k <= n

    out = 0.0

    for y_tr, y_pr in zip(y_true, y_predict):
        assert len(y_pr) == n

        out += y_tr in y_pr[:k]

    return out / len(y_true)
