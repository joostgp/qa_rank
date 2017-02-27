# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:32:09 2017

@author: joostbloom

Script to pre calculate and evaluate ranked responses to questions using
various scoring methods. Using ranked voting the ensemble rank is stored in an
text file.

"""
from itertools import product
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from util import tokenize, first_sentence, dist_matrix_to_rank_k
from util import cosine_dist, tokens_in_common
from util import noun_chunks_match, mean_word_vector
from util import mean_rank, mean_recall

DATASET = 'Acme_dataset.xlsx'
OUTPUT = 'Acme_dataset_pre.txt'

# Maximum results to include
N_RESULT_MAX = 20

# Maximum results to include in evaluations
N_RESULT_SCORE = 10


def load_data():
    ''' Loads data as Pandas dataframe '''

    X = pd.read_excel(DATASET)

    # Ignore last row
    X.drop(X.index[-1], inplace=True)

    X['Question Key'] = X['Question Key'].astype(int)

    return X


def tokenize_data(X):
    ''' Stores token lists of questions and response in dataframe X'''
    X['t_q'] = X['Question Text'].apply(tokenize)
    X['t_r'] = X['Response'].apply(tokenize)
    X['t_r_f'] = X['Response'].apply(lambda x: tokenize(first_sentence(x)))

    return X


def rank_keyword_matching(X, k):
    ''' Rank responses based on keywords matches '''
    n = X.shape[0]
    mat = np.zeros((n, n))

    for (iq, ir) in product(range(n), range(n)):
        mat[iq, ir] = tokens_in_common(X.iloc[iq]['t_q'],  X.iloc[ir]['t_r'])

    r = dist_matrix_to_rank_k(mat, k)

    return r


def rank_cos_count_vector(X, k):
    ''' Rank responses based on cosine similarity between count vectors '''
    n = X.shape[0]
    mat = np.zeros((n, n))

    vectorizer = CountVectorizer()

    resp = vectorizer.fit_transform(X['t_r'].apply(lambda x: ' '.join(x)))
    ques = vectorizer.transform(X['t_q'].apply(lambda x: ' '.join(x)))

    for (iq, ir) in product(range(n), range(n)):
        mat[iq, ir] = cosine_dist(ques[iq, :], resp[ir, :])

    r = dist_matrix_to_rank_k(mat, k)

    return r


def rank_cos_tfifd_vector(X, k):
    ''' Rank responses based on cosine similarity between tf-ifd vectors '''
    n = X.shape[0]
    mat = np.zeros((n, n))

    vectorizer = TfidfVectorizer()

    resp = vectorizer.fit_transform(X['t_r'].apply(lambda x: ' '.join(x)))
    ques = vectorizer.transform(X['t_q'].apply(lambda x: ' '.join(x)))

    for (iq, ir) in product(range(n), range(n)):
        mat[iq, ir] = cosine_dist(ques[iq, :], resp[ir, :])

    r = dist_matrix_to_rank_k(mat, k)

    return r


def rank_cos_tfifd_vector_first_sent(X, k):
    ''' Rank responses based on cosine similarity between tf-ifd vectors
    of first sentence only '''

    n = X.shape[0]
    mat = np.zeros((n, n))

    vectorizer = TfidfVectorizer()

    resp = vectorizer.fit_transform(X['t_r_f'].apply(lambda x: ' '.join(x)))
    ques = vectorizer.transform(X['t_q'].apply(lambda x: ' '.join(x)))

    for (iq, ir) in product(range(n), range(n)):
        mat[iq, ir] = cosine_dist(ques[iq, :], resp[ir, :])

    r = dist_matrix_to_rank_k(mat, k)

    return r


def rank_noun_chuncks(X, k):
    ''' Rank responses based on noun_chunks matches '''
    n = X.shape[0]
    mat = np.zeros((n, n))

    for (iq, ir) in product(range(n), range(n)):
        mat[iq, ir] = noun_chunks_match(X.iloc[iq]['Question Text'],
                                        ' '.join(X.iloc[ir]['t_r']))

    r = dist_matrix_to_rank_k(mat, k)

    return r


def rank_mean_word_vector(X, k):
    ''' Rank responses based on cosine similarity between mean word vectors '''
    n = X.shape[0]
    mat = np.zeros((n, n))

    mean_q = X['t_q'].apply(lambda x: mean_word_vector(' '.join(x)))
    mean_r = X['t_r'].apply(lambda x: mean_word_vector(' '.join(x)))

    for (iq, ir) in product(range(n), range(n)):
        mat[iq, ir] = cosine_dist(mean_q.values[iq].reshape(1, -1),
                                  mean_r.values[ir].reshape(1, -1))

    r = dist_matrix_to_rank_k(mat, k)

    return r


def ensemble_ranks(ranks_collection, k):
    '''Ensemble collection of response rank using ranked voting '''
    # Get number of questions from first list of ranks_collection
    NQ = ranks_collection[0].shape[0]

    # Prepare output list
    rank_ensemble = []

    # Per question determine sum rank of each ranker
    for nq in range(NQ):
        r_final = defaultdict(int)

        for ranks in ranks_collection:
            for i, rank in enumerate(ranks[nq]):
                r_final[rank] += (k - i) * (rank != -1)

        # Sort per sum
        r_final_sort = sorted(r_final, key=r_final.get, reverse=True)

        # Back fill with mask value
        while len(r_final_sort) < k:
            r_final_sort.append(-1)

        # Report top scoring items
        rank_ensemble.append(r_final_sort[:k])

    return rank_ensemble


def store_output(X, ranks):
    ''' Stores output in txt file '''
    with open(OUTPUT, 'w+') as f:
        for ir, r in enumerate(ranks):
            qid = X.iloc[ir]['Question Key']
            rids = [str(X.iloc[x]['Question Key']) for x in r]
            f.write('{}: {}\n'.format(qid, ', '.join(rids)))


def print_score(X, r):
    print('Done! Scoring: mean rank = {:.2f}, mean recall = {:.2f}'
          .format(mean_rank(X.index, r, N_RESULT_SCORE),
                  mean_recall(X.index, r, N_RESULT_SCORE)))


if __name__ == '__main__':

    print('Loading and preprocessing data from {}'.format(DATASET))
    X = load_data()
    X = tokenize_data(X)

    print('Calculating rank using keyword matching')
    r_keyword = rank_keyword_matching(X, N_RESULT_MAX)
    print_score(X, r_keyword)

    print('Calculating rank using count vector similarity')
    r_count = rank_cos_count_vector(X, N_RESULT_MAX)
    print_score(X, r_count)

    print('Calculating rank using tf-ifd vector similarity')
    r_tfifd = rank_cos_tfifd_vector(X, N_RESULT_MAX)
    print_score(X, r_tfifd)

    print('Calculating rank using tf-ifd vector similarity on first sentence')
    r_tfifd_first = rank_cos_tfifd_vector_first_sent(X, N_RESULT_MAX)
    print_score(X, r_tfifd_first)

    print('Calculating rank using noun chunk matching')
    r_noun = rank_noun_chuncks(X, N_RESULT_MAX)
    print_score(X, r_noun)

    print('Calculating rank using mean word vector similarity')
    r_word_vector = rank_mean_word_vector(X, N_RESULT_MAX)
    print_score(X, r_word_vector)

    print('Calculating rank ensemble')
    all_r = (r_keyword, r_count, r_tfifd,
             r_tfifd_first, r_noun, r_word_vector)
    r_ensemble = ensemble_ranks(all_r, N_RESULT_MAX)
    print_score(X, r_ensemble)

    print('Storing results in {}'.format(OUTPUT))
    store_output(X, r_ensemble)
