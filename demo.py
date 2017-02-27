# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:03:23 2017

@author: joostbloom
"""
import os
import random

import pandas as pd


DATASET = 'Acme_dataset.xlsx'
OUTPUT = 'Acme_dataset_pre.txt'

# Number of responses to show
SHOWMAX = 5


def load_output():
    ''' Loads ranked responses per question from output file '''
    with open(OUTPUT, 'r') as f:

        all_out = {}

        for r in f:

            qid, rids = r.split(':')
            qid = int(qid)

            rids = [int(x) for x in rids.split(', ')]

            all_out[qid] = rids

    return all_out


def load_data():
    ''' Loads data as Pandas dataframe '''

    X = pd.read_excel(DATASET)

    # Ignore last row
    X.drop(X.index[-1], inplace=True)

    X['Question Key'] = X['Question Key'].astype(int)

    return X


def print_question(X, key):
    ''' Prints question for given key '''

    q_str = X.loc[X['Question Key'] == key, 'Question Text'].values[0]
    q_str = q_str.encode('ascii', 'ignore')
    print('You asked: {} ({})\n'.format(q_str, key))


def print_top_answers(X, R, key):
    ''' Prints answers for given key '''

    print('My answers:')

    rs = R[key][:SHOWMAX]

    for i, r in enumerate(rs):
        r_str = X.loc[X['Question Key'] == r, 'Response'].values[0]
        r_str = r_str.encode('ascii', 'ignore')
        print('{} - {} ({})\n'.format(i, r_str, r))

if __name__ == '__main__':

    if not os.path.exists(OUTPUT):
        print('Output file not found. Executing precalculate.py for you...')
        os.system('python precalculate.py')
        print('Done! Start asking questions...')

    # Load questions and pre-calculated response ranks
    X = load_data()
    R = load_output()
    keys = X['Question Key'].values

    while True:

        nq = raw_input('Enter a question key [suggestion: {}, "q" to quit]:'
                       .format(random.choice(keys)))

        if nq == 'q':
            print('Bye!')
            break

        try:
            nq = int(nq)
            print_question(X, nq)
            print_top_answers(X, R, nq)
        except:
            print('{} is an unknown question key. Please try again.'
                  .format(nq))
