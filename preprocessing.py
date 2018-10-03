#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
In this file, we'll load our data, clean it
"""

import pandas as pd


train_sources = [
    {
        'lang': 'en',
        'path': './data/bitcoin_en.json'
    },
    {
        'lang': 'es',
        'path': './data/bitcoin_es.json'
    },
    {
        'lang': 'fr',
        'path': './data/bitcoin_fr.json'
    },
    {
        'lang': 'nl',
        'path': './data/bitcoin_nl.json'
    },   
    {
        'lang': 'ru',
        'path': './data/bitcoin_ru.json'
    },
    {
        'lang': 'de',
        'path': './data/bitcoin_de.json'
    }
]

test_sources = [
    {
        'lang': 'en',
        'path': './data/google_en.json'
    },
    {
        'lang': 'es',
        'path': './data/google_es.json'
    },
    {
        'lang': 'fr',
        'path': './data/google_fr.json'
    },
    {
        'lang': 'nl',
        'path': './data/google_nl.json'
    },   
    {
        'lang': 'ru',
        'path': './data/google_ru.json'
    },
    {
        'lang': 'de',
        'path': './data/google_de.json'
    }
]

def prepare_train_dataset():
    """
    This method prepares training dataset
    """
    X_train = []
    for source in train_sources:
        X_raw = pd.read_json(source['path'])
        X = X_raw[['title', 'content', 'description']]
        X = X.assign(lang=source['lang'])
        X_train.append(X.dropna(subset=['content', 'title']))
    df = pd.concat(X_train)

    X = df.sample(frac=1)
    return X


def prepare_test_dataset():
    """
    This method prepares test dataset
    """
    X_train = []
    for source in test_sources:
        X_raw = pd.read_json(source['path'])
        X = X_raw[['title', 'content', 'description']]
        X = X.assign(lang=source['lang'])
        X_train.append(X.dropna(subset=['content', 'title']))
    df = pd.concat(X_train)

    X = df.sample(frac=1)
    return X
