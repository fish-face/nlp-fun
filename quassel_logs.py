#!/usr/bin/env python3
import numpy as np
import psycopg2
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import textacy
import re
import json
import os
from glob import glob
from base64 import b64encode
import hashlib

import textacy.keyterms

DBHOST = "localhost"
PASSWORD = open('password.txt', 'r').read()
DUMMY_NICK = "John"
BUFFERS = ('#notpron', '#quassel', '#latex', '#Ins.general-chat', '#Str.general-chat')
SAMPLES = 4000
PORT = 5433
en = textacy.load_spacy('en_core_web_sm')

class Db(object):
    cache_dir = 'sql_cache'
    def __init__(self, invalidate=False):
        self.cursor = None
        if invalidate:
            self._cache = {}
        else:
            self.load_cache()
        self.dirty = []

    def load_cache(self):
        self._cache = {}
        for file in glob(os.path.join(self.cache_dir, '*')):
            fn = os.path.split(file)
            with open(file, 'r') as fd:
                self._cache[fn[-1]] = json.load(fd)

    def cache_key(self, key):
        return b64encode(hashlib.md5(key.encode('utf-8')).digest(), altchars=b'_-').decode('ascii').replace('=', '')
        #return b64encode(key.encode('ascii')).replace(b'=', b'').decode('ascii')

    def write_cache(self, key, value):
        fn = self.cache_key(key)
        self._cache[fn] = value
        with open(os.path.join(self.cache_dir, fn), 'w') as fd:
            json.dump(value, fd)

    def connect(self):
        print('connecting')
        conn = psycopg2.connect(database="quassel", user="quassel", password=PASSWORD, host=dbhost, port=PORT)
        self.cursor = conn.cursor()
        print('connected')

    def execute(self, query, use_cache=True, write_cache=True):
        try:
            return self._cache[self.cache_key(query)]
        except KeyError:
            print('cache miss')
            if self.cursor is None:
                self.connect()
            self.cursor.execute(query)
            value = list(self.cursor.fetchall())
            self.write_cache(query, value)
            return value

    def get(self):
        # Perform a stratified sample from the buffers in BUFFERS. Doing this as a UNION is faster
        # than with window functions.
        query="""
WITH q AS (
    SELECT buffer.buffername, backlog.message
    FROM backlog JOIN buffer ON buffer.bufferid = backlog.bufferid
    WHERE backlog.type IN (1, 2, 4) AND
        LENGTH(backlog.message) > 10 AND
        buffer.buffername IN %s
    )""" % str(BUFFERS)
        query += '\nUNION'.join(["""
(SELECT message, buffername
FROM q
WHERE buffername = '%s'
LIMIT %d)""" % (buffer, SAMPLES) for buffer in BUFFERS])
        #print(query)
        return self.execute(query)

    def sender_to_nick(self, sender):
        return sender[:sender.find('!')]

    def get_nicks(self):
        nicks = self.execute("SELECT sender, LENGTH(sender) FROM sender WHERE POSITION('!' in sender) > 3")
        nicks = set(self.sender_to_nick(r[0]) for r in nicks)
        return {n for n in nicks if n not in en.vocab}

def filter(line):
    line = nickre.sub(DUMMY_NICK, line)
    #line = textacy.Doc((w if w not in NICKS else DUMMY_NICK for w in line), lang=en)
    #line = textacy.Doc(line, lang=en)
    return line

def process(doc):
    return doc.to_bag_of_terms()

def train(data, labels, nicks):
    vectorizer = CountVectorizer(ngram_range=(1,3))
    tokenize = vectorizer.build_tokenizer()
    preprocess = vectorizer.build_preprocessor()
    stop_words = set()
    for n in nicks:
        stop_words |= set(tokenize(preprocess(n)))
    vectorizer.stop_words = stop_words

    tfidf = sklearn.feature_extraction.text.TfidfTransformer()
    scaler = StandardScaler(with_mean=False)
    classifier = SGDClassifier(penalty='l1', alpha=0.002, random_state=43, max_iter=5)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', tfidf),
        ('scaler', scaler),
        ('clf', classifier)])
    parameters = {
        'vect__ngram_range': [(1,1),(1,2),(1,3)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-1, 1e-2, 1e-3),
        'clf__penalty': ('l1', 'l2')
    }
    #if cv is low, best_score_ will be low, but the training/test scores may still be high.
    search = GridSearchCV(pipeline, parameters, cv=3, iid=True, n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=43, stratify=labels)
    #model.fit(X_train, list(y_train))

    model = search.fit(X_train, y_train)
    print('Best parameters:', model.best_params_)
    print('Best CV score: %.2f' % (model.best_score_))
    print('Training score: %.2f' % (model.score(X_train, y_train)))
    print('Test score: %.2f' % (model.score(X_test, y_test)))
    predicted = model.predict(X_test)
    print(metrics.classification_report(y_test, predicted))
    print(metrics.confusion_matrix(y_test, predicted))

    return model

def main():
    print("starting")
    db = Db()
    global nickre
    print("getting nicks")
    nicks = list(db.get_nicks())
    pat = '|'.join(re.escape(n) for n in nicks)
    pat = '((?<=\W)|^)(%s)((?=\W)|$)' % pat
    print("compiling pattern")
    #nickre = re.compile(pat, re.I)

    print("getting data")
    rows = db.get()
    rows = np.array(rows)
    print("training")
    #data = [filter(r) for r in rows[:,0]]
    model = train(rows[:,0], rows[:,1], nicks)
    #for r in db.get()[-10:]:
    #    print(r[2])
    #    doc = filter(r[2], nickre)
    #    print(doc)
    #    print(list(textacy.keyterms.sgrank(doc, ngrams=(1,), normalize='lemma')))

if __name__ == '__main__':
    main()
