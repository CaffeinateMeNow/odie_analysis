import pickle
import logging
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from scattertext.termscoring.ScaledFScore import ScaledFScore

from n_grams import tweets_to_n_gram_counts, extract_tweet_text,\
    tokenize_tweet, tweet_text_to_n_grams
import config


def get_cohort_tweet_text(cohort_prefix, timespan):
    client = Elasticsearch(config.ELASTICSEARCH_HOST)

    s = Search(using=client, index=f'{cohort_prefix}*').query("match_all")\
        .filter({"range":
                {"@timestamp": {"gte": timespan[0], "lte": timespan[1]}}})
    tweets = (hit.to_dict() for hit in s.scan())
    return extract_tweet_text(tweets)


def get_cohort_tweets(cohort_prefix, timespan):
    client = Elasticsearch(config.ELASTICSEARCH_HOST)

    s = Search(using=client, index=f'{cohort_prefix}*').query("match_all")\
        .filter({"range":
                {"@timestamp": {"gte": timespan[0], "lte": timespan[1]}}})
    logging.info(s)
    tweets = {}
    for hit in tqdm(s.scan(), desc='Reading tweets'):
        hit = hit.to_dict()
        tweets[hit['id']] = hit
    return tweets


def tweets_to_lsh(tweet_text):
    # Create LSH index
    lsh = MinHashLSH(threshold=0.95, num_perm=128)

    for tweet_id, text in tqdm(
            tweet_text.items(), smoothing=0, desc='Hashing'):
        m1 = MinHash(num_perm=128)
        words = set(tokenize_tweet(text))
        for word in words:
            m1.update(word.encode('utf8'))
        lsh.insert(tweet_id, m1)
    return lsh


def tweets_to_doc_groups(lsh):
    tweet_id_to_group = {}
    for i, hashtable in enumerate(lsh.hashtables):
        for j, (key, tweets) in enumerate(dict(hashtable).items()):
            for tweet in tweets:
                tweet_id_to_group[tweet] = i*j + j
    return tweet_id_to_group


def doc_groups_to_docs(tweet_id_to_group, tweet_text):
    docs = defaultdict(str)
    for tweet_id, group_i in tqdm(
            tweet_id_to_group.items(), desc='Prepping docs'):
        docs[group_i] += f'{tweet_text[tweet_id]}\n'

    group_to_tweet_ids = defaultdict(list)
    for tweet_id, group in tweet_id_to_group.items():
        group_to_tweet_ids[group].append(tweet_id)
    return docs


def tweets_to_docs(tweet_text, lsh_cache_name):
    if Path(lsh_cache_name).exists():
        lsh = pd.read_pickle(lsh_cache_name)
    else:
        lsh = tweets_to_lsh(tweet_text)
    with open(lsh_cache_name, 'wb') as f:
        pickle.dump(lsh, f, protocol=4)
    tweet_id_to_group = tweets_to_doc_groups(lsh)
    docs = doc_groups_to_docs(tweet_id_to_group, tweet_text)
    tokenized_docs = {}
    for i, doc in docs.items():
        tokens = list(tokenize_tweet(doc))
        if len(tokens) > 0:
            tokenized_docs[i] = tokens
    return tokenized_docs, tweet_id_to_group


def calc_top_n_grams(tweets, n_gram_lens=[1, 2, 3], k=50):
    output = defaultdict(Counter)
    for tweet in tqdm(tweets.values(), desc='Tokenizing'):
        for n_gram, tokens in tweet_text_to_n_grams(
                tweet, n_gram_lens).items():
            output[n_gram].update(tokens)
    return {n_gram: output[n_gram].most_common(k) for n_gram in n_gram_lens}


def extract_accts(tweets):
    return set([t['user']['id'] for t in tweets.values()])


def n_gram_counter_to_dict(counter):
    output = {}
    for n_gram, count in dict(counter).items():
        output[' '.join(n_gram)] = count
    return output


def summarize_cohort(cohort_prefix, timespan, top_k=50):
    tweets = get_cohort_tweets(cohort_prefix, timespan)
    n_grams = tweets_to_n_gram_counts(tweets, [1, 2, 3])
    return {
        'n_accounts': len(extract_accts(tweets)),
        'n_tweets': len(tweets),
        'top_unigrams': n_gram_counter_to_dict(n_grams[1].most_common(top_k)),
        'top_bigrams': n_gram_counter_to_dict(n_grams[2].most_common(top_k)),
        'top_trigrams': n_gram_counter_to_dict(n_grams[3].most_common(top_k))
    }


def calc_f1_scores(n_grams_a, n_grams_b):
    all_n_grams_a = sum(n_grams_a.values(), Counter())
    all_n_grams_b = sum(n_grams_b.values(), Counter())
    vocab = sorted(list((all_n_grams_a + all_n_grams_b).keys()))
    n_gram_a_counts = [all_n_grams_a[v] for v in vocab]
    n_gram_b_counts = [all_n_grams_b[v] for v in vocab]
    scores = ScaledFScore.get_scores(np.array(n_gram_a_counts),
                                     np.array(n_gram_b_counts))
    return pd.DataFrame({
        'cohort_a_count': n_gram_a_counts,
        'cohort_b_count': n_gram_b_counts,
        'scaled_f1': scores,
    }, index=vocab).sort_values(
        ['scaled_f1', 'cohort_a_count', 'cohort_b_count'],
        ascending=[True, True, False])


def compare_cohorts(cohort_prefix_a, timespan_a,
                    cohort_prefix_b, timespan_b,
                    top_k=50):
    tweets_a = get_cohort_tweets(cohort_prefix_a, timespan_a)
    n_grams_a = tweets_to_n_gram_counts(tweets_a, [1, 2, 3])
    accts_a = extract_accts(tweets_a)

    tweets_b = get_cohort_tweets(cohort_prefix_b, timespan_b)
    n_grams_b = tweets_to_n_gram_counts(tweets_b, [1, 2, 3])
    accts_b = extract_accts(tweets_b)

    f1_scores = calc_f1_scores(n_grams_a, n_grams_b)
    most_characteristic_a = f1_scores.sort_values(
        ['scaled_f1', 'cohort_a_count', 'cohort_b_count'],
        ascending=[False, False, True]).head(top_k)
    most_characteristic_a = list(
        map(lambda x: ' '.join(x), most_characteristic_a.index.values))
    most_characteristic_b = f1_scores.sort_values(
        ['scaled_f1', 'cohort_a_count', 'cohort_b_count'],
        ascending=[True, True, False]).head(top_k)
    most_characteristic_b = list(
        map(lambda x: ' '.join(x), most_characteristic_b.index.values))

    return {
        'summary_a': {
            'n_accounts': len(accts_a),
            'n_tweets': len(tweets_a),
            'top_unigrams':
                n_gram_counter_to_dict(n_grams_a[1].most_common(top_k)),
            'top_bigrams':
                n_gram_counter_to_dict(n_grams_a[2].most_common(top_k)),
            'top_trigrams':
                n_gram_counter_to_dict(n_grams_a[3].most_common(top_k))
        },
        'summary_b': {
            'n_accounts': len(accts_b),
            'n_tweets': len(tweets_b),
            'top_unigrams':
                n_gram_counter_to_dict(n_grams_b[1].most_common(top_k)),
            'top_bigrams':
                n_gram_counter_to_dict(n_grams_b[2].most_common(top_k)),
            'top_trigrams':
                n_gram_counter_to_dict(n_grams_b[3].most_common(top_k))
        },
        'f1_scores': {
            'most_characteristic_a': most_characteristic_a,
            'most_characteristic_b': most_characteristic_b,
        }
    }
