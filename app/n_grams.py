import math
import string
import html
import concurrent.futures
from collections import Counter, defaultdict
from datetime import datetime
from itertools import islice, chain

import ujson
import pytz
from tqdm import tqdm

import twokenize
from nltk.corpus import stopwords

EXCLUDED_TOKENS = set(stopwords.words('english'))
EXCLUDED_TOKENS.add('')


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def tokenize_tweet(text):
    for s in twokenize.tokenizeRawTweetText(process_tweet_text(text.lower())):
        token = s.strip().strip(string.punctuation)
        if (token not in EXCLUDED_TOKENS
                and not token.startswith('https://t.co/')):
            yield token


def process_tweet_text(text):
    return html.unescape(text)\
        .replace(' & ', ' and ')\
        .replace('’', "'")\
        .replace('‘', "'")\
        .replace('“', '"')\
        .replace('”', '"')\
        .replace('—', '-')


def tweet_text(tweet):
    if 'retweeted_status' in tweet:
        return process_tweet_text(tweet['retweeted_status']['full_text'])
    return process_tweet_text(tweet['full_text'])


def tweet_str_to_n_grams(line, n_gram_lens=[2]):
    tweet = ujson.loads(line)
    created_at = datetime.strptime(tweet['created_at'],
                                   '%a %b %d %H:%M:%S +0000 %Y')\
        .replace(tzinfo=pytz.UTC)
    text = tweet_text(tweet)
    tokens = list(tokenize_tweet(text))
    return {
        'acct_id': tweet['user']['id'],
        'timestamp': created_at,
        'n_grams': chain.from_iterable(
            [window(tokens, n_gram) for n_gram in n_gram_lens])
    }


def tweet_text_to_n_grams(text, n_gram_lens):
    tokens = list(tokenize_tweet(text))
    return {n_gram_len:
            window(tokens, n_gram_len) for n_gram_len in n_gram_lens}


def timestamp_to_timespan(timestamp, timespans):
    for timespan in timespans:
        if timespan[0] <= timestamp <= timespan[1]:
            return timespan
    return None


def tweets_file_to_n_gram_by_timespan_counts_parallel(
        filename, timespans, n_gram_lens=[2], limit=None):
    futures = []
    timespan_to_n_gram_users = defaultdict(lambda: defaultdict(set))
    timespan_to_n_gram_user_counts = defaultdict(Counter)
    with open(filename) as f, \
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

        for line in tqdm(f, desc='Making jobs'):
            futures.append(
                    executor.submit(tweet_str_to_n_grams, line, n_gram_lens))

            if limit and len(futures) >= limit:
                break

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc='Running'):
            result = future.result()
            acct_id, timestamp, n_grams = (
                    result['acct_id'], result['timestamp'], result['n_grams'])
            timespan = timestamp_to_timespan(timestamp, timespans)

            if timespan is None:
                continue

            for n_gram in n_grams:
                if acct_id in timespan_to_n_gram_users[timespan][n_gram]:
                    continue
                timespan_to_n_gram_user_counts[timespan][n_gram] += 1
                timespan_to_n_gram_users[timespan][n_gram].add(acct_id)
    return timespan_to_n_gram_user_counts


def extract_tweet_text(tweet):
    if 'retweeted_status' in tweet:
        return tweet['retweeted_status']['full_text']
    return tweet['full_text']


def tweets_to_n_gram_counts(tweets, n_gram_lens=[2]):
    n_gram_users = defaultdict(set)
    n_gram_user_counts = defaultdict(Counter)
    for tweet in tqdm(tweets.values(), miniters=5000,
                      mininterval=1, desc='Extracting n-grams'):
        acct_id = tweet['user']['id']
        tweet_n_grams = tweet_text_to_n_grams(
                extract_tweet_text(tweet), n_gram_lens)

        for n_gram_len, n_grams in tweet_n_grams.items():
            for n_gram in n_grams:
                if acct_id in n_gram_users[n_gram]:
                    continue
                n_gram_user_counts[n_gram_len][n_gram] += 1
                n_gram_users[n_gram].add(acct_id)
    return n_gram_user_counts


def tweets_file_to_n_gram_by_timespan_counts(
        filename, timespans, n_gram_lens=[2], limit=None):
    timespan_to_n_gram_users = defaultdict(lambda: defaultdict(set))
    timespan_to_n_gram_user_counts = defaultdict(Counter)
    with open(filename) as f:
        for i, line in tqdm(enumerate(f), miniters=5000, mininterval=1):
            if limit and i + 1 >= limit:
                break

            result = tweet_str_to_n_grams(line, n_gram_lens)
            acct_id, timestamp, n_grams = (
                    result['acct_id'], result['timestamp'], result['n_grams'])
            timespan = timestamp_to_timespan(timestamp, timespans)

            if timespan is None:
                continue

            for n_gram in n_grams:
                if acct_id in timespan_to_n_gram_users[timespan][n_gram]:
                    continue
                timespan_to_n_gram_user_counts[timespan][n_gram] += 1
                timespan_to_n_gram_users[timespan][n_gram].add(acct_id)
    return timespan_to_n_gram_user_counts


def z_score_for_n_gram(n_gram, community_counts, global_counts,
                       community_size=1000, global_size=1000):
    mu_g = global_counts[n_gram] / global_size
    mu_c = community_counts[n_gram] / community_size
    sigma_2_g = mu_g * (1-mu_g)
    z_score = (mu_c - mu_g) / (math.sqrt(sigma_2_g) /
                               math.sqrt(community_size))
    return z_score


def gen_z_scores(n_gram_counts, n_gram_baseline_counts,
                 limit=int(1e5), community_size=1000, global_size=1000):
    z_scores = {}
    for n_gram, _ in tqdm(n_gram_counts.most_common(limit), desc='Z-scoring'):
        try:
            z_scores[n_gram] = z_score_for_n_gram(
                    n_gram, n_gram_counts, n_gram_baseline_counts,
                    community_size, global_size)
        except Exception:
            continue
    return z_scores
