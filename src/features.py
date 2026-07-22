"""Hand-implemented text features from CSE 158 Assignment 2.

These functions are extracted from the original Assignment2 notebook so the
hand-written TF / DF / TF-IDF logic stays visible and readable.
"""

from collections import defaultdict
import math
import string

def counts(dataset):
    """
    Compute the most frequent unigrams, bigrams, and combined features in the dataset.
    Returns:
        topUnigrams, topBigrams, topBoth
    """
    punctuation = set(string.punctuation)
    
    # Count unigrams and bigrams
    unigramCount = defaultdict(int) # {word_set1 -> count1, ...}
    bigramCount = defaultdict(int)
    
    # 1. Counts for unigrams and bigrams
    for d in dataset:
        # safely extract
        review = d.get("reviewText", "").lower()
        summary = d.get("summary", "").lower()
    
        # combine before cleaning
        text = review + " " + summary
    
        # remove punctuation
        r = ''.join([c if c not in punctuation else ' ' for c in text])
    
        # split all tokens
        ws = r.split()

        
        # Count unigrams
        for w in ws:
            unigramCount[w] += 1

        # Count bigrams
        for i in range(len(ws) - 1):
            bg = ws[i] + " " + ws[i+1]
            bigramCount[bg] += 1
    
    # 2. Count both the unigrams and bigrams
    wordCount = defaultdict(int) # {word1 -> count1, ...}

    for w, c in unigramCount.items():
        wordCount[w] += c
    for bg, c in bigramCount.items():
        wordCount[bg] += c


    # 3. Sort the top 1000 words in uni, bi, and both grams
    mostCommonUnigrams = [w for w, _ in sorted(unigramCount.items(), key=lambda x: -x[1])[:1000]]
    mostCommonBigrams = [w for w, _ in sorted(bigramCount.items(),  key=lambda x: -x[1])[:1000]]
    mostCommonBoth    = [w for w, _ in sorted(wordCount.items(),    key=lambda x: -x[1])[:1000]]

    return mostCommonUnigrams, mostCommonBigrams, mostCommonBoth

def feature(datum, wordId, wordSet, which):
    """
    Build a feature vector (count-based) for a single review.
    The vector includes:
        - unigram counts
        - bigram counts
        - offset term (bias)
    """
    
    feat = [0] * len(wordSet)
    punctuation = set(string.punctuation)

    review = datum.get("reviewText", "").lower()
    summary = datum.get("summary", "").lower()
    
    text = review + " " + summary

    r = ''.join([c if c not in punctuation else ' ' for c in text])
    
    ws = r.split()

    if which in ("unigrams", "both"):
        for w in ws:
            if w in wordId:
                feat[wordId[w]] += 1

    if which in ("bigrams", "both"):
        for i in range(len(ws) - 1):
            bg = ws[i] + " " + ws[i+1]
            if bg in wordId:
                feat[wordId[bg]] += 1

    feat.append(1)  # offset
    return feat

def TF(query, wordSet, which):
    """
    Term Frequency: Binary indicator if word is present
    which: "unigrams", "bigrams", or "both"
    """
    tf = defaultdict(int)
    punctuation = set(string.punctuation)
    
    review = query.get("reviewText", "").lower()
    summary = query.get("summary", "").lower()
    text = review + " " + summary
    r = ''.join([c if c not in punctuation else ' ' for c in text])
    ws = r.split()
    
    # Unigrams
    if which in ("unigrams", "both"):
        for w in ws:
            if w in wordSet:
                tf[w] = 1
    
    # Bigrams
    if which in ("bigrams", "both"):
        for i in range(len(ws) - 1):
            bg = ws[i] + " " + ws[i+1]
            if bg in wordSet:
                tf[bg] = 1
    
    return tf

def DF(dataset, wordSet, which):
    """
    Document Frequency: How many documents contain each word
    which: "unigrams", "bigrams", or "both"
    """
    df = defaultdict(int)
    
    for d in dataset:
        review = d.get("reviewText", "").lower()
        summary = d.get("summary", "").lower()
        text = review + " " + summary
        r = ''.join([c if c not in string.punctuation else ' ' for c in text])
        ws = r.split()
        
        seen = set()  # each vocab only count once in one review
        
        # Unigrams
        if which in ("unigrams", "both"):
            for w in ws:
                if w in wordSet and w not in seen:
                    df[w] += 1
                    seen.add(w)
        
        # Bigrams
        if which in ("bigrams", "both"):
            for i in range(len(ws) - 1):
                bg = ws[i] + " " + ws[i+1]
                if bg in wordSet and bg not in seen:
                    df[bg] += 1
                    seen.add(bg)
    
    return df

def TFIDF(query, df, dataset, wordSet, which):
    """
    Compute TF-IDF vector for a single document
    """
    tf = TF(query, wordSet, which)  # 传入 which
    N = len(dataset)
    
    tfidfQuery = [
        tf[w] * math.log2(N / df[w]) if df[w] > 0 else 0
        for w in wordSet
    ]
    return tfidfQuery
