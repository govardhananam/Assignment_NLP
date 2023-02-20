import collections
import random
from process_corpus import preprocessing
def compute_unigrams(text):
    # split the text into words
    words = text.split()
    # use collections.Counter to compute the frequency of each word
    unigram_counts = collections.Counter(words)
    return dict(unigram_counts)

def compute_bigrams(text):
    # split the text into words
    words = text.split()
    # use collections.defaultdict to store the frequency of each bigram
    bigram_counts = collections.defaultdict(int)
    # loop through the words and compute the frequency of each bigram
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigram_counts[bigram] += 1
    return dict(bigram_counts)


text = "loop through the words and compute the frequency of each"
unigrams = compute_unigrams(text)
print(unigrams)

bigrams = compute_bigrams(text)
print(bigrams)


# Problem 2
def bigrams(sentences):
    bigram_counts = collections.defaultdict(int)
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            if '<s>' in bigram or '<\\s>' in bigram:
                continue
            bigram_counts[bigram] += 1
    return dict(bigram_counts)

def unigrams(sentences):
    unigram_counts = collections.defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            if word in ['<s>', '<\\s>']:
                continue
            unigram_counts[word] += 1
    return dict(unigram_counts)

#a and b
hamlet_sentences = preprocessing('hamlet.txt')
hamlet_corpus_size = len(hamlet_sentences)
print("Corpus size:", hamlet_corpus_size)
hamlet_bigrams = bigrams(hamlet_sentences)
hamlet_bigrams_sorted = sorted(hamlet_bigrams.items(), key=lambda x: x[1], reverse=True)
print("Top 10 most frequent bigrams in Hamlet corpus:")
for i in range(10):
    print(hamlet_bigrams_sorted[i][0], hamlet_bigrams_sorted[i][1])


macbeth_sentences = preprocessing('macbeth.txt')
mac_corpus_size = len(macbeth_sentences)
print("Corpus size:", mac_corpus_size)
macbeth_bigrams = bigrams(macbeth_sentences)
macbeth_bigrams_sorted = sorted(macbeth_bigrams.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 most frequent bigrams in Macbeth corpus:")
for i in range(10):
    print(macbeth_bigrams_sorted[i][0], macbeth_bigrams_sorted[i][1])



#c

def bigram_probability(bigram, bigrams_dict, corpus_size):
    return bigrams_dict[bigram] / corpus_size

pair_bigram = ('hath', 'discretion')

try:
    hamlet_bigram_prob = bigram_probability(pair_bigram, hamlet_bigrams, hamlet_corpus_size)
    print("Unsmoothed bigram probability for 'hath' and 'discretion' in Hamlet corpus:", hamlet_bigram_prob)
except KeyError:
    print("Unsmoothed bigram probability for 'hath' and 'discretion' in Hamlet corpus: 0")
try:
    macbeth_bigram_prob = bigram_probability(pair_bigram, macbeth_bigrams, mac_corpus_size)
    print("Unsmoothed bigram probability for 'hath' and 'discretion' in Macbeth corpus:", macbeth_bigram_prob)
except KeyError:
    print("Unsmoothed bigram probability for 'hath' and 'discretion' in Macbeth corpus: 0")


#d

def compare_unigrams(corpus1, corpus2):
    unigram_counts1 = collections.defaultdict(int)
    unigram_counts2 = collections.defaultdict(int)

    for sentence in corpus1:
        for word in sentence:
            if word not in ['<s>', '<\\s>']:
                unigram_counts1[word] += 1
    
    for sentence in corpus2:
        for word in sentence:
            if word not in ['<s>', '<\\s>']:
                unigram_counts2[word] += 1
    
    unique_unigrams1 = set(unigram_counts1.keys()) - set(unigram_counts2.keys())
    unique_unigrams2 = set(unigram_counts2.keys()) - set(unigram_counts1.keys())

    return unique_unigrams1, unique_unigrams2



hamlet_unique, macbeth_unique = compare_unigrams(hamlet_sentences, macbeth_sentences)

print("Unique unigrams in Hamlet corpus:", hamlet_unique)
print("Unique unigrams in Macbeth corpus:", macbeth_unique)


def compare_bigrams(bigram1, bigram2):
    common_bigrams = set(bigram1.keys()).intersection(set(bigram2.keys()))
    print("Common bigrams:", common_bigrams)

    differences_bigrams = set(bigram1.keys()).symmetric_difference(set(bigram2.keys()))
    print("Bigrams that are different:", differences_bigrams)

compare_bigrams(hamlet_bigrams, macbeth_bigrams)




#problem 3

import random

def generate_random_sentence_unsmoothed_bigram(unigrams, bigrams, max_length=20):
    sentence = ['<s>']
    while len(sentence) < max_length and sentence[-1] != '<\\s>':
        previous_word = sentence[-1]
        if previous_word in bigrams:
            next_word = random.choice(list(bigrams[previous_word].keys()))
            sentence.append(next_word)
        else:
            next_word = random.choice(list(unigrams.keys()))
            sentence.append(next_word)
    return ' '.join(sentence)

def compute_perplexity(sentence, unigrams, bigrams):
    tokens = sentence.split()
    perplexity = 1.0
    for i in range(1, len(tokens)):
        previous_word = tokens[i-1]
        current_word = tokens[i]
        bigram_count = bigrams[previous_word][current_word] if previous_word in bigrams and current_word in bigrams[previous_word] else 0
        unigram_count = unigrams[current_word]
        probability = (bigram_count + 1) / (unigram_count + len(unigrams))
        perplexity *= 1/probability
    perplexity = pow(perplexity, 1/len(tokens))
    return perplexity


import random

hamlet_unigrams = unigrams(hamlet_sentences)
random.seed(17)
sentence = generate_random_sentence_unsmoothed_bigram(hamlet_unigrams, hamlet_bigrams, 10)
print(sentence)


perplexity = compute_perplexity(sentence, hamlet_unigrams, hamlet_bigrams)
print(perplexity)


#problem 4

def unigram(tokens, smooth=True):
    if smooth:
        counts = collections.defaultdict(lambda: 1)
        for token in tokens:
            counts[token] += 1
    else:
        counts = collections.defaultdict(int)
        for token in tokens:
            counts[token] += 1
    total = sum(counts.values())
    probs = {token: count/total for token, count in counts.items()}
    return probs


def bigram(tokens, smooth=True):
    if smooth:
        counts = collections.defaultdict(lambda: collections.defaultdict(lambda: 1))
    else:
        counts = collections.defaultdict(lambda: collections.defaultdict(int))
    for i in range(len(tokens) - 1):
        token1 = tokens[i]
        token2 = tokens[i+1]
        counts[token1][token2] += 1
    probs = {}
    for token1, next_tokens in counts.items():
        total = sum(next_tokens.values())
        probs[token1] = {token2: count/total for token2, count in next_tokens.items()}
    return probs



bigram_probs = bigram(hamlet_bigrams, smooth=True)
prob_hd = bigram_probs['hath'].get('discretion', 0)
print(f"The smoothed bigram probability of 'hath' and 'discretion' in the hamlet corpus is {prob_hd:.4f}")




