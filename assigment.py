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

def generate_sentence(bigram_model, max_length=10, start_token='<s>', end_token='<\\s>'):
    sentence = [start_token]
    current_token = start_token
    while len(sentence) < max_length and current_token != end_token:
        # get the possible next tokens for the current token
        next_tokens = [next_token for (token, next_token) in bigram_model.keys() if token == current_token]
        # select a random next token from the possible tokens
        next_token = random.choice(next_tokens)
        sentence.append(next_token)
        current_token = next_token
    sentence.append(end_token)
    return sentence

random.seed(17)
random_sentence = generate_sentence(hamlet_bigrams)
print(random_sentence)




def compute_unigrams(text, smoothing=False, delta=1):
    # split the text into words
    words = text.split()
    # use collections.Counter to compute the frequency of each word
    unigram_counts = collections.Counter(words)
    if smoothing:
        # add delta to the frequency of each word
        for word in set(words):
            unigram_counts[word] += delta
    # normalize the frequencies to obtain probabilities
    total = sum(unigram_counts.values())
    unigram_probs = {word: count/total for word, count in unigram_counts.items()}
    return unigram_probs

def compute_bigrams(text, smoothing=False, delta=1):
    # split the text into words
    words = text.split()
    # use collections.defaultdict to store the frequency of each bigram
    bigram_counts = collections.defaultdict(int)
    # loop through the words and compute the frequency of each bigram
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigram_counts[bigram] += 1
    if smoothing:
        # add delta to the frequency of each bigram
        for bigram in bigram_counts.keys():
            bigram_counts[bigram] += delta
    # normalize the frequencies to obtain probabilities
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        unigram_count = sum(bigram_counts[bigram[i:i+1]] for i in range(2))
        bigram_probs[bigram] = count / unigram_count
    return bigram_probs

def generate_random_sentence(bigram_probs, start_token="<s>", end_token="</s>", max_length=10):
    sentence = [start_token]
    current_word = start_token
    while current_word != end_token and len(sentence) < max_length:
        next_word = random.choices(list(bigram_probs[current_word].keys()), weights=list(bigram_probs[current_word].values()))[0]
        sentence.append(next_word)
        current_word = next_word
    return sentence

def compute_perplexity(sentence, bigram_probs):
    prob = 1
    for i in range(len(sentence) - 1):
        bigram = (sentence[i], sentence[i+1])
        prob *= bigram_probs.get(bigram, 0)
    perplexity = pow(1/prob, 1/len(sentence))
    return perplexity


