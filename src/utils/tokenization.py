from tqdm import tqdm
import numpy as np
from typing import Tuple
from collections import defaultdict
import nltk
import nagisa
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
japanese_pos_tags = {
    'nouns':'名詞',
    'particles':'助詞', 
    'adjectival_nouns':'形状詞',
    'auxiliary_verbs':'助動詞',
    'verbs':'動詞',
}

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
alphanumeric_pattern = re.compile(r'^[a-zA-Z0-9]+$')

def tokenize_english(data:np.ndarray, cutoff:int=0, vocab_size:int=5000) -> Tuple[np.ndarray, dict]:
    '''
    tokenize the ENGLISH sentences contained in a numpy array
    Arguments:
        data (numpy array): array of english sentences
        cutoff (int): treshold of mininum frequency for a word to be included into the vocabulary
        vocab_size (int): if specified, build the vocabulary with the most frequent words up to the vocab_size
    Returns:
        tokens_array (numpy array): array of list of tokens
        vocab (dict): hashmap containing each unique word from all the sentences mapped to a unique integer index
    '''
    all_tokens = []
    for sentence in tqdm(data, desc='tokenizing english'):
        tokens = word_tokenize(sentence)
        stop_words = set([*stopwords.words('english')])
        stop_words_filtered_tokens  = [word for word in tokens if (word.lower() not in stop_words)]
        regex_filtered_tokens  = [word for word in tokens if (alphanumeric_pattern.match(word.lower()))]
        filtered_tokens = list(set(stop_words_filtered_tokens).intersection(regex_filtered_tokens))
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
        all_tokens.append(lemmatized_tokens)
    tokens_array = np.array(all_tokens, dtype=object)
    vocab = build_vocab(tokens_array, vocab_size, cutoff)

    return tokens_array, vocab

def tokenize_japanese(data:np.ndarray, cutoff:int=0, vocab_size:int=5000) -> Tuple[np.ndarray, dict]:
    '''
    tokenize the JAPANESE sentences contained in a numpy array
    Arguments:
        data (numpy array): array of japanese sentences
        cutoff (int): treshold below which a word that is less frequent tan cutoff will not be included in the vocabulary
        vocab_size (int): if specified, build the vocabulary with the most frequent words up to the vocab_size
    Returns:
        tokens_array (numpy array): array of list of tokens
        vocab (dict): hashmap containing each unique word from all the sentences mapped to a unique integer index
    '''         
    tokens_array = np.array([nagisa.extract(t, extract_postags=[japanese_pos_tags['nouns']]).words for t in tqdm(data, desc='tokenizing japanese')], dtype=object)
    vocab = build_vocab(tokens_array, vocab_size, cutoff)

    return tokens_array, vocab


    

def build_vocab(tokens_array, vocab_size, cutoff):
    '''
    takes in the tokenized data and build the vocabulary.
    Arguments:
        tokens_array (numpy array): array of tokens
        vocab_size (int): if specified, build the vocabulary with the most frequent words up to the vocab_size
        cutoff (int): treshold below which a word that is less frequent tan cutoff will not be included in the vocabulary
    Returns:
        vocab (dict): hashmap containing each unique word from all the sentences mapped to a unique integer index
    '''     
    vocabulary = defaultdict(int)
    for tokens in tokens_array:
        for token in tokens:
            vocabulary[token] += 1

    if vocab_size is not None:
        filtered_vocabulary = dict(sorted(vocabulary.items(), key=lambda x: -x[1])[:vocab_size])
    else:
        filtered_vocabulary = {key: value for key, value in vocabulary.items() if value > cutoff}
    vocab = {element: idx+1 for idx, element in enumerate(list(set(filtered_vocabulary.keys())))}
    return vocab