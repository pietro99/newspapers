import torch
from tqdm import tqdm
import numpy as np
def make_bow(sentences:np.ndarray, vocab:dict)->torch.Tensor:
    '''
    creates a bag-of-words representation from a tokenized list of sentences and a vocabulary
    Arguments:
        sentences (numpy array): array of lists where each list is the token in a sentence
        vocab (dictionary): hashmap of word to index of each word in the vocabulary
    Returns:
        vector: Tensor of dimensions NxL where N is the number of sentences and L is the lenght of the vocabulary. 
                For each sentence the the vector stores the number of times each word (from the vocabulary) appears in the sentence.
    '''
    vector = torch.zeros(len(sentences), len(vocab))
    for i, sentence in tqdm(enumerate(sentences), desc='making a bag of words'):
        for token in sentence:
            if token in vocab:
                vector[i, (vocab[token]-1)]+=1

    return vector


def text_to_int(sentences:np.ndarray, vocab:dict)->np.ndarray:
    '''
    transform the tokenized array of sentences to a integer representation where each token is represented by a integer index 
    that's unique in the vocabulary
    Arguments:
        sentences (numpy array): array of lists where each list is the token in a sentence
        vocab (dictionary): hashmap of word to index of each word in the vocabulary
    Returns:
        transformed_array (numpy array): array of tokenized sentences with index representation
    '''
    int_sentences = []
    for sentence in tqdm(sentences, desc='token array -> int array'):
        int_sentence = []
        for token in sentence:
            if token in vocab:
                int_sentence.append(vocab[token])
        int_sentences.append(int_sentence)
    transformed_array = np.array(int_sentences, dtype=object)
    return transformed_array


