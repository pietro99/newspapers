

from src.config.settings import DATA_PATH
from src.utils.tokenization import tokenize_english, tokenize_japanese
from src.utils.bow import make_bow, text_to_int
import numpy as np
import pandas as pd
import torch
import os
import json
import click
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

@click.command()
@click.option('-d', '--data','file_name', required=True, type=str,help='csv file name')
@click.option('-n','--name', 'name', help='name for the files to be saved')
@click.option('-nb','--no_bow', 'no_bow', is_flag=True, show_default=True, default=False, help='weather to process bag of words')
@click.option('-s','--subset', 'subset',default=None, help='subset of data to select')

def preprocess(file_name, no_bow, name, subset):

    '''### PREPERE REQUIRED PATHS ###'''
    data_file_path = os.path.join(DATA_PATH, 'raw', file_name)
    features_file_path = os.path.join(DATA_PATH, 'processed', name+'_tokens')
    vocab_file_path = os.path.join(DATA_PATH, 'processed', name+'_vocab.json')
    bow_file_path = os.path.join(DATA_PATH, 'processed', name+'_bow.pt')
    targets_file_path = os.path.join(DATA_PATH, 'processed', name+'_targets')
    labels_path = os.path.join(DATA_PATH, 'additional', name+'_labels.json')

    '''### READ DATA AND EXTRACT FEATURES AND LABELS ###'''
    df = pd.read_csv(data_file_path, sep='\t', dtype=str).sample(frac=1).reset_index(drop=True)
    df['text'] = df['text'].astype(str)
    df['source'] = df['source'].astype(str)
    text = df['text'].to_numpy()[:subset]

    sources_cat = df['source'].to_numpy()[:subset]
    sources = label_encoder.fit_transform(sources_cat)

    '''### TOKENIZATION ###'''
    if file_name == 'japanese_news.csv':
        tokens, vocab = tokenize_japanese(text)
    else:
        tokens, vocab = tokenize_english(text)

    category_to_integer_mapping = {str(integer):category for category, integer in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    int_tokens = text_to_int(tokens, vocab)

    empty_indices = np.where([len(lst) == 0 for lst in int_tokens])[0]
    tokens = np.delete(tokens, empty_indices)
    int_tokens = np.delete(int_tokens, empty_indices)
    sources = np.delete(sources, empty_indices)

    '''### SAVE RESULTS ###'''
    with open(labels_path, 'w') as json_file:
        json.dump(category_to_integer_mapping, json_file)

    with open(vocab_file_path, 'w') as json_file:
        json.dump(vocab, json_file)
    
    np.save(features_file_path, int_tokens, allow_pickle=True)
    np.save(targets_file_path, sources, allow_pickle=True)

    '''### BAG OF WORDS ###'''
    if not no_bow:
        bow = make_bow(tokens, vocab)
        torch.save(bow,bow_file_path)

if __name__ == '__main__':
    preprocess()