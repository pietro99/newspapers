from src.config.settings import DATA_PATH, MODELS_PATH
from src.models.models import NearestMeanClassifier
from src.test.main import test
import os
import numpy as np
import torch 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from datetime import datetime
import click

@click.command()
@click.option('-r', '--run_name','run_name', default=str(datetime.now().strftime('%d_%H-%M-%S')), type=str, help='name of the training run (directory to save to)')
@click.option('-ff', '--feature_file','feature_file', required=True, type=str, help='name of the processed file containing bag-of-words')
@click.option('-tf', '--target_file','target_file', required=True, type=str, help='name of the processed file containing bag-of-words')
@click.option('-bs', '--batch_size','batch_size', default=100, type=int, help='size of training batches')
def neareastMean(run_name, feature_file, target_file, batch_size):
    
    '''### PREPERE PATHS ###'''
    features_path = os.path.join(DATA_PATH, 'processed', feature_file)
    targets_path = os.path.join(DATA_PATH, 'processed', target_file)
    model_path = os.path.join(MODELS_PATH, run_name ,'model.pt')
    os.makedirs(os.path.dirname(model_path))
    '''### LOAD PROCESSED DATA ###'''
    tokens = np.load(features_path, allow_pickle=True)
    lenghts = torch.tensor([[len(text)] for text in tokens], dtype=torch.float)
    targets =  torch.tensor(np.load(targets_path, allow_pickle=True), dtype=torch.float)

    '''### CREATE DATASET AND LOADERS ###'''
    dataset = TensorDataset(lenghts, targets)
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_set , batch_size=batch_size, pin_memory=True)
    x, y = train_set.dataset.tensors

    num_samples_in_class = torch.bincount(targets.clone().to(torch.long))
    class_weights = len(targets) / (num_samples_in_class * len(num_samples_in_class))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    nmc = NearestMeanClassifier(y.unique().size(0), x.shape[1])
    nmc.compute_means(x, y)
    torch.save(nmc.state_dict(), model_path)
    labels = list(set(targets.tolist()))
    test(nmc, criterion, test_loader, labels, run_name)


if __name__ == '__main__':
    neareastMean()