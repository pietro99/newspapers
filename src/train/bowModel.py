from src.config.settings import DATA_PATH, MODELS_PATH
from src.models.models import MLP
import os
import numpy as np
import torch 
import click
from torch.utils.data import DataLoader, TensorDataset,random_split
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from src.train.main import train
from src.test.main import test

@click.command()
@click.option('-r', '--run_name','run_name', default=str(datetime.now().strftime('%d_%H-%M-%S')), type=str, help='name of the training run (directory to save to)')
@click.option('-ff', '--feature_file','feature_file', required=True, type=str, help='name of the processed file containing bag-of-words')
@click.option('-tf', '--target_file','target_file', required=True, type=str, help='name of the processed file containing bag-of-words')
@click.option('-bs', '--batch_size','batch_size', default=100, type=int, help='size of training batches')
@click.option('--no_train','train_model', is_flag=True, show_default=True, default=True, help='weather to train the model')
@click.option('--no_test','test_model', is_flag=True, show_default=True, default=True, help='weather to test the model')
@click.option('--checkpoint','model_checkpoint_runname', default=None, help='path to a saved model to initialize the model')
def trainBOW(run_name, feature_file, target_file, batch_size, train_model, test_model, model_checkpoint_runname):
    
    '''### PREPERE PATHS ###'''
    targets_path = os.path.join(DATA_PATH, 'processed', target_file)
    bow_path = os.path.join(DATA_PATH, 'processed', feature_file)
    if model_checkpoint_runname is not None:
        model_path = os.path.join(MODELS_PATH, model_checkpoint_runname ,'model.pt')

    '''### LOAD PROCESSED DATA ###'''
    targets =  torch.tensor(np.load(targets_path, allow_pickle=True), dtype=torch.long)
    features = torch.load(bow_path)

    '''### CREATE DATASET AND LOADERS ###'''
    dataset = TensorDataset(features, targets)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset , batch_size=batch_size, pin_memory=True)

    num_samples_in_class = torch.bincount(targets)
    class_weights = len(targets) / (num_samples_in_class * len(num_samples_in_class))
    labels = list(set(targets.tolist()))
    model = MLP(features.shape[1], 200, len(labels))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer =  optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

        
    if model_checkpoint_runname is not None:
        print('initializing model form checkpoint')
        saved_state_dict = torch.load(model_path)
        model.load_state_dict(saved_state_dict)

    if train_model:
        train(model, criterion, optimizer, train_loader, val_loader, labels, run_name)
    if test_model:
        test(model, criterion, test_loader, labels, run_name)

if __name__ == '__main__':
    trainBOW()
