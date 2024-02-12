from src.config.settings import RESULTS_PATH, MODELS_PATH
import os
import numpy as np
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils.metrics import accuracy_per_class
import pandas as pd
import warnings

# Filter out the specific warning for np.nanmean
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)


def train(model:nn.Module, 
          criterion:nn.Module, 
          optimizer:optim.Optimizer, 
          train_loader:DataLoader, 
          val_loader:DataLoader, 
          labels:list, 
          run_name:str,
          patience:int = 10,
          num_epochs:int=250) -> None:
    
    '''##### INITIALIZE FILE PATHS #####'''
    model_path = os.path.join(MODELS_PATH, run_name ,'model.pt')
    results_path = os.path.join(RESULTS_PATH, 'train', run_name, 'result_metrics.csv')
    os.makedirs(os.path.dirname(model_path), exist_ok=False)
    os.makedirs(os.path.dirname(results_path), exist_ok=False)

    pd.DataFrame({'loss':[], 
                'val_loss':[], 
                **{f'val_f1s_{i}': [] for i in range(len(labels))},
                **{f'f1s_{i}': [] for i in range(len(labels))},
                **{f'val_precisions_{i}': [] for i in range(len(labels))},
                **{f'precisions_{i}': [] for i in range(len(labels))},
                **{f'val_recalls_{i}': [] for i in range(len(labels))},
                **{f'recalls_{i}': [] for i in range(len(labels))},
                **{f'accuracies_{i}': [] for i in range(len(labels))},
                **{f'val_accuracies_{i}': [] for i in range(len(labels))},
                }).to_csv(results_path, mode='w', index=False)

    best_loss = 10**6 #for early stopping
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        losses, accuracies, f1s, recalls, precisions = [],[],[],[],[]
        val_losses, val_accuracies, val_f1s, val_recalls, val_precisions = [],[],[],[],[]
        
        '''##### TRAINING LOOP#####'''
        model.train()
        for batch in tqdm(train_loader, desc='training', leave=False):
            optimizer.zero_grad()
            features, targets = batch
            out = model.forward(features)
            loss = criterion(out,targets.to(torch.long))
            predicted_class = torch.argmax(out, dim=1)
            loss.backward()
            optimizer.step()

            precisions.append(precision_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
            f1s.append(f1_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
            recalls.append(recall_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
            accuracies.append(accuracy_per_class(targets, predicted_class, labels=labels))

            losses.append(loss.item())


        '''##### VALIDATION LOOP #####'''
        model.eval()
        for batch in tqdm(val_loader, desc='validation', leave=False ):
            optimizer.zero_grad()
            features, targets = batch
            out = model.forward(features)
            loss = criterion(out,targets.to(torch.long))
            predicted_class = torch.argmax(out, dim=1)
            val_precisions.append(precision_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
            val_f1s.append(f1_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
            val_recalls.append(recall_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
            val_accuracies.append(accuracy_per_class(targets, predicted_class, labels=labels))
            val_losses.append(loss.item())

        '''##### EARLY STOPPING  #####'''
        mean_val_loss = np.array(val_losses).mean()
        if mean_val_loss <= best_loss:
            best_loss = mean_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
    
        if epoch-best_epoch >= patience:
            print(f'Early stopping triggered at Epoch: {epoch}')
            break;
        

        '''##### SAVE METRICS  #####'''
        val_f1_dict = {f'val_f1s_{i}': v for i, v in enumerate(np.nanmean(np.array(val_f1s), axis=0))}
        f1_dict = {f'f1s_{i}': v for i, v in enumerate(np.nanmean(np.array(f1s), axis=0))}

        val_precisions_dict = {f'val_precisions_{i}': v for i, v in enumerate(np.nanmean(np.array(val_precisions), axis=0))}
        precisions_dict = {f'precisions_{i}': v for i, v in enumerate(np.nanmean(np.array(precisions), axis=0))}

        val_recalls_dict = {f'val_recalls_{i}': v for i, v in enumerate(np.nanmean(np.array(val_recalls), axis=0))}
        recalls_dict = {f'recalls_{i}': v for i, v in enumerate(np.nanmean(np.array(recalls), axis=0))}

        accuracies_dict = {f'accuracies_{i}': v for i, v in enumerate(np.nanmean(np.array(accuracies), axis=0))}
        val_accuracies_dict = {f'val_accuracies_{i}': v for i, v in enumerate(np.nanmean(np.array(val_accuracies), axis=0))}

        pd.DataFrame({'loss':[np.array(losses).mean()], 
                    'val_loss':[mean_val_loss], 
                    **f1_dict,
                    **val_f1_dict,
                    **precisions_dict,
                    **val_precisions_dict,
                    **recalls_dict,
                    **val_recalls_dict,
                    **accuracies_dict, 
                    **val_accuracies_dict
                    }
                    ).to_csv(results_path, mode='a', index=False, header=False)


 