from src.config.settings import RESULTS_PATH
import os
import numpy as np
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.utils.metrics import accuracy_per_class
import pandas as pd
import warnings


# Filter out the specific warning for np.nanmean
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)


@torch.no_grad()
def test( model:nn.Module, 
          criterion:nn.Module, 
          test_loader:DataLoader, 
          labels:list, 
          run_name:str) -> None:
    
    results_path = os.path.join(RESULTS_PATH, 'test', run_name, 'result_metrics.csv')
    os.makedirs(os.path.dirname(results_path))

    pd.DataFrame({'loss':[], 
                'avg_accuracies':[],
                'avg_recalls': [],
                'avg_precisions': [],
                'avg_f1s':[], 
                **{f'f1s_{i}': [] for i in range(len(labels))},
                **{f'precisions_{i}': [] for i in range(len(labels))},
                **{f'recalls_{i}': [] for i in range(len(labels))},
                **{f'accuracies_{i}': [] for i in range(len(labels))},
                }).to_csv(results_path, mode='w', index=False)

    losses, accuracies, f1s, recalls, precisions =  [],[],[],[],[]
    avg_accuracies, avg_f1s, avg_recalls, avg_precisions =[],[],[],[]

    '''##### TESTING LOOP#####'''
    for batch in tqdm(test_loader, desc='testing', leave=False):
        features, targets = batch
        out = model.forward(features)
        loss = criterion(out,targets.to(torch.long))
        predicted_class = torch.argmax(out, dim=1)

        precisions.append(precision_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
        f1s.append(f1_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
        recalls.append(recall_score(targets, predicted_class, average=None, labels=labels, zero_division=np.nan))
        accuracies.append(accuracy_per_class(targets, predicted_class, labels=labels))

        avg_precisions.append(precision_score(targets, predicted_class, labels=labels,average='weighted', zero_division=np.nan))
        avg_f1s.append(f1_score(targets, predicted_class, labels=labels,average='weighted', zero_division=np.nan))
        avg_recalls.append(recall_score(targets, predicted_class, labels=labels,average='weighted', zero_division=np.nan))
        avg_accuracies.append(accuracy_score(targets, predicted_class))

        losses.append(loss.item())


    '''### SAVE METRICS ###'''

    f1_dict = {f'f1s_{i}': v for i, v in enumerate(np.nanmean(np.array(f1s), axis=0))}

    precisions_dict = {f'precisions_{i}': v for i, v in enumerate(np.nanmean(np.array(precisions), axis=0))}

    recalls_dict = {f'recalls_{i}': v for i, v in enumerate(np.nanmean(np.array(recalls), axis=0))}

    accuracies_dict = {f'accuracies_{i}': v for i, v in enumerate(np.nanmean(np.array(accuracies), axis=0))}
    
    pd.DataFrame({'loss':[np.array(losses).mean()], 
                'avg_accuracies':[np.array(avg_accuracies).mean()],
                'avg_recalls': [np.array(avg_recalls).mean()],
                'avg_precisions': [np.array(avg_precisions).mean()],
                'avg_f1s':[np.array(avg_f1s).mean()], 
                **f1_dict,
                **precisions_dict,
                **recalls_dict,
                **accuracies_dict, 
                }
                ).to_csv(results_path, mode='a', index=False, header=False)

