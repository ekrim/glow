import torch
from torch.utils.data import Dataset

from zamlexplain.data import load_data


def get_data(dataset):
  if dataset == 'lendingclub':
    x, y, scaler = load_data(dataset, is_tree=False, scaler='standardize')
    dataset
    
  else:
    raise ValueError('no such dataset')

