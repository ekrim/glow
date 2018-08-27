import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import distributions
from torch.nn.parameter import Parameter
from sklearn import datasets

from zamlexplain.data import load_data

from model import RealNVP


PRINT_FREQ = 2000   # print loss every __ samples seen


def train(param, x, y):

  dim_in = x.shape[1]
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  dataloader = DataLoader(
    torch.from_numpy(x.astype(np.float32)),
    batch_size=param.batch_size,
    shuffle=True,
    num_workers=2
  )

  flow = RealNVP(dim_in, device)
  flow.to(device)
  flow.train()

  optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=param.lr)

  it, print_cnt = 0, 0 
  while it < param.total_it:

    for i, data in enumerate(dataloader):
     
      loss = -flow.log_prob(data.to(device)).mean()
    
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()
      
      it += data.shape[0]
      print_cnt += data.shape[0]
      if print_cnt > PRINT_FREQ:
        print('it {:d} -- loss {:.03f}'.format(it, loss))
        print_cnt = 0

    torch.save(flow.state_dict(), 'flow_model.pytorch')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='lendingclub', help='dataset to use')
  parser.add_argument('--batch_size', default=64, type=int, help='batch size')
  parser.add_argument('--total_it', default=10000, type=int, help='number of training samples')
  parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

  args = parser.parse_args(sys.argv[1:])

  if args.dataset == 'lendingclub':
    x, y, scaler = load_data('lendingclub', is_tree=False, scaler_type='standardize')
    x = np.concatenate([x, np.zeros(x.shape[0])[:,None]], axis=1)
      
  else:
    x, y = datasets.make_moons(n_samples=30000, noise=0.05)
  x = x.astype(np.float32)

  train(args, x, y)
