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


PRINT_FREQ = 1000   # print loss every __ samples seen


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

  nets = lambda: nn.Sequential(
    nn.Linear(dim_in, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, dim_in), 
    nn.Tanh())

  nett = lambda: nn.Sequential(
    nn.Linear(dim_in, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, dim_in))

  masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
  prior = distributions.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))

  flow = RealNVP(nets, nett, masks, prior, device)
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


class RealNVP(nn.Module):
  def __init__(self, nets, nett, masks, prior, device):
    super(RealNVP, self).__init__()
    
    self.device = device
    self.prior = prior
    self.mask = nn.Parameter(masks, requires_grad=False)
    self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
    self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
    
  def g(self, z):
    x = z
    for i in range(len(self.t)):
      x_ = x*self.mask[i]
      s = self.s[i](x_)*(1 - self.mask[i])
      t = self.t[i](x_)*(1 - self.mask[i])
      x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
    return x

  def f(self, x):
    log_det_J, z = x.new_zeros(x.shape[0]), x
    for i in reversed(range(len(self.t))):
      z_ = self.mask[i] * z
      s = self.s[i](z_) * (1-self.mask[i])
      t = self.t[i](z_) * (1-self.mask[i])
      z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
      log_det_J -= s.sum(dim=1)
    return z, log_det_J
  
  def log_prob(self, x):
    z, logp = self.f(x)
    return self.prior.log_prob(z) + logp
    
  def sample(self, batchSize): 
    z = self.prior.sample((batchSize, 1))
    logp = self.prior.log_prob(z)
    x = self.g(z)
    return x


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='lendingclub', help='dataset to use')
  parser.add_argument('--batch_size', default=64, help='batch size')
  parser.add_argument('--total_it', default=10000, help='number of training samples')
  parser.add_argument('--lr', default=1e-4, help='learning rate')

  args = parser.parse_args(sys.argv[1:])

  if args.dataset == 'lendingclub':
    x, y, scaler = load_data('lendingclub', is_tree=False, scaler_type='standardize')
      
  else:
    x, y = datasets.make_moons(n_samples=30000, noise=0.05)
    x = x.astype(np.float32)

  train(args, x, y)
