import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import distributions
from torch.nn.parameter import Parameter

from .data import get_data


def train(param, x, y):

  dataloader = DataLoader(
    torch.from_numpy(x.astype(np.float32)),
    batch_size=param.batch_size,
    shuffle=True,
    num_workers=2
  )

  nets = lambda: nn.Sequential(
    nn.Linear(2, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, 2), 
    nn.Tanh())

  nett = lambda: nn.Sequential(
    nn.Linear(2, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, 256), 
    nn.LeakyReLU(), 
    nn.Linear(256, 2))

  masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
  prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

  flow = RealNVP(nets, nett, masks, prior)
  flow.to(device)
  flow.train()

  optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=param.lr)

  it = 0 
  while it < param.total_it:

    for i, data in enumerate(dataloader):
      loss = -flow.log_prob(data).mean()
    
      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()
    
      if t % 500 == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss)


class RealNVP(nn.Module):
  def __init__(self, nets, nett, mask, prior):
    super(RealNVP, self).__init__()
    
    self.prior = prior
    self.mask = nn.Parameter(mask, requires_grad=False)
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
  
  def log_prob(self,x):
    z, logp = self.f(x)
    return self.prior.log_prob(z) + logp
    
  def sample(self, batchSize): 
    z = self.prior.sample((batchSize, 1))
    logp = self.prior.log_prob(z)
    x = self.g(z)
    return x


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_arugment('--dataset', default='lendingclub', help='dataset to use')
  parser.add_argument('--batch_size', default=64, help='batch size')
  parser.add_argument('--total_it', default=10000, help='number of training samples')
  parser.add_argument('--lr', default=1e-4, help='learning rate')

  args = parser.parse_args(sys.argv[1:])
  dataloader = get_data(param.dataset)
  train(args, dataloader)
