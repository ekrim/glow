import sys
import argparse
from functools import reduce
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from zamlexplain.data import load_data

from model import RealNVP


def mean_sd(df_x, df_gen):
  return pd.DataFrame(OrderedDict({
    'data mean': df_x.mean(), 
    'synth mean': df_gen.mean(), 
    'data sd': df_x.std(), 
    'synth sd': df_gen.std()}))


def negative_check(df):
  return (df < 0).sum()


def categorical_check(df, cat_cols):
  bad_idx = []
  for cat_idx in cat_cols:
    bad_idx += [df.iloc[:, cat_idx].sum(axis=1) != 1.0]
  return bad_idx


def get_pref(lst):
  cnt = 0
  while all([lst[0][cnt] == el[cnt] for el in lst]):
    cnt += 1

  pref = lst[0][:cnt]
  suffs = [el[cnt:] for el in lst]
  return pref, suffs


def categorical_hist(df_x, df_gen, scaler):
  fig = plt.figure(1, figsize=(10, 10))
  cnt = 0
  for cat_idx in scaler.cat_cols:
    n_var = len(cat_idx)
    if n_var > 1:
      x_vals = np.argmax(df_x.iloc[:, cat_idx].as_matrix(), axis=1)
      gen_vals = np.argmax(df_gen.iloc[:, cat_idx].as_matrix(), axis=1)

      x_hist = np.histogram(x_vals, np.arange(n_var+1))[0]
      x_hist = x_hist/np.sum(x_hist)

      gen_hist = np.histogram(gen_vals, np.arange(n_var+1))[0]
      gen_hist = gen_hist/np.sum(gen_hist)

      pref, suffs = get_pref(scaler.columns[cat_idx])
      ax = plt.subplot(2, 2, cnt+1)
      cnt += 1
      plt.bar(np.arange(n_var)-0.1, x_hist, width=0.15, color='b', align='center')
      plt.bar(np.arange(n_var)+0.1, gen_hist, width=0.15, color='r', align='center')
      plt.xticks(np.arange(n_var), suffs, rotation=40, ha='right')
      plt.xlabel('feature')
      plt.title(pref)

  handles, labels = ax.get_legend_handles_labels()
  fig.legend(handles, ['real', 'synth'], loc='upper center')
  plt.show()

def payment_error(df):
  def payment(p, r, n):
    r /= 12
    return p*(r*(1+r)**n)/((1+r)**n - 1) 
  
  term = np.array([36 if t36 > t60 else 60 for t36, t60 in zip(df['term_60months'], df['term_36months'])])
  calc = payment(df['loan_amnt'], df['int_rate'], term)

  return pd.DataFrame({
    'synth installment': df['installment'], 
    'calc installment': calc}).round(2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='flow_model.pytorch', help='training RealNVP model')
  
  args = parser.parse_args(sys.argv[1:])

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

  x, y, scaler = load_data('lendingclub', is_tree=False, scaler_type='standardize') 
  x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1).astype(np.float32)
  
  flow = RealNVP(x.shape[1], device) 
  flow.load_state_dict(torch.load(args.model))
  flow.to(device)
  flow.eval()
  print(flow.mask)

  # produce samples
  x_gen = flow.g(flow.prior.sample((10000,))).detach().cpu().numpy()[:,:-1]
  np.save('samples.npy', x_gen)

  df_x = scaler.as_dataframe(x[:,:-1])
  df_gen = scaler.as_dataframe(x_gen)

  print(df_x.head(10))
  print(df_gen.head(10).iloc[:,:20])

  # check means vs. sd
  print(mean_sd(df_x, df_gen))

  # check negative values
  print(negative_check(df_gen))
  
  # check categorical
  bad_idx = categorical_check(df_gen, scaler.cat_cols)
  for idx, cat_idx in zip(bad_idx, scaler.cat_cols):
    print(scaler.columns[cat_idx])
    print(df_gen.loc[idx].iloc[:, cat_idx].head(50).as_matrix().astype(int))

  # check categorical hist
  categorical_hist(df_x, df_gen, scaler)

  # check payment calculation error
  print(payment_error(df_gen))
