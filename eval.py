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

  df_x = df_x.iloc[:,:17]
  df_gen = df_gen.iloc[:,:17]

  mean_x = df_x.mean()
  mean_gen = df_gen.mean()
  mean_err = 100*(mean_gen - mean_x)/mean_x

  df_mean = pd.DataFrame(OrderedDict({
    'data mean': mean_x, 
    'synth mean': mean_gen,
    'err %': mean_err})).round({'data mean': 2, 'synth mean': 2, 'err %': 0})

  std_x = df_x.std()
  std_gen = df_gen.std()
  std_err = 100*(std_gen - std_x)/std_x

  df_std = pd.DataFrame(OrderedDict({
    'data std': std_x, 
    'synth std': std_gen,
    'err %': std_err})).round({'data std': 2, 'synth std': 2, 'err %': 0})

  return df_mean, df_std


def negative_check(df):
  return (df < 0).sum()


def categorical_check(df, scaler):
  good_ohe = []
  pref_list = []
  for cat_idx in scaler.cat_cols:
    good_ohe += [((np.abs(df.iloc[:, cat_idx].round(0) - 0) > 0.0001).sum(axis=1) == len(cat_idx)-1).sum()]

    pref_list += [get_pref(scaler.columns[cat_idx])[0].rstrip('_')]

  pct_good = 100*(df.shape[0] - np.array(good_ohe))/df.shape[0]
  return pd.DataFrame(OrderedDict({'var pref': pref_list, '% good ohe': pct_good})).round(0)


def get_pref(lst):
  if len(lst) == 1:
    pref = lst[0]
    suffs = ['']
  else:
    cnt = 0
    while all([lst[0][cnt] == el[cnt] for el in lst]):
      cnt += 1

    pref = lst[0][:cnt]
    suffs = [el[cnt:] for el in lst]
  return pref, suffs


def categorical_hist(df_x, df_gen, scaler):
  fig = plt.figure(1, figsize=(8, 8))
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
      plt.subplot(2, 2, cnt+1)
      cnt += 1
      obj1 = plt.bar(np.arange(n_var)-0.1, x_hist, width=0.15, color='b', align='center')
      obj2 = plt.bar(np.arange(n_var)+0.1, gen_hist, width=0.15, color='r', align='center')
      plt.xticks(np.arange(n_var), suffs, rotation=30, ha='right')
      plt.title(pref.rstrip('_'))

  plt.subplots_adjust(hspace=0.4)
  fig.legend([obj1, obj2], ['real', 'synth'], loc='upper center')


def payment_error(df):
  def payment(p, r, n):
    r /= 12
    return p*(r*(1+r)**n)/((1+r)**n - 1) 
  
  term = np.array([36 if t36 >= t60 else 60 for t36, t60 in zip(df['term_60months'], df['term_36months'])])
  calc = payment(df['loan_amnt'], df['int_rate'], term)

  error = 100* (calc - df['installment'])/calc
  fig = plt.figure(2)
  error.plot.hist(ax=fig.gca(), title='% error in payment calculation', range=[-100, 100], bins=50)
  plt.xlabel('%')


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

  print(df_x.head(20))
  print(df_gen.head(20).iloc[:,:17])
  print(df_gen.head(20).iloc[:, 17:])

  # check means vs. sd
  df_mean, df_sd = mean_sd(df_x, df_gen)
  print(df_mean)
  print(df_sd)

  # check negative values
  print(negative_check(df_gen))
  
  # check categorical
  print(categorical_check(df_gen, scaler))

  # check categorical hist
  categorical_hist(df_x, df_gen, scaler)

  # check payment calculation error
  payment_error(df_gen)

  plt.show()
