import sys
import argparse
from functools import reduce
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
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
  return pd.DataFrame(OrderedDict({'var pref': pref_list, '% good OHEu': pct_good})).round(0)


def un_ohe(df, scaler):
  df = df.copy()
  cat_cols = [cat_idx for cat_idx in scaler.cat_cols if len(cat_idx) > 1]
  for cat_idx in cat_cols:
    pref, suffs = get_pref(scaler.columns[cat_idx])
    suffs = np.array(suffs)
    df[pref] = suffs[np.argmax(df.iloc[:, cat_idx].as_matrix(), axis=1)]

  cat_arr = np.array(reduce(lambda x,y: x+y, cat_cols))
  return df.drop(labels=scaler.columns[cat_arr], axis=1)


def drop_static(df):
  df = df.copy()
  to_drop = []
  for i in range(df.shape[1]):
    if len(df.iloc[:,i].unique()) == 1:
      to_drop += [i]
  return df.drop(labels=df.columns[to_drop], axis=1)
    

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
  return pref.rstrip('_'), suffs


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
  parser.add_argument('--n_samples', default=10000, type=int, help='number of samples to use for reconstruction quality tests')
  parser.add_argument('--quality', action='store_true', help='run reconstruction quality tests')
  parser.add_argument('--sensitivity', action='store_true', help='run sensitivity demo')
  parser.add_argument('--improvement', action='store_true', help='run score improvement demo')
  parser.add_argument('--likelihood', action='store_true', help='run log likelihood monitoring demo')
  
  args = parser.parse_args(sys.argv[1:])

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
  print(device)

  x, y, scaler = load_data('lendingclub', is_tree=False, scaler_type='standardize') 
  x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1).astype(np.float32)
  
  flow = RealNVP(x.shape[1], device) 
  if device == 'cpu':
    flow.load_state_dict(torch.load(args.model, map_location='cpu'))
  else:
    flow.load_state_dict(torch.load(args.model))
  flow.to(device)
  flow.eval()

  # produce samples
  x_gen = flow.g(flow.prior.sample((args.n_samples,))).detach().cpu().numpy()[:,:-1]
  np.save('samples.npy', x_gen)

  df_x = scaler.as_dataframe(x[:,:-1])
  df_gen = scaler.as_dataframe(x_gen)

  # reconstruction quality ---------------------
  if args.quality:
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

  # build a model
  param = {'max_depth': 4, 'silent': 1, 'objective': 'binary:logistic'}
  num_round = 20
  num_train = 35000
  bst = xgb.train(param, xgb.DMatrix(x[:num_train], label=y[:num_train]), num_round)
  pred_val = bst.predict(xgb.DMatrix(x[num_train:]))
  val_auc = roc_auc_score(y[num_train:], pred_val)
  print('\nAUC score on val set: {:.03f}'.format(val_auc))

  pred_fn = lambda x: bst.predict(xgb.DMatrix(x))
  shap_fn = lambda x: bst.predict(xgb.DMatrix(x), pred_contribs=True)
  inf_fn = lambda x: flow.f(torch.from_numpy(x.astype(np.float32)).to(device))[0].detach().cpu().numpy()
  gen_fn = lambda z: flow.g(torch.from_numpy(z.astype(np.float32)).to(device)).detach().cpu().numpy()
  logp_fn = lambda x: flow.log_prob(torch.from_numpy(x.astype(np.float32)).to(device)).detach().cpu().numpy()

  if args.sensitivity: 
    i = num_train+1
    noise_sd = 0.1
    n_nbhrs = 200
    x_test = x[i][None,:]
    z_test = inf_fn(x_test) 
    z_nbhr = z_test + noise_sd * np.random.randn(n_nbhrs, x.shape[1]).astype(np.float32)
    x_nbhr = gen_fn(z_nbhr) 
    pred_nbhr = pred_fn(x_nbhr)
    pred_test = pred_fn(x_test)
    shap_values = shap_fn(x_test)[0][:-2]

    # PI

    # LIME
    mod = Ridge(alpha=0.1, fit_intercept=True, normalize=True)
    mod.fit(x_nbhr, pred_nbhr.flatten())
    sensitivity = scaler.columns[np.argsort(-np.abs(mod.coef_[:-1]))][:10]

    shap = scaler.columns[np.argsort(-np.abs(shap_values))][:10]
    print('\nSensitivity top 10')
    print(sensitivity)
    print('\nShap top 10')
    print(shap)

  if args.improvement:
    z = inf_fn(x)
    mean0 = np.mean(z[y==0], axis=0)
    mean1 = np.mean(z[y==1], axis=0)
    improve_vec = mean1 - mean0
    
    pred = pred_fn(x)

    lowest_idx = np.argsort(pred)[:100]
    rej_idx = np.random.choice(lowest_idx)
    z_rej = inf_fn(x[rej_idx][None,:])
    improve_vec = mean1 - z_rej.flatten()
   
    alpha = np.linspace(0, 0.2, 10)
    z_path = z_rej + alpha[:,None] * improve_vec[None,:]
    x_path = gen_fn(z_path)
    score_path = pred_fn(x_path)

    norm = lambda x: np.sqrt(np.sum(x**2, axis=1))
    delta = norm(x_path - x_path[0][None,:])/norm(x_path[0][None,:])
    improvement_plan = drop_static(un_ohe(scaler.as_dataframe(x_path[:,:-1]), scaler))
    print(improvement_plan)

    plt.plot(delta, score_path, '.-')
    plt.title('Score improvement plan for sample {:d}'.format(rej_idx))
    plt.ylabel('XGB model score')
    plt.xlabel('$||\Delta x|| / ||x||$')

  if args.likelihood:
    y_pred = pred_fn(x)
    should_approve = y == 1
    pred_err = 1 - y_pred[should_approve]
    logp = logp_fn(x[should_approve])
    plt.plot(-logp, pred_err, '.')

  plt.show()
