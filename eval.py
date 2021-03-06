import sys
import argparse
from functools import reduce
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge, LinearRegression
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


def fix_df(x, scaler, return_numpy=False):
  x = scaler.inverse_transform(x.copy())
  for cat_idx in scaler.cat_cols:
    if len(cat_idx) == 1:
      x[:, cat_idx] = (x[:,cat_idx] > 0.5).astype(np.float32)
    else:
      new_ohe = np.zeros((x.shape[0], len(cat_idx)), dtype=np.float32)
      new_ohe[np.arange(x.shape[0]), np.argmax(x[:, cat_idx], axis=1)] = 1.0
      x[:, cat_idx] = new_ohe

  # delinq_2yrs, inq, mths, mths, open
  for i in [5, 6, 7, 8, 9, 10, 12, 16]:
    x[x[:,i] < 0, i] = 0.0
    x[:, i] = np.round(x[:, i])

  if return_numpy:
    return x
  else: 
    return pd.DataFrame(x, columns=scaler.columns)


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
  
  term = np.array([36 if t36 >= t60 else 60 for t60, t36 in zip(df['term_60months'], df['term_36months'])])
  calc = payment(df['loan_amnt'], df['int_rate'], term)

  df_payment = pd.DataFrame({'Synth installment': df['installment'], 'Calc installment': calc})
  df_payment.to_csv('installment.csv', index=False)

  error = 100* (calc - df['installment'])/calc
  fig = plt.figure(2)
  error.plot.hist(ax=fig.gca(), title='% error in payment calculation', range=[-100, 100], bins=50)
  plt.xlabel('%')


def quality_test(df_x, df_gen, scaler):
  # check means vs. sd
  df_mean, df_sd = mean_sd(df_x, df_gen)
  print(df_mean)
  df_mean.to_csv('mean.csv')
  print(df_sd)
  df_sd.to_csv('std.csv')

  categorical_hist(df_x, df_gen, scaler)
  payment_error(df_gen)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='flow_model.pytorch', help='training RealNVP model')
  parser.add_argument('--n_samples', default=10000, type=int, help='number of samples to use for reconstruction quality tests')
  parser.add_argument('--quality', action='store_true', help='run reconstruction quality tests')
  parser.add_argument('--sensitivity', action='store_true', help='run sensitivity demo')
  parser.add_argument('--improvement', action='store_true', help='run score improvement demo')
  
  args = parser.parse_args(sys.argv[1:])

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
  print(device)

  x, y, scaler = load_data('lendingclub', is_tree=False, scaler_type='standardize') 
  x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1).astype(np.float32)
  
  flow = RealNVP(x.shape[1], device) 
  if device.type == 'cpu':
    flow.load_state_dict(torch.load(args.model, map_location='cpu'))
  else:
    flow.load_state_dict(torch.load(args.model))
  flow.to(device)
  flow.eval()

  # produce samples
  x_gen = flow.g(flow.prior.sample((args.n_samples,))).detach().cpu().numpy()[:,:-1]
  np.save('samples.npy', x_gen)

  df_x = scaler.as_dataframe(x[:,:-1])
  df_gen = fix_df(x_gen, scaler) 

  df_gen.to_csv('real.csv')
  # reconstruction quality ---------------------
  if args.quality:
    quality_test(df_x, df_gen, scaler)

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
    noise_sd = 0.1
    n_nbhrs = 40

    i = np.random.randint(num_train, x.shape[0], 1)[0]

    print('\nSensitivity for sample {:d}'.format(i))
    x_test = x[i][None,:]
    z_test = inf_fn(x_test) 
    z_nbhr = z_test + noise_sd * np.random.randn(n_nbhrs, z_test.shape[1]).astype(np.float32)
    x_nbhr = gen_fn(z_nbhr)
    
    def fixer(x, scaler):
      """Make fixed np array that is standardized"""
      x_new = fix_df(x[:,:-1], scaler, return_numpy=True)
      x_new = scaler.transform(x_new)
      return np.concatenate([x_new, np.zeros((x_new.shape[0],1), dtype=np.float32)], axis=1)

    x_nbhr = fixer(x_nbhr, scaler)
      
    pred_nbhr = pred_fn(x_nbhr)
    pred_test = pred_fn(x_test)
    shap_values = shap_fn(x_test)[0][:-2]

    best_idx_shap = np.argsort(-np.abs(shap_values))[:10]
     
    the_crew_np = np.concatenate([x_test, x_nbhr], axis=0)[:, :-1] 
    the_crew = drop_static(un_ohe(scaler.as_dataframe(the_crew_np), scaler))
    
    col_list = list(scaler.columns)
    vals = []
    for col in the_crew.columns:
      i_col = col_list.index(col) 
      
      mod = LinearRegression(fit_intercept=True, normalize=False)
      mod.fit(the_crew_np[:, i_col][:,None], np.append(pred_test, pred_nbhr))
      vals += [mod.coef_[0]]
    
    vals = np.array(vals)
    best_idx = np.argsort(-np.abs(vals))
    to_show = np.min([len(best_idx), 5])
    best_idx = best_idx[:to_show]
    
    cols_to_use = the_crew.columns[best_idx]
    df_sens = pd.DataFrame(OrderedDict({
      'Sensitivity': vals[best_idx],
      'Feature': df_x.loc[i, cols_to_use]}), index=cols_to_use).round({'Sensitivity': 3, 'Feature': 2})

    best_idx_shap = best_idx_shap[:to_show]
    cols_to_use = scaler.columns[best_idx_shap]
    df_shap = pd.DataFrame(OrderedDict({
      'Shapley': shap_values[best_idx_shap],
      'Feature': df_x.loc[i, cols_to_use]}), index=cols_to_use).round({'Shapley': 3, 'Feature': 2})
    
    print('Score for sample: {:.03f}'.format(pred_test[0]))
    df_sens.to_csv('top_sens.csv')
    df_shap.to_csv('top_shap.csv')
    print(df_sens)
    print(df_shap)


  if args.improvement:
    z = inf_fn(x)
    mean1 = np.mean(z[y==1], axis=0)
    
    pred = pred_fn(x)

    lowest_idx = np.argsort(pred)[:100]
    rej_idx = np.random.choice(lowest_idx)
    z_rej = inf_fn(x[rej_idx][None,:])
    improve_vec = mean1 - z_rej.flatten()

    # to mean
    alpha = np.linspace(0, 0.2, 10)
    z_path = z_rej + alpha[:,None] * improve_vec[None,:]
    x_path = gen_fn(z_path)
    x_path = fix_df(x_path[:,:-1], scaler, return_numpy=True)
    x_path = np.concatenate([scaler.transform(x_path), np.zeros((x_path.shape[0], 1), dtype=np.float32)], axis=1)
    score_path = pred_fn(x_path)

    norm = lambda x: np.sqrt(np.sum(x**2, axis=1))
    delta = norm(x_path - x_path[0][None,:])/norm(x_path[0][None,:])
    improvement_plan = drop_static(un_ohe(scaler.as_dataframe(x_path[:,:-1]), scaler))
    improvement_plan.to_csv('improvement_plan.csv', index=False)
    print(improvement_plan)

    plt.plot(delta, score_path, '.-')
    plt.title('Score improvement plan for sample {:d}'.format(rej_idx))
    plt.ylabel('XGB model score')
    plt.xlabel('$||\Delta x|| / ||x||$')

  plt.show()
