import pandas as pd
import pickle as pkl
import numpy as np

# Import data and massage
evaluation_indices = pd.read_csv('Data/Validation Indices Original.csv', delimiter=',', header=None).values.astype('int64')
eval_index, cv_index = np.hsplit(evaluation_indices, 2)
train = pd.read_csv('Data/NewFeatures.csv', delimiter=',', header=None).values.astype('float32')
train_data, other_info = np.hsplit(train, 2)
targets, OG_row, EQ_ind, CV_ind = np.hsplit(other_info, 4)
targets = targets.astype('float16')
OG_row = OG_row.astype('int64')
EQ_ind = EQ_ind.astype('int64')
CV_ind = CV_ind.astype('int64')
mod_eval = pd.read_csv('Data/Validation Indices Modified.csv', delimiter=',', header=None).values.astype('int64')
mod_eval_index, mod_cv_index, _, _ = np.hsplit(mod_eval, 4)

logtrain = pd.read_csv('Data/NewFeatures_logtransformed.csv', delimiter=',', header=0).values.astype('float32')

log_std, log_skew, log_kurt, log_sixth, _, _, _ = np.hsplit(logtrain, 7)
train_data_logs = np.concatenate((log_std, log_skew, log_kurt, log_sixth), axis=1)

del logtrain, log_std, log_skew, log_kurt, log_sixth, other_info

if np.max(mod_eval_index) > len(train_data_logs):  # Prevents from dividing twice on accident when re-running code
    mod_eval_index = mod_eval_index // 10

with open('Data/Pickles/train_data.pkl', 'wb') as f:
    pkl.dump(train_data_logs, f)
with open('Data/Pickles/targets.pkl', 'wb') as f:
    pkl.dump(targets, f)
with open('Data/Pickles/CV_ind.pkl', 'wb') as f:
    pkl.dump(CV_ind, f)
with open('Data/Pickles/EQ_ind.pkl', 'wb') as f:
    pkl.dump(EQ_ind, f)
with open('Data/Pickles/mod_cv_index.pkl', 'wb') as f:
    pkl.dump(mod_cv_index, f)
with open('Data/Pickles/mod_eval_index.pkl', 'wb') as f:
    pkl.dump(mod_eval_index, f)
