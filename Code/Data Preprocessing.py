import kaggle
import numpy as np
import pandas as pd
from Code.Util import *

# Import data for the first time
kaggle.api.authenticate()
kaggle.api.competition_download_files('LANL-Earthquake-Prediction', 'Data')

raw = pd.read_csv('Data/train.csv.zip', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}).values

data = raw[:, 0]
new_len = (len(data)-50)//10
ss = np.empty((new_len, 50))
for i in range(new_len):
    ss[i] = data[(i*10):(i*10)+50]

# Adding in additional central moments could improve accuracy (third, fourth, sixth). Some of these will likely need to be scaled by standard deviation, as well as
# log transformed to be more normally distributed.
train_rms = np.log(np.sqrt(np.square(ss).sum(axis=1)/50 - np.square(ss.sum(axis=1)/50)))
train_rms = np.concatenate((np.array((0, 0, 0, 0, 0)), train_rms)).reshape(-1, 1)

pkl_dump(train_rms, 'Data/Pickles/train_data_trim.pkl')

# Targets
targets = raw[::10, 1].astype('float16').reshape(-1, 1)
pkl_dump(targets, 'Data/Pickles/targets.pkl')

# Create additional helpers
# EQ_ind
targets_offset = np.concatenate((targets[1:], np.array(targets[-1]).reshape(1, 1)), axis=0)
incr = (targets_offset > targets)*1
cum = np.cumsum(incr)
EQ_ind = (cum + 1).reshape(-1, 1)
pkl_dump(EQ_ind, 'Data/Pickles/EQ_ind.pkl')

# CV_ind
cv_mapping = pkl_load('Data/Pickles/cv_mapping.pkl')
df = pd.DataFrame(EQ_ind).merge(pd.DataFrame(cv_mapping), how='inner', left_on=0, right_on=0)
CV_ind = np.array(df[1]).reshape(-1, 1)
pkl_dump(CV_ind, 'Data/Pickles/CV_ind.pkl')

# mod_cv_index and mod_eval_index are simply the indices I chose to use for evaluation. Data spread every 150,000 chunks, ignoring overlaps.
