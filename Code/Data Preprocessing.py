import kaggle
import numpy as np
import pandas as pd
import pickle as pkl

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

with open('Data/Pickles/train_data_trim.pkl', 'wb') as f:
    pkl.dump(train_rms, f)
