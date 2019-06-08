import numpy as np
from Code.Util import *

# Preprocess according to the current scheme
test = pkl_load('Data/Pickles/raw_test_data.pkl', 'rb')

new_len = (test.shape[1]-50)//10
ss = np.empty((test.shape[0], new_len, 50))
for i in range(test.shape[0]):
    for j in range(new_len):
        ss[i, j] = test[i, (j*10):(j*10)+50]

train_rms = np.log(np.sqrt(np.square(ss).sum(axis=1)/50 - np.square(ss.sum(axis=1)/50)))
test_set = np.log(np.sqrt(np.square(ss).sum(axis=-1)/50 - np.square(ss.sum(axis=-1)/50)))
test_set = test_set.reshape(2624, 14995, 1)

pkl_dump(test_set, 'Data/Pickles/processed_test_data.pkl')
