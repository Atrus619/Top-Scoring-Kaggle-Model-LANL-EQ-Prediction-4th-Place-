import numpy as np
import pickle as pkl

# Preprocess according to the current scheme
with open('Data/Pickles/raw_test_data.pkl', 'rb') as f:
    test = pkl.load(f)

new_len = (test.shape[1]-50)//10
ss = np.empty((test.shape[0], new_len, 50))
for i in range(test.shape[0]):
    for j in range(new_len):
        ss[i, j] = test[i, (j*10):(j*10)+50]

train_rms = np.log(np.sqrt(np.square(ss).sum(axis=1)/50 - np.square(ss.sum(axis=1)/50)))
test_set = np.log(np.sqrt(np.square(ss).sum(axis=-1)/50 - np.square(ss.sum(axis=-1)/50)))
test_set = test_set.reshape(2624, 14995, 1)

with open('Data/Pickles/processed_test_data.pkl', 'wb') as f:
    pkl.dump(test_set, f)
