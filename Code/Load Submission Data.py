import glob
import pandas as pd
import numpy as np
from Code.Util import *

# Aggregate all test files for later retrieval
test_paths = glob.glob('Data/Test Data/*.csv')
x = ''
for y in test_paths:
    x += ', ' + y
segs = pd.DataFrame(re.findall(r'seg.{7}', x)).rename(columns={0: 'seg_id'})
segs.to_csv('Data/Submissions/submission.csv', index=False)
test_size = len(test_paths)
test = np.empty((test_size, 150000))
for i, x in enumerate(test_paths):
    test[i] = np.genfromtxt(x, skip_header=1)

pkl_dump(test, 'Data/Pickles/raw_test_data.pkl')
