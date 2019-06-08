import numpy as np
import pickle as pkl
import kaggle
import pandas as pd
kaggle.api.authenticate()

with open('Data/Pickles/processed_test_data.pkl', 'rb') as f:
    test_set = pkl.load(f)

# For blending CV training runs together (Run one or the other). Models and scalers come from EQModel_TF2.py training.
# predictions = np.empty((2624, 5))
# for i in range(5):
#     scaled_test_set = scalers[i].transform(test_set.reshape(2624, 14995)).reshape(2624, 14995, 1)
#     predictions[:, i] = models[i].predict(scaled_test_set).reshape(-1)
# output = predictions.mean(axis=1)

# For a single run (Run one or the other). Model and scaler come from EQModel_TF2.py training.
scaled_test_set = scaler.transform(test_set.reshape(2624, 14995))
output = model.predict(scaled_test_set.reshape(2624, 14995, 1)).reshape(-1)

# Output predictions
submission_file = pd.read_csv('Data/Submissions/submission.csv')
submission_file['time_to_failure'] = output
submission_file.to_csv('Data/Submissions/Submission.csv', index=False)

# Submit to kaggle via api
kaggle.api.competition_submit(competition='LANL-Earthquake-Prediction', file_name='Data/Submissions/Submission.csv',
                              message='This submission would have gotten fourth place! Private score of 2.29740')
