from sklearn import preprocessing
from Code.Generators import *
import pickle as pkl
from Code.Models import CRNN

# Import data
# Older data with additional moments
# with open('Data/Pickles/train_data.pkl', 'rb') as f:
#     train_data_logs = pkl.load(f)
with open('Data/Pickles/targets.pkl', 'rb') as f:
    targets = pkl.load(f)
with open('Data/Pickles/CV_ind.pkl', 'rb') as f:
    CV_ind = pkl.load(f)
with open('Data/Pickles/EQ_ind.pkl', 'rb') as f:
    EQ_ind = pkl.load(f)
with open('Data/Pickles/mod_cv_index.pkl', 'rb') as f:
    mod_cv_index = pkl.load(f)
with open('Data/Pickles/mod_eval_index.pkl', 'rb') as f:
    mod_eval_index = pkl.load(f)
with open('Data/Pickles/train_data_trim.pkl', 'rb') as f:
    train_data_trim = pkl.load(f)

# For CV, comment out if doing a single run
# models = []
# scalers = []
# for i in range(1, 6):
#     print("Fold {}:".format(i))
#     fold = i
    fold = 1  # Can set to anything if not doing CV, only affects the validation generator
    eq_include = [3, 8, 1, 5, 12, 14, 10, 2, 15, 11]
    # eq_include = list(range(1, 18))
    mask = np.in1d(EQ_ind, eq_include)
    cv_mask = (CV_ind != 0).reshape(-1)
    cv_train = train_data_trim[mask & cv_mask]
    cv_targets = targets[mask & cv_mask]
    cv_eqs = EQ_ind[mask & cv_mask]

    scaler = preprocessing.StandardScaler()
    cv_train = scaler.fit_transform(cv_train)
    cv_val = scaler.transform(train_data_trim)

    batch_size = 64
    lookback = 14995
    offset = 15000

    train_gen = DataGenerator(data=cv_train,
                              targets=cv_targets,
                              indices=cv_eqs,
                              min_index=0,
                              max_index=None,
                              batch_size=batch_size,
                              lookback=lookback,
                              offset=offset,
                              shuffle_start=True,
                              shuffle_feed=True)

    val_gen = ValDataGenerator(data=cv_val,
                               targets=targets,
                               eval_index=mod_eval_index,
                               cv_index=mod_cv_index,
                               cv=fold,
                               batch_size=batch_size,
                               lookback=lookback)

    tf.keras.backend.clear_session()
    model = CRNN()
    model.train(num_epochs=20, train_gen=train_gen, val_gen=val_gen, fold=fold, restore_best_weights=False)
    # For CV - comment out if doing a single run
    # models.append(model)
    # scalers.append(scaler)

    model.plot_curves(smoothed=None)
