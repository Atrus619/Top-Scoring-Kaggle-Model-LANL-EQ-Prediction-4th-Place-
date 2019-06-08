import os
import shutil
import pickle as pkl


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def del_dir(name):
    if os.path.isdir('./Saved Models/{}'.format(name)):
        shutil.rmtree('./Saved Models/{}'.format(name))
    if os.path.isdir('./Error Plots/{}'.format(name)):
        shutil.rmtree('./Error Plots/{}'.format(name))
    if os.path.isdir('./Train and Test Losses/{}'.format(name)):
        shutil.rmtree('./Train and Test Losses/{}'.format(name))


def pkl_dump(obj, filepath):
    with open(filepath, 'wb') as f:
        pkl.dump(obj, f)


def pkl_load(filepath):
    with open(filepath, 'rb') as f:
        return pkl.load(f)
