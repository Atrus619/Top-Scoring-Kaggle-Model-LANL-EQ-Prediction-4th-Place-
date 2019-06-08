import tensorflow as tf
import numpy as np
import random


class ValDataGenerator(tf.keras.utils.Sequence):
    """Generates data"""
    def __init__(self, data, targets, eval_index, cv_index, cv, batch_size, lookback):
        self.data = data
        self.data_width = self.data.shape[1]
        self.targets = targets
        self.eval_index = eval_index
        self.cv_index = cv_index
        self.cv = cv
        self.batch_size = batch_size
        self.lookback = lookback
        self.row_master = self.eval_index[self.cv_index == self.cv]

    def __len__(self):
        """Denotes number of batches per epoch. Cuts off after max_index is reached."""
        return len(self.eval_index[self.cv_index == self.cv])//self.batch_size + 1

    def __getitem__(self, index):
        """
        Returns a batch
        rows marks the ending index of each sample within data for a batch
        """
        rows = self.row_master[index * self.batch_size:(index + 1) * self.batch_size]
        samples, label = self.__data_generation(rows)
        return samples, label

    def __data_generation(self, rows):
        """Generates one batch of data samples and targets"""
        samples = np.empty((len(rows), self.lookback, self.data_width)).astype('float32')
        label = np.empty((len(rows), 1)).astype('float32')
        for j in range(len(rows)):
            samples[j, ] = self.data[(rows[j] - self.lookback):rows[j]]
            label[j] = self.targets[rows[j]]
        return samples, label


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data"""
    def __init__(self, data, targets, indices, batch_size, min_index=0, max_index=None,
                 lookback=149950, offset=150000, shuffle_start=True, shuffle_feed=True):
        if max_index is None:
            self.max_index = len(data)
        else:
            self.max_index = max_index
        self.data = data[min_index:self.max_index].astype('float32')
        self.data_width = self.data.shape[1]
        self.targets = targets[min_index:self.max_index].astype('float32')
        self.indices = indices[min_index:self.max_index]
        self.batch_size = batch_size
        self.lookback = lookback
        self.offset = offset
        self.shuffle_start = shuffle_start
        self.shuffle_feed = shuffle_feed
        self.epoch_start = self.lookback+5
        self.pre_len = (self.max_index - min_index + self.offset - self.lookback) // (self.batch_size * self.offset)
        self.row_master = list(range(self.epoch_start, self.epoch_start + self.pre_len * self.batch_size * self.offset, self.offset))  # indices in data of all samples
        self.on_epoch_end()

    def __len__(self):
        """Denotes number of batches per epoch. Cuts off after max_index is reached."""
        return len(self.row_master) // self.batch_size + 1

    def __getitem__(self, index):
        """
        Returns a batch
        rows marks the ending index of each sample within data for a batch
        """
        rows = self.row_master[index * self.batch_size:(index + 1) * self.batch_size]
        samples, labels = self.__data_generation(rows)
        return samples, labels

    def on_epoch_end(self):
        """If shuffle is true, then we start from a new initial index"""
        self.epoch_start = self.lookback+5
        if self.shuffle_start:
            self.epoch_start += random.randint(0, self.offset)
            self.row_master = list(range(self.epoch_start, self.epoch_start + self.pre_len * self.batch_size * self.offset, self.offset))
        # if self.perform_os is not None:
        #     self.over_sample()
        self.adjust_cross_eqs()
        if self.shuffle_feed:
            np.random.shuffle(self.row_master)

    def adjust_cross_eqs(self):
        """Deletes samples that have an earthquake occur during them to occur later, so that an EQ does not occur within it."""
        del_list = []
        for i, row in enumerate(self.row_master):
            if self.indices[row] != self.indices[row - self.lookback + 1]:
                del_list.append(i)
        self.row_master = np.delete(self.row_master, del_list)

    def __data_generation(self, rows):
        """Generates one batch of data samples and targets"""
        samples = np.empty((len(rows), self.lookback, self.data_width)).astype('float32')
        labels = np.empty((len(rows), 1)).astype('float32')
        for j in range(len(rows)):
            samples[j, ] = self.data[(rows[j] - self.lookback):rows[j]]
            labels[j] = self.targets[rows[j]]
        return samples, labels


class One_Sample_Only_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, targets, lookback):
        self.data = data
        self.targets = targets
        self.lookback = lookback
        self.epoch_start = self.lookback+5
        self.data_width = self.data.shape[1]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        samples = self.data[self.epoch_start - self.lookback: self.epoch_start].reshape(1, self.lookback, self.data_width)
        labels = self.targets[self.epoch_start]
        return samples, labels



