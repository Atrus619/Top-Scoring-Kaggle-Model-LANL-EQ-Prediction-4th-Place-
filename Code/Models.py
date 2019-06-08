import tensorflow as tf
import numpy as np
import time
import pandas as pd


class CRNN(tf.keras.Model):
    def __init__(self):
        super(CRNN, self).__init__()
        # Consider LocallyConnected1D
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=50, strides=1, padding='same',
                                            activation=None, kernel_initializer='he_uniform', name='conv1a')
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=100, strides=None, name='pool1')
        self.gru1 = tf.keras.layers.GRU(units=32, name='gru1')
        self.dense1 = tf.keras.layers.Dense(units=16, activation=None, name='dense1')
        self.output1 = tf.keras.layers.Dense(units=1, activation='relu', name='output1')
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.total_epochs = 0
        self.stored_train_loss = []
        self.stored_val_loss = []

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.lrelu(x)
        x = self.pool1(x)
        x = self.gru1(x)
        x = self.dense1(x)
        x = self.lrelu(x)
        return self.output1(x)

    def train_step(self, sample, label):
        with tf.GradientTape() as tape:
            predictions = self.call(sample)
            loss = self.mae(label, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)

    def eval_once(self, sample, label):
        predictions = self.call(sample)
        loss = self.mae(label, predictions)
        self.eval_loss(loss)

    def train(self, num_epochs, train_gen, val_gen, fold=None, restore_best_weights=True):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        self.store_gradients = np.empty((num_epochs, ))
        for epoch in range(num_epochs):
            start_time = time.time()
            self.total_epochs += 1
            self.train_loss.reset_states()
            self.eval_loss.reset_states()
            for samples, labels in train_gen:
                self.train_step(samples, labels)
            train_gen.on_epoch_end()
            for samples, labels in val_gen:
                self.eval_once(samples, labels)
            print('Epoch: {0}, Time: {1:.2f}, Train Loss: {2:.2f}, Test Loss: {3:.2f}'.format(self.total_epochs,
                                                                                              time.time() - start_time,
                                                                                              self.train_loss.result(),
                                                                                              self.eval_loss.result()))
            self.stored_train_loss.append(self.train_loss.result())
            self.stored_val_loss.append(self.eval_loss.result())
            if self.eval_loss.result().numpy() == min(self.stored_val_loss).numpy():
                self.save_weights(filepath='Checkpoints/best_model_fold_{}.hdf5'.format(fold))
        print('Training Complete.')
        if restore_best_weights:
            print('Restoring Best Weights.')
            self.load_weights('Checkpoints/best_model_fold_{}.hdf5'.format(fold))

    def plot_curves(self, smoothed=None):
        import matplotlib.pyplot as plt
        if smoothed:
            train = pd.DataFrame(self.stored_train_loss).rolling(window=smoothed).mean()
            test = pd.DataFrame(self.stored_val_loss).rolling(window=smoothed).mean()
        else:
            train, test = self.stored_train_loss, self.stored_val_loss
        plt.plot(train, color='b', label='Train')
        plt.plot(test, color='r', label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('MAE (s)')
        plt.legend(loc='upper right')
        best_epoch = self.calc_best_epoch()
        plt.title('Training Curve - Best Loss at Epoch ' + str(best_epoch))
        plt.show()

    def calc_best_epoch(self):
        best_loss = min(self.stored_val_loss)
        for i in range(len(self.stored_val_loss)):
            if self.stored_val_loss[i] == best_loss:
                return i+1

    def best_loss(self):
        return min(self.stored_val_loss).numpy()
