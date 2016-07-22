# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

from RegressionBase import RegressionBase

class LogisticRegression(RegressionBase):

    def __init__(self, trX, trY, numStep, numParameter, learning_rate):
        RegressionBase.__init__(self, trX, trY, numStep, numParameter, learning_rate)

    def training(self, loss):
        return tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

    def loss(self, output, supervisor_labels_placeholder):
        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(output, supervisor_labels_placeholder, name = 'xentropy')
        return tf.reduce_mean(x_entropy, name = 'xentropy_mean')

if __name__ == '__main__':
    data = pd.read_csv('test_data.csv')
    trX = data[[\
    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9','X10', 'X11', 'X12',\
    'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21','X22', 'X23',\
    'X24', 'X25', 'X26', 'X27'\
    ]].fillna(0)
    trY = data['Y']

    numStep = 10000
    numParameter = len(trX.columns)
    learning_rate = 0.5

    r = LogisticRegression(trX, trY, numStep, numParameter, learning_rate)
    r.run()

    # loss = 0.282286
    # b = -6.56112
    # W0 = 0.226928
    # W1 = 0.238033
    # W2 = -0.0118023
    # W3 = 0.499244
