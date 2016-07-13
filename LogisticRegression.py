# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

from RegressionBase import RegressionBase

class LogisticRegression(RegressionBase):

    def __init__(self, trX, trY, numStep, learning_rate):
        RegressionBase.__init__(self, trX, trY, numStep, learning_rate)

    def loss(self, output, supervisor_labels_placeholder):
        x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(output, supervisor_labels_placeholder, name = 'xentropy')
        return tf.reduce_mean(x_entropy, name = 'xentropy_mean')

if __name__ == '__main__':
    data = pd.read_csv('test_data.csv')
    trX = data[['X1', 'X2', 'X3', 'X4']]
    trY = data['Y']

    numStep = 500000
    learning_rate = 0.005

    r = LogisticRegression(trX, trY, numStep, learning_rate)
    r.run()
