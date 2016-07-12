# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

from RegressionBase import RegressionBase

class LogisticRegression(RegressionBase):

    def __init__(self, trX, trY, numStep, learning_rate):
        RegressionBase.__init__(self, trX, trY, numStep, learning_rate)

    def sigmoid(self, a):
        return 1 / (1 + tf.exp(-a))

    def inference(self, input_placeholder, W, b):
        return self.sigmoid(tf.matmul(input_placeholder, W) + b)

    def loss(self, output, supervisor_labels_placeholder):
        _t = tf.mul(supervisor_labels_placeholder, tf.log(output))
        _f = tf.mul(1 - supervisor_labels_placeholder, tf.log(1 - output))
        return - tf.add(_t, _f)

if __name__ == '__main__':
    data = pd.read_csv('test_data.csv')
    trX = data[['X1', 'X2', 'X3', 'X4']]
    trY = data['Y']

    numStep = 1000
    learning_rate = 0.5

    r = LogisticRegression(trX, trY, numStep, learning_rate)
    r.run()
