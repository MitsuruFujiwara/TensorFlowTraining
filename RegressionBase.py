# -*- coding: utf-8 -*-

# Base class for regression analysis with tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf

class RegressionBase(object):

    def __init__(self, trX, trY, numStep):
        self.trX = trX
        self.trY = trY
        self.numStep = numStep
        self.M = trX.shape[1]

    def inference(self, input_placeholder):
        M = self.M
        W = tf.Variable(tf.zeros([M, M]))
        b = tf.Variable(tf.zeros([M]))
        return tf.nn.softmax(tf.matmul(input_placeholder, W) + b)

    def loss(self, output, supervisor_labels_placeholder):
        return -tf.reduce_sum(supervisor_labels_placeholder * tf.log(output))

    def training(self, loss):
        return tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    def run(self):
        M = self.M
        supervisor_labels_placeholder = tf.placeholder("float", [None, M])
        input_placeholder = tf.placeholder("float", [M, None])
        feed_dict = {input_placeholder: self.trX, supervisor_labels_placeholder: self.trY}

        with tf.Session() as sess:
            output = self.inference(input_placeholder)
            loss = self.loss(output, supervisor_labels_placeholder)
            training_op = self.training(loss)

            init = tf.initialize_all_variables()
            sess.run(init)

            for step in range(self.numStep):
                sess.run(training_op, feed_dict = feed_dict)
                if step % 100 == 0:
                    print sess.run(loss, feed_dict = feed_dict)

if __name__ == '__main__':
    data = pd.read_csv('test_data.csv')
    trX = data[['X1', 'X2', 'X3', 'X4']]
    trY = data['Y']
    print trX.shape

    """
    trX = [
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
    ]

    trY = [
    [0., 1., 0.],
    [0., 0., 1.],
    [1., 0., 0.]
    ]
    """

    numStep = 10000

    r = RegressionBase(trX, trY, numStep)
    r.run()
