# -*- coding: utf-8 -*-

# Base class for regression analysis with tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf

class RegressionBase(object):

    def __init__(self, trX, trY, numStep, numParameter, learning_rate):
        self.N = len(trY)
        self.trX = trX
        self.trY = trY.reshape(self.N, 1)
        self.numStep = numStep
        self.numParameter = numParameter
        self.learning_rate = learning_rate
        self.b = tf.Variable([0.])
        self.W = tf.Variable(tf.zeros([numParameter, 1]))
        self.supervisor_labels_placeholder = tf.placeholder(tf.float32, shape = (self.N, None))
        self.input_placeholder = tf.placeholder(tf.float32, shape = (self.N, None))

    def inference(self, input_placeholder, W, b):
        return tf.matmul(input_placeholder, W) + b

    def loss(self, output, supervisor_labels_placeholder):
        return tf.reduce_mean(tf.square(supervisor_labels_placeholder - output))

    def training(self, loss):
        return tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
#        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def run(self):
        feed_dict = {self.input_placeholder: self.trX, self.supervisor_labels_placeholder: self.trY}
        with tf.Session() as sess:
            output = self.inference(self.input_placeholder, self.W, self.b)
            loss = self.loss(output, self.supervisor_labels_placeholder)
            training_op = self.training(loss)

            init = tf.initialize_all_variables()
            sess.run(init)

            for step in range(self.numStep):
                sess.run(training_op, feed_dict = feed_dict)
                # print loss and parameters for each 100 steps
                if step % 100 == 0:
                    self.printProgress(sess, step, feed_dict, loss)

            self.printResult(sess)
            self.W_ = sess.run(self.W)
            self.b_ = sess.run(self.b)

    def printProgress(self, sess, step, feed_dict, loss):
        # print training progress
        print "step = " + str(step)\
        + ", loss = " + str(sess.run(loss, feed_dict = feed_dict))\
        + " " + str(sess.run(self.b[0])) + " "\
        + str(list(sess.run(self.W[i, 0]) for i in range(self.numParameter))).replace("[", ", ").replace("]", ", ")

    def printResult(self, sess):
        # print final results
        print "b = " + str(sess.run(self.b[0]))
        for i, t in enumerate(sess.run(self.W)):
            print "W" + str(i + 1) + " = " + str(t[0])

if __name__ == '__main__':
    data = pd.read_csv('test_data.csv')
    trX = data[[\
    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12',\
    'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21','X22', 'X23',\
    'X24', 'X25', 'X26', 'X27'\
    ]].fillna(0)
    trY = data['Y2']

    numStep = 10000
    numParameter = len(trX.columns)
    learning_rate = 0.5

    r = RegressionBase(trX, trY, numStep, numParameter, learning_rate)
    r.run()

    # loss = 114.568
    # b = -4.69642
    # W0 = 0.335111
    # W1 = 0.319222
    # W2 = 0.152218
    # W3 = 0.641776
