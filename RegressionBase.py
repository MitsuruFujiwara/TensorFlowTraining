# -*- coding: utf-8 -*-

# Base class for regression analysis with tensorflow

import numpy as np
import pandas as pd
import tensorflow as tf

class RegressionBase(object):

    def __init__(self, trX, trY, numStep, learning_rate):
        self.numStep = numStep
        self.learning_rate = learning_rate
        self.M = trX.shape[1]
        self.N = trX.shape[0]
        self.trX = trX
        self.trY = trY.reshape(self.N, 1)

    def inference(self, input_placeholder, W, b):
        return tf.matmul(input_placeholder, W) + b

    def loss(self, output, supervisor_labels_placeholder):
        return tf.reduce_mean(tf.square(supervisor_labels_placeholder - output))

    def training(self, loss):
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

    def run(self):
        self.W = tf.Variable(tf.zeros([self.M, 1]))
        self.b = tf.Variable([0.])
        supervisor_labels_placeholder = tf.placeholder(tf.float32, shape = (self.N, 1))
        input_placeholder = tf.placeholder(tf.float32, shape = (self.N, self.M))

        feed_dict = {input_placeholder: self.trX, supervisor_labels_placeholder: self.trY}

        with tf.Session() as sess:
            output = self.inference(input_placeholder, self.W, self.b)
            loss = self.loss(output, supervisor_labels_placeholder)
            training_op = self.training(loss)

            init = tf.initialize_all_variables()
            sess.run(init)

            for step in range(self.numStep):
                sess.run(training_op, feed_dict = feed_dict)
#                print sess.run(loss, feed_dict = feed_dict)
                if step % 100 == 0:
                    print "step = " + str(step) + ", loss = " + str(sess.run(loss, feed_dict = feed_dict))\
                    + " " + str(sess.run(self.b[0])) + str(list(sess.run(self.W[i, 0]) for i in range(0, self.M))).replace("[", ", ").replace("]", ", ") 

            print "b = " + str(sess.run(self.b[0]))
            for i in range(0, self.M):
                print "W" + str(i) + " = " + str(sess.run(self.W[i, 0]))

if __name__ == '__main__':
    data = pd.read_csv('test_data.csv')
    trX = data[['X1', 'X2', 'X3', 'X4']]
    trY = data['Y2']

    numStep = 10000
    learning_rate = 0.005

    r = RegressionBase(trX, trY, numStep, learning_rate)
    r.run()
