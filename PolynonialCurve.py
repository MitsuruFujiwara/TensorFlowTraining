# -*- coding: utf-8 -*-

# test code for polynomial curve fitting

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Randgen

class PolynomialCurve:

    def __init__(self, M, input_data, target_data):
        self.M = M # number of flow
        self.input_data = input_data
        self.target_data = target_data

    def y(self):
        x = self.input_data
        self.W0 = tf.Variable(np.random.random())
        self.W1 = tf.Variable(np.random.random())
        self.W2 = tf.Variable(np.random.random())
        self.W3 = tf.Variable(np.random.random())
        _y = self.W0 + self.W1 * x + self.W2 * x ** 2.0 + self.W3 * x ** 3.0
        return _y

    def loss(self):
        _loss = tf.reduce_mean(tf.square(self.y() - self.target_data))
        return _loss

    def run(self):
        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(self.loss())

        # For initializing the variables.
        init = tf.initialize_all_variables()

        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        # Fit the plane.
        for step in xrange(0, self.M):
            sess.run(train)

        print sess.run(self.W0), sess.run(self.W1), sess.run(self.W2), sess.run(self.W3)

if __name__ == '__main__':
    N = 100
    sigma = 0.3

    r = Randgen.Randgen(N, sigma)
    s = r.sin_wave_input()
    t = r.sin_wave_target()

    M = 5000
    p = PolynomialCurve(M, s, t)
    p.run()
