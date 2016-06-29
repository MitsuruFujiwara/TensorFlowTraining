# -*- coding: utf-8 -*-

# test code for polynomial curve fitting

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import Randgen

class PolynomialCurve:

    def __init__(self, N, M, trX, trY, target_y, target_x):
        self.N = N # number of flow
        self.M = M # order of the polynomial
        self.trX = trX
        self.trY = trY
        self.t_y = target_y # just for plotting data
        self.t_x = target_x # just for plotting data

    def y(self, x, w, b):
        _y = b
        for i in range(0, self.M):
            _y += tf.mul(w[i], tf.pow(x, i + 1))
        return _y

    def loss(self, hypo_y, y):
        return tf.reduce_mean(tf.square(hypo_y - y))

    def run(self):
        b = tf.Variable([0.0])
        w = tf.Variable(tf.zeros(self.M))
        x = tf.placeholder('float', shape = (100))
        y = tf.placeholder('float', shape = (100))
        y_hypo = self.y(x, w, b)

        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(self.loss(y_hypo, y))

        # For initializing the variables.
        init = tf.initialize_all_variables()

        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        # Fit the plane.
        for i in xrange(0, self.N):
            sess.run(train, feed_dict = {x: self.trX[i], y: self.trY[i]})
            if i % 100 == 0:
                print "Number of flow = " + str(i)

        self.w =[]
        self.w.append(sess.run(b[0]))
        for i in range(0, self.M):
            self.w.append(sess.run(w[i]))

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        x = np.linspace(0.0, 1.0, num=101)

        ax.plot(x, self.plt_y(x), 'k-', label='fitted line', linewidth=10, alpha=0.3)
        ax.scatter(self.t_x, self.t_y, label='target data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.grid(True)
        plt.show()

    def plt_y(self, x):
        y = self.w[0]
        for i in range(1, self.M + 1):
            y += self.w[i] * np.power(x, i)
        return y

if __name__ == '__main__':
    # Generate test data
    n = 10000
    m = 100
    k = 5
    sigma = 0.03

    r = Randgen.Randgen(m, n, sigma)
    trY = list(r.sin_wave_y())
    trX = r.x
    target_y = list(r.sin_wave_target())[0]
    target_x = r.x[0]

    p = PolynomialCurve(n, k, trX, trY, target_y, target_x)
    p.run()
    p.plot()
