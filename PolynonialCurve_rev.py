# -*- coding: utf-8 -*-

# Polynomial curve fitting with using RgeressionBase class

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import Randgen
from RegressionBase import RegressionBase

class PolynomialCurve(RegressionBase):
    
    def __init__(self, trX, trY, numStep, numParameter, learning_rate):
        RegressionBase.__init__(self, trX, trY, numStep, numParameter, learning_rate)
        
    def inference(self, input_placeholder, W, b):
        y = b[0]
        x = input_placeholder
        for i in range(self.numParameter):
            y += tf.mul(W[i, 0], tf.pow(x, i + 1))
        return y

    def training(self, loss):
        return tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        x = np.linspace(0.0, 1.0, num=101)
        w = sess.run(self.W)
        b = sess.run(self.b)

        ax.plot(x, self.inference(x), 'k-', label='fitted line', linewidth=10, alpha=0.3)
        ax.scatter(self.trX, self.trY, label='target data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # Generate test data
    n = 1000 # number of data sets
    m = 1 # number of random variable in a data set
    sigma = 0.03 # volatility of data set

    r = Randgen.Randgen(m, n, sigma)
    trY = pd.Series(list(r.sin_wave_y())).reshape(n, 1)
    trX = pd.Series(r.x).reshape(n, 1)
    
    numStep = 1000 # number of trainig
    numParameter = 9 # order of polynomial
    learning_rate = 0.5
    
    p = PolynomialCurve(trX, trY, numStep, numParameter, learning_rate)
    p.run()
    p.plot