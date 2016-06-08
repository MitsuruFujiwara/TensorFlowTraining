# -*- coding: utf-8 -*-

# test code for caluculating pi with Gauss–Legendre algorithm

import tensorflow as tf
import numpy as np

class Calc_pi:

    def __init__(self, N):
        self.N = N # number of steps

    def a(self, n):
        if n == 0:
            return 1.0
        return tf.div(tf.add(self.a(n-1), self.b(n-1)), 2.0)

    def b(self,n):
        if n == 0:
            return 1.0 / tf.sqrt(2.0)
        return tf.sqrt(tf.mul(self.a(n-1), self.b(n-1)))

    def t(self, n):
        if n == 0:
            return 1.0 / 4.0
        return tf.sub(self.t(n-1), tf.mul(self.p(n-1), tf.pow(tf.sub(self.a(n-1), self.a(n)), 2.0)))

    def p(self, n):
        if n == 0:
            return 1.0
        return tf.mul(2.0, self.p(n-1))

    def get_pi(self):
        return tf.div(tf.pow(tf.add(self.a(self.N), self.b(self.N)), 2.0), (tf.mul(4.0, self.t(self.N))))

if __name__ == '__main__':
    N = 10
    c = Calc_pi(N)
    pi = c.get_pi()

    sess = tf.Session()
    result = sess.run(pi)

    print result
