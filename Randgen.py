# -*- coding: utf-8 -*-

# Generating random numbers used for machine learning

import numpy  as np
import matplotlib.pyplot as plt

class Randgen:

    def __init__(self, N, M, sigma):
        self.N = N # Nnumber of random variables
        self.M = M # Number of steps
        self.sigma = sigma

    def sin_wave_y(self):
        self.x = list(self.__x())
        for i in range(0, self.M):
            w = np.random.normal(0.0, self.sigma, self.N)
            yield np.sin(2 * np.pi * self.x[i]) + w

    def __x(self):
        for i in range(0, self.M):
            yield np.random.uniform(0.0, 1.0, self.N)

    def sin_wave_target(self):
        self.x = list(self.__x())
        for i in range(0, self.M):
            yield np.sin(2 * np.pi * self.x[i])

if __name__ == '__main__':
    r = Randgen(100, 1000, 0.03)
    y = list(r.sin_wave_y())
    x = r.x

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x[0], y[0])
    plt.show()
