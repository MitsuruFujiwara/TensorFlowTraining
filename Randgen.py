# -*- coding: utf-8 -*-

# Generating random numbers used for machine learning

import numpy  as np
import matplotlib.pyplot as plt

class Randgen:

    def __init__(self, N, sigma):
        self.N = N # Nnumber of random numbers
        self.u = np.random.uniform(0.0, 1.0, self.N)
        self.sigma = sigma

    def sin_wave_input(self):
        w = np.random.normal(0.0, self.sigma, self.N)
        s = np.sin(2 * np.pi * self.u) + w
        return s

    def sin_wave_target(self):
        t = np.sin(2 * np.pi * self.u)
        return t

if __name__ == '__main__':
    r = Randgen(100, 0.03)
    s = r.sin_wave_input()
    t = r.sin_wave_target()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(r.u, t)
    plt.show()
