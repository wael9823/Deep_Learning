#!/usr/bin/env python3

# First Problem of HW1 of Deep Learning
# Author = "Wael Mohammed"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

fig, ax = plt.subplots(1, 1)
data = np.load("PoissonX.npy")
a = np.unique(data)

d = np.diff(np.unique(data)).min()
left_of_first_bin = data.min() - float(d)/2
right_of_last_bin = data.max() + float(d)/2
plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d), density = True)
plt.xticks(a)
plt.title("Poisson distribution with parameter = 4.3")
plt.xlabel("Data")
plt.ylabel("Probability")
mu = 4.3
ax.plot(a, poisson.pmf(a,mu), 'b', label='poisson pmf')
plt.show()