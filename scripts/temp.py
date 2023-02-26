import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu = -0.008
variance = 1/(10**7)
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()
