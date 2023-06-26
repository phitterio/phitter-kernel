import numpy
import scipy.stats
import matplotlib.pyplot as plt
import math

def getData(direction):
    sample_distribution_file = open(direction, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data

path = "C: / Users / USUARIO1 / Desktop / Fitter / data.txt"
data = getData(path)

z = 0.5384
percentiles = [scipy.stats.norm.cdf(0.5384 * i) for i in range(-3,4,2)]
x1, x2, x3, x4 = [scipy.stats.scoreatpercentile(data, 100 * x) for x in percentiles]
print(x1, x2, x3, x4)

m = x4 - x3
n = x2 - x1
p = x3 - x2
print(m,n,p)

lambda_ = (p * math.sqrt((((1 + p / m) * (1 + p / n) - 2) ** 2-4))) / (p ** 2 / (m * n) - 1)
xi_ = 0.5 * (x3 + x2)-0.5 * lambda_ + p * (p / n - p / m) / (2 * (p ** 2 / (m * n) - 1))
delta_ = z / math.acosh(0.5 *  math.sqrt((1 + p / m) * (1 + p / n)))
gamma_ = delta_ * math.asinh((p / n - p / m) * math.sqrt((1 + p / m) * (1 + p / n)-4) / (2 * (p ** 2 / (m * n) - 1)))


print(lambda_)
print(xi_)
print(gamma_)
print(delta_)


