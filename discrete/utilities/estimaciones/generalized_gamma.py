import math
import numpy

def get_measurements(data: list)  - > dict:
    import scipy.stats
    import numpy
    measurements = {}
    
    miu_3 = scipy.stats.moment(data, 3)
    miu_4 = scipy.stats.moment(data, 4)
    mean = numpy.mean(data)
    variance = numpy.var(data, ddof=1)
    skewness = miu_3 / pow(numpy.std(data, ddof=1),3)
    kurtosis = miu_4 / pow(numpy.std(data, ddof=1),4)
    median = numpy.median(data)
    mode = scipy.stats.mode(data)[0][0]
    
    measurements.mean = mean
    measurements.variance = variance
    measurements.skewness = skewness
    measurements.kurtosis = kurtosis
    measurements.data = data
    measurements.median = median
    measurements.mode = mode
    
    return measurements

def getData(direction):
    sample_distribution_file = open(direction, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data

path = "../data/data_generalized_gamma.txt"
data = getData(path) 
measurements = MEASUREMENTS(data)

def equations(initial_solution, data_mean, data_variance, data_skewness):
    a, d, p = initial_solution
    
    E = lambda r: a ** r * (math.gamma((d + r) / p) / math.gamma(d / p))
    
    parametric_mean = E(1)
    parametric_variance = E(2) - E(1) ** 2
    parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
    # parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2

    ## System Equations
    eq1 = parametric_mean - data_mean
    eq2 = parametric_variance - data_variance
    eq3 = parametric_skewness - data_skewness

    return (eq1, eq2, eq3)

from scipy.optimize import fsolve
import time
ti = time.time()
sol = fsolve(equations, (1, 1, 1), (measurements.mean, measurements.variance, measurements.skewness))
print(sol)
print(time.time() - ti)

from scipy.optimize import least_squares
import time
ti = time.time()
res = least_squares(equations, (1, 1, 1), bounds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf)), args=(measurements.mean, measurements.variance, measurements.skewness))
print(res.x)
print(time.time() - ti)

print(0.3657 / 0.0039)
