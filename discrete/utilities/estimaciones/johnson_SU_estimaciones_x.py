import scipy.stats
import math

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

def get_parameters(measurements):
    ## Percentiles
    z = 0.5384
    percentiles = [scipy.stats.norm.cdf(0.5384 * i) for i in range(-3,4,2)]
    x1, x2, x3, x4 = [scipy.stats.scoreatpercentile(measurements.data, 100 * x) for x in percentiles]
    
    ## Calculation m,n,p
    m = x4 - x3
    n = x2 - x1
    p = x3 - x2
    
    ## Calculation distribution parameters
    lambda_ = (2 * p * math.sqrt((m / p) * (n / p) - 1)) / ((m / p + n / p - 2) * math.sqrt(m / p + n / p + 2))
    xi_ = 0.5 * (x3 + x2) + p * (n / p - m / p) / (2 * (m / p + n / p - 2))
    delta_ = 2 * z / math.acosh(0.5 * (m / p + n / p))
    gamma_ = delta_ * math.asinh((n / p - m / p) / (2 * math.sqrt((m / p) * (n / p) - 1)))
    
    parameters = {"xi": xi_, "lambda": lambda_, "gamma": gamma_, "delta": delta_}
    return parameters

def getData(direction):
    sample_distribution_file = open(direction, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    return data

path = "C: / Users / USUARIO / Desktop / Fitter / data / data_johnson_SU.txt"
data = getData(path)
measurements = MEASUREMENTS(data)

print(get_parameters(measurements))