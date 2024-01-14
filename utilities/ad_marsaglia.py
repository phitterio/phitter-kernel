import numpy
import scipy.optimize


def adinf(z):
    if z < 2:
        return (z**-0.5) * numpy.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
    return numpy.exp(-numpy.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))


def errfix(n, x):
    def g1(t):
        return numpy.sqrt(t) * (1 - t) * (49 * t - 102)

    def g2(t):
        return -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t

    def g3(t):
        return -130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * t) * t) * t) * t) * t

    c = 0.01265 + 0.1757 / n
    if x < c:
        return (0.0037 / (n**3) + 0.00078 / (n**2) + 0.00006 / n) * g1(x / c)
    elif x > c and x < 0.8:
        return (0.04213 / n + 0.01365 / (n**2)) * g2((x - c) / (0.8 - c))
    else:
        return (g3(x)) / n


def AD(n, z):
    """
    Parameters
    ==========
    n : size of data
    z : statistic of Anderson Darling A2

    Returns
     -  -  -  -  -  -  -
    Return the probability that critical value (integral) is less than test statstic
    """
    return adinf(z) + errfix(n, adinf(z))


def ad_critical_value(q, n):
    def f(x):
        return AD(n, x) - q
    root = scipy.optimize.newton(f, 2)
    return root


def ad_p_value(n, z):
    return 1 - AD(n, z)
