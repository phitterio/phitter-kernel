from scipy.optimize import fsolve
import numpy

def equations(initial_solution, p):
    alpha, beta, min_, max_ = initial_solution
    _mean, _variance, _skewness, _kurtosis = [397.3594351, 866.3432464, 0.226723222, 2.985509107]
    
    ## Mean
    f1 = min_ + (alpha / ( alpha + beta )) * (max_ - min_) - _mean
    ## Variance
    f2 = ((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))) * (max_ - min_) ** 2 - _variance
    ## Skewness
    f3 = 2 * ((beta - alpha) / (alpha + beta + 2)) * numpy.sqrt((alpha + beta + 1) / (alpha * beta)) - _skewness
    ## Kurtosis
    f4 = 3 * (((alpha + beta + 1) * (2 * (alpha + beta) ** 2  + (alpha * beta) * (alpha + beta - 6))) / ((alpha * beta) * (alpha + beta + 2) * (alpha + beta + 3))) - _kurtosis
    return (f1, f2, f3, f4)

sol = fsolve(equations, (1, 1, 1, 1), [397.3594351, 866.3432464, 0.226723222, 2.985509107])
print(sol)

from scipy.optimize import least_squares
res = least_squares(equations, (1, 1, 1, 1), bounds = ((0, 0, 0, 0), (1000, 1000, 1000, 1000)), args=("p"))
print(res.x)