import scipy.stats
import math
from measurements__ import MEASUREMENTS

def test_chi_square(data, distribution_class):
    """
    Chi Square test to evaluate that a sample is distributed according to a probability 
    distribution.
    
    The hypothesis that the sample is distributed following the probability distribution
    is not rejected if the test statistic is less than the critical value or equivalently
    if the p - value is less than 0.05
    
    Parameters
    ==========
    data: iterable
        data set
    distribution: class
        distribution class initialized whit parameters of distribution and methods
        cdf() and get_num_parameters()
        
    Return
    ======
    result_test_chi2: dict
        1. test_statistic(float):
            sum over all classes of the value (expected - observed) ^ 2 / expected 
        2. critical_value(float):
            inverse of the distribution chi square to 0.95 with freedom degrees
            n - 1 minus the number of parameters of the distribution.
        3. p - value([0,1]):
            right - tailed probability of the test statistic for the chi - square distribution
            with the same degrees of freedom as for the critical value calculation.
        4. rejected(bool):
            decision if the null hypothesis is rejected. If it is false, it can be 
            considered that the sample is distributed according to the probability 
            distribution. If it's true, no.
    """
    ## Init a instance of class
    measurements = MEASUREMENTS(data)
    distribution = distribution_class(measurements)
    
    ## Parameters and preparations
    N = measurements.length
    frequencies = measurements.frequencies
    freedom_degrees = len(frequencies.items()) - 1
    
    ## Calculation of errors
    errors = []
    for i, observed in frequencies.items():
        expected = math.ceil(N * (distribution.pmf(i)))
        errors.append(((observed - expected) ** 2) / expected)
    
    ## Calculation of indicators
    statistic_chi2 = sum(errors)
    critical_value = scipy.stats.chi2.ppf(0.95, freedom_degrees)
    p_value = 1 -  scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    
    ## Construction of answer
    result_test_chi2 = {
        "test_statistic": statistic_chi2, 
        "critical_value": critical_value, 
        "p - value": p_value,
        "rejected": rejected
        }
    
    return result_test_chi2
    
if __name__ == "__main__":
    from distributions.bernoulli import BERNOULLI
    from distributions.binomial import BINOMIAL
    from distributions.geometric import GEOMETRIC
    from distributions.hypergeometric import HYPERGEOMETRIC
    from distributions.logarithmic import LOGARITHMIC
    from distributions.negative_binomial import NEGATIVE_BINOMIAL
    from distributions.poisson import POISSON
    from distributions.uniform import UNIFORM
    
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        return data
    
    _all_distributions = [
        BERNOULLI, BINOMIAL, GEOMETRIC, HYPERGEOMETRIC, LOGARITHMIC, NEGATIVE_BINOMIAL, POISSON, UNIFORM
    ]

    for distribution_class in _all_distributions:
        print(distribution_class.__name__)
        path = "./data/data_" + distribution_class.__name__.lower() + ".txt"
        data = get_data(path)                
        print(test_chi_square(data, distribution_class))