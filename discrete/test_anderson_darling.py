import math
from measurements.measurements import MEASUREMENTS_DISCRETE
import sys
sys.path.append("../utilities")
import ad_marsaglia as ad

def test_anderson_darling(data, distribution_class):
    """
    Anderson Darling test to evaluate that a sample is distributed according to a probability 
    distribution.
    
    The hypothesis that the sample is distributed following the probability distribution
    is not rejected if the test statistic is less than the critical value or equivalently
    if the p-value is less than 0.05
    
    Parameters
    ==========
    data: iterable
        data set
    distribution: class
        distribution class initialized whit parameters of distribution and methods
        cdf() and get_num_parameters()
        
    Return
    ======
    result_test_ks: dict
        1. test_statistic(float):
            sum over all data(Y) of the value ((2k - 1) / N) * (ln[Fn(Y[k])] + ln[1 - Fn(Y[N - k + 1])]).
        2. critical_value(float):
            calculation of the Anderson Darling critical value using Marsaglia - Marsaglia function.
            whit size of sample N as parameter.
        3. p-value[0,1]:
            probability of the test statistic for the Anderson - Darling distribution
            whit size of sample N as parameter.
        4. rejected(bool):
            decision if the null hypothesis is rejected. If it is false, it can be 
            considered that the sample is distributed according to the probability 
            distribution. If it's true, no.
            
    References
     -  -  -  -  -  -  -  -  -  - 
    .. [1] Marsaglia, G., & Marsaglia, J. (2004). 
           Evaluating the anderson - darling distribution. 
           Journal of Statistical Software, 9(2), 1 - 5.
    .. [2] Sinclair, C. D., & Spurr, B. D. (1988).
           Approximations to the distribution function of the anderson—darling test statistic.
           Journal of the American Statistical Association, 83(404), 1190 - 1191.
    .. [3] Lewis, P. A. (1961). 
           Distribution of the Anderson - Darling statistic. 
           The Annals of Mathematical Statistics, 1118 - 1124.
    """
    ## Init a instance of class
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = distribution_class(measurements)

    ## Parameters and preparations
    N = measurements.length
    data.sort()
    
    ## Calculation S
    S = 0
    for k in range(N):
        c1 = math.log(distribution.cdf(data[k]))
        c2 = math.log(1 - distribution.cdf(data[N - k - 1]))
        c3 = (2 * (k + 1) - 1) / N
        S += c3 * (c1 + c2)
    
    ## Calculation of indicators
    A2 = -N - S
    critical_value = ad.ad_critical_value(0.95, N)
    p_value = ad.ad_p_value(N, A2)
    rejected = A2 >= critical_value
    
    ## Construction of answer
    result_test_ad = {
        "test_statistic": A2, 
        "critical_value": critical_value,
        "p-value": p_value,
        "rejected": rejected
        }
    
    return result_test_ad

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
        try:                   
            print(test_anderson_darling(data, distribution_class))
        except:
            print("q")