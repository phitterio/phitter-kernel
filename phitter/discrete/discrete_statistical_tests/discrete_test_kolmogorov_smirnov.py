import numpy
import scipy.stats


def evaluate_discrete_test_kolmogorov_smirnov(distribution, discrete_measures):
    """
    Kolmogorov Smirnov test to evaluate that a sample is distributed according to a probability
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
        cdf() and num_parameters()

    Return
    ======
    result_test_ks: dict
        1. test_statistic(float):
            sum over all data of the value |Sn - Fn|
        2. critical_value(float):
            inverse of the kolmogorov - smirnov distribution to 0.95 whit size of
            sample N as parameter.
        3. p-value[0,1]:
            probability of the test statistic for the kolmogorov - smirnov distribution
            whit size of sample N as parameter.
        4. rejected(bool):
            decision if the null hypothesis is rejected. If it is false, it can be
            considered that the sample is distributed according to the probability
            distribution. If it's true, no.
    """
    ## Calculation of errors
    Fn = distribution.cdf(discrete_measures.data)
    errors_before = numpy.abs(discrete_measures.Sn_ks_before[discrete_measures.idx_ks] - Fn[discrete_measures.idx_ks])
    errors_after = numpy.abs(discrete_measures.Sn_ks_after[discrete_measures.idx_ks] - Fn[discrete_measures.idx_ks])
    
    ## This is not equal to the previous
    # statistic_ks, p_value = scipy.stats.kstest(discrete_measures.data, distribution.cdf)

    ## Calculation of indicators
    statistic_ks = max(numpy.max(errors_before), numpy.max(errors_after))
    critical_value = discrete_measures.critical_value_ks
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, discrete_measures.size)
    rejected = statistic_ks >= critical_value

    ## Construction of answer
    result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}

    return result_test_ks


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from discrete_distributions import DISCRETE_DISTRIBUTIONS
    from discrete_measures import DiscreteMeasures

    def get_data(path: str) -> list[int]:
        sample_distribution_file = open(path, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    for id_distribution, distribution_class in DISCRETE_DISTRIBUTIONS.items():
        print(id_distribution)
        path = f"../discrete_distributions_sample/sample_{id_distribution}.txt"
        data = get_data(path)

        ## Init a instance of class
        discrete_measures = DiscreteMeasures(data)
        distribution = distribution_class(discrete_measures=discrete_measures)
        print(evaluate_discrete_test_kolmogorov_smirnov(distribution, discrete_measures))
