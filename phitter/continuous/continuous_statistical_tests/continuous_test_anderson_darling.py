import numpy


def evaluate_continuous_test_anderson_darling(distribution, continuous_measures):
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
        cdf() and num_parameters()

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
    ==========
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

    ## Parameters and preparations
    N = continuous_measures.size

    ## Calculation S
    S = numpy.sum(((2 * (numpy.arange(N) + 1) - 1) / N) * (numpy.log(distribution.cdf(continuous_measures.data)) + numpy.log(1 - distribution.cdf(continuous_measures.data[::-1]))))

    ## Calculation of indicators
    A2 = -N - S
    critical_value = continuous_measures.critical_value_ad
    p_value = continuous_measures.ad_p_value(N, A2)
    rejected = A2 >= critical_value

    ## Construction of answer
    result_test_ad = {"test_statistic": A2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}

    return result_test_ad


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_distributions import CONTINUOUS_DISTRIBUTIONS
    from continuous_measures import CONTINUOUS_MEASURES

    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    for id_distribution, distribution_class in CONTINUOUS_DISTRIBUTIONS.items():
        print(id_distribution)
        path = f"../continuous_distributions_sample/sample_{id_distribution}.txt"
        data = get_data(path)

        ## Init a instance of class
        continuous_measures = CONTINUOUS_MEASURES(data)
        distribution = distribution_class(continuous_measures=continuous_measures)
        print(evaluate_continuous_test_anderson_darling(distribution, continuous_measures))
