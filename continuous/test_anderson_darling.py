import math
from measurements.measurements import MEASUREMENTS_CONTINUOUS
import sys

sys.path.append("../utilities")
import ad_marsaglia as ad


def test_anderson_darling(data, distribution, measurements):
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
    result_test_ad = {"test_statistic": A2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}

    return result_test_ad


if __name__ == "__main__":
    from distributions.alpha import ALPHA
    from distributions.arcsine import ARCSINE
    from distributions.argus import ARGUS
    from distributions.beta import BETA
    from distributions.beta_prime import BETA_PRIME
    from distributions.beta_prime_4p import BETA_PRIME_4P
    from distributions.bradford import BRADFORD
    from distributions.burr import BURR
    from distributions.burr_4p import BURR_4P
    from distributions.cauchy import CAUCHY
    from distributions.chi_square import CHI_SQUARE
    from distributions.chi_square_3p import CHI_SQUARE_3P
    from distributions.dagum import DAGUM
    from distributions.dagum_4p import DAGUM_4P
    from distributions.erlang import ERLANG
    from distributions.erlang_3p import ERLANG_3P
    from distributions.error_function import ERROR_FUNCTION
    from distributions.exponential import EXPONENTIAL
    from distributions.exponential_2p import EXPONENTIAL_2P
    from distributions.f import F
    from distributions.fatigue_life import FATIGUE_LIFE
    from distributions.folded_normal import FOLDED_NORMAL
    from distributions.frechet import FRECHET
    from distributions.f_4p import F_4P
    from distributions.gamma import GAMMA
    from distributions.gamma_3p import GAMMA_3P
    from distributions.generalized_extreme_value import GENERALIZED_EXTREME_VALUE
    from distributions.generalized_gamma import GENERALIZED_GAMMA
    from distributions.generalized_gamma_4p import GENERALIZED_GAMMA_4P
    from distributions.generalized_logistic import GENERALIZED_LOGISTIC
    from distributions.generalized_normal import GENERALIZED_NORMAL
    from distributions.generalized_pareto import GENERALIZED_PARETO
    from distributions.gibrat import GIBRAT
    from distributions.gumbel_left import GUMBEL_LEFT
    from distributions.gumbel_right import GUMBEL_RIGHT
    from distributions.half_normal import HALF_NORMAL
    from distributions.hyperbolic_secant import HYPERBOLIC_SECANT
    from distributions.inverse_gamma import INVERSE_GAMMA
    from distributions.inverse_gamma_3p import INVERSE_GAMMA_3P
    from distributions.inverse_gaussian import INVERSE_GAUSSIAN
    from distributions.inverse_gaussian_3p import INVERSE_GAUSSIAN_3P
    from distributions.johnson_sb import JOHNSON_SB
    from distributions.johnson_su import JOHNSON_SU
    from distributions.kumaraswamy import KUMARASWAMY
    from distributions.laplace import LAPLACE
    from distributions.levy import LEVY
    from distributions.loggamma import LOGGAMMA
    from distributions.logistic import LOGISTIC
    from distributions.loglogistic import LOGLOGISTIC
    from distributions.loglogistic_3p import LOGLOGISTIC_3P
    from distributions.lognormal import LOGNORMAL
    from distributions.maxwell import MAXWELL
    from distributions.moyal import MOYAL
    from distributions.nakagami import NAKAGAMI
    from distributions.nc_chi_square import NC_CHI_SQUARE
    from distributions.nc_f import NC_F
    from distributions.nc_t_student import NC_T_STUDENT
    from distributions.normal import NORMAL
    from distributions.pareto_first_kind import PARETO_FIRST_KIND
    from distributions.pareto_second_kind import PARETO_SECOND_KIND
    from distributions.pert import PERT
    from distributions.power_function import POWER_FUNCTION
    from distributions.rayleigh import RAYLEIGH
    from distributions.reciprocal import RECIPROCAL
    from distributions.rice import RICE
    from distributions.semicircular import SEMICIRCULAR
    from distributions.trapezoidal import TRAPEZOIDAL
    from distributions.triangular import TRIANGULAR
    from distributions.t_student import T_STUDENT
    from distributions.t_student_3p import T_STUDENT_3P
    from distributions.uniform import UNIFORM
    from distributions.weibull import WEIBULL
    from distributions.weibull_3p import WEIBULL_3P

    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    _all_distributions = [
        ALPHA,
        ARCSINE,
        ARGUS,
        BETA,
        BETA_PRIME,
        BETA_PRIME_4P,
        BRADFORD,
        BURR,
        BURR_4P,
        CAUCHY,
        CHI_SQUARE,
        CHI_SQUARE_3P,
        DAGUM,
        DAGUM_4P,
        ERLANG,
        ERLANG_3P,
        ERROR_FUNCTION,
        EXPONENTIAL,
        EXPONENTIAL_2P,
        F,
        FATIGUE_LIFE,
        FOLDED_NORMAL,
        FRECHET,
        F_4P,
        GAMMA,
        GAMMA_3P,
        GENERALIZED_EXTREME_VALUE,
        GENERALIZED_GAMMA,
        GENERALIZED_GAMMA_4P,
        GENERALIZED_LOGISTIC,
        GENERALIZED_NORMAL,
        GENERALIZED_PARETO,
        GIBRAT,
        GUMBEL_LEFT,
        GUMBEL_RIGHT,
        HALF_NORMAL,
        HYPERBOLIC_SECANT,
        INVERSE_GAMMA,
        INVERSE_GAMMA_3P,
        INVERSE_GAUSSIAN,
        INVERSE_GAUSSIAN_3P,
        JOHNSON_SB,
        JOHNSON_SU,
        KUMARASWAMY,
        LAPLACE,
        LEVY,
        LOGGAMMA,
        LOGISTIC,
        LOGLOGISTIC,
        LOGLOGISTIC_3P,
        LOGNORMAL,
        MAXWELL,
        MOYAL,
        NAKAGAMI,
        NC_CHI_SQUARE,
        NC_F,
        NC_T_STUDENT,
        NORMAL,
        PARETO_FIRST_KIND,
        PARETO_SECOND_KIND,
        PERT,
        POWER_FUNCTION,
        RAYLEIGH,
        RECIPROCAL,
        RICE,
        SEMICIRCULAR,
        TRAPEZOIDAL,
        TRIANGULAR,
        T_STUDENT,
        T_STUDENT_3P,
        UNIFORM,
        WEIBULL,
        WEIBULL_3P,
    ]

    _my_distributions = [DAGUM, DAGUM_4P, POWER_FUNCTION, RICE, RAYLEIGH, RECIPROCAL, T_STUDENT, GENERALIZED_GAMMA_4P]
    _my_distributions = [PERT]
    for distribution_class in _my_distributions:
        print(distribution_class.__name__)
        path = "./data/data_" + distribution_class.__name__.lower() + ".txt"
        data = get_data(path)

        ## Init a instance of class
        measurements = MEASUREMENTS_CONTINUOUS(data)
        distribution = distribution_class(measurements)
        print(test_anderson_darling(data, distribution, measurements))
