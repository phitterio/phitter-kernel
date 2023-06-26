if __name__ == "__main__":
    from distributions.beta import BETA
    from distributions.burr import BURR
    from distributions.cauchy import CAUCHY
    from distributions.chi_square import CHI_SQUARE
    from distributions.dagum import DAGUM
    from distributions.erlang import ERLANG
    from distributions.error_function import ERROR_FUNCTION
    from distributions.exponencial import EXPONENCIAL
    from distributions.f import F
    from distributions.fatigue_life import FATIGUE_LIFE
    from distributions.frechet import FRECHET
    from distributions.gamma import GAMMA
    from distributions.generalized_extreme_value import GENERALIZED_EXTREME_VALUE
    from distributions.generalized_gamma import GENERALIZED_GAMMA
    from distributions.generalized_gamma_4P import GENERALIZED_GAMMA_4P
    from distributions.generalized_logistic import  GENERALIZED_LOGISTIC
    from distributions.generalized_normal import GENERALIZED_NORMAL
    from distributions.gumbel_left import GUMBEL_LEFT
    from distributions.gumbel_right import GUMBEL_RIGHT
    from distributions.hypernolic_secant import HYPERBOLIC_SECANT
    from distributions.inverse_gamma import INVERSE_GAMMA
    from distributions.inverse_gaussian import INVERSE_GAUSSIAN
    from distributions.johnson_SB import JOHNSON_SB
    from distributions.johnson_SU import JOHNSON_SU
    from distributions.kumaraswamy import KUMARASWAMY
    from distributions.laplace import LAPLACE
    from distributions.levy import LEVY
    from distributions.loggamma import LOGGAMMA
    from distributions.logistic import LOGISTIC
    from distributions.loglogistic import LOGLOGISTIC
    from distributions.lognormal import LOGNORMAL
    from distributions.nakagami import NAKAGAMI
    from distributions.normal import NORMAL
    from distributions.pareto_first_kind import PARETO_FIRST_KIND
    from distributions.pareto_second_kind import PARETO_SECOND_KIND
    from distributions.pearson_type_6 import PEARSON_TYPE_6
    from distributions.pert import PERT
    from distributions.power_function import POWER_FUNCTION
    from distributions.rayleigh import RAYLEIGH
    from distributions.reciprocal import RECIPROCAL
    from distributions.rice import RICE
    from distributions.t import T
    from distributions.trapezoidal import TRAPEZOIDAL
    from distributions.triangular import TRIANGULAR
    from distributions.uniform import UNIFORM
    from distributions.weibull import WEIBULL
    
    from test_chi_square import test_chi_square
    from test_kolmogorov_smirnov import test_kolmogorov_smirnov
    from test_anderson_darling import test_anderson_darling
    
    from colorama import init, Fore, Back, Style
    init(convert=True)
    
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data
    
    _all_distributions = [
        BETA, BURR, CAUCHY, CHI_SQUARE, DAGUM, ERLANG, ERROR_FUNCTION, 
        EXPONENCIAL, F, FATIGUE_LIFE, FRECHET, GAMMA, GENERALIZED_EXTREME_VALUE, 
        GENERALIZED_GAMMA, GENERALIZED_LOGISTIC, GENERALIZED_NORMAL, GUMBEL_LEFT, 
        GUMBEL_RIGHT, HYPERBOLIC_SECANT, INVERSE_GAMMA, INVERSE_GAUSSIAN, JOHNSON_SB, 
        JOHNSON_SU, KUMARASWAMY, LAPLACE, LEVY, LOGGAMMA, LOGISTIC, LOGLOGISTIC,
        LOGNORMAL,  NAKAGAMI, NORMAL, PARETO_FIRST_KIND, PARETO_SECOND_KIND, PEARSON_TYPE_6, 
        PERT, TRAPEZOIDAL, TRIANGULAR,UNIFORM, WEIBULL
    ]

    _my_distributions = [LOGGAMMA, PEARSON_TYPE_6]
    _my_distributions = [BETA, LEVY, RICE, INVERSE_GAMMA, GENERALIZED_GAMMA_4P, F]
    for distribution_class in _my_distributions:
        print(Fore.BLUE + distribution_class.__name__)
        print(Fore.BLUE + "=" * len(distribution_class.__name__))
        # path = "./data/data_" + distribution_class.__name__.lower() + ".txt"
        path = ". / data_tet.txt"
        data = get_data(path)
                
        if(test_chi_square(data, distribution_class)["rejected"] == False):
            print(Fore.CYAN + str("Test Chi Square: ") + str(test_chi_square(data, distribution_class)))
        else:
            print(Fore.RED + str("Test Chi Square: ") + str(test_chi_square(data, distribution_class)))
            
        if(test_kolmogorov_smirnov(data, distribution_class)["rejected"] == False):
            print(Fore.CYAN + str("Test Kolmogorov - Smirnov: ") + str(test_kolmogorov_smirnov(data, distribution_class)))
        else:
            print(Fore.RED + str("Test Kolmogorov - Smirnov: ") + str(test_kolmogorov_smirnov(data, distribution_class)))
            
        if(test_anderson_darling(data, distribution_class)["rejected"] == False):
            print(Fore.CYAN + str("Test Anderson - Darling: ") + str(test_anderson_darling(data, distribution_class)))
        else:
            print(Fore.RED + str("Test Anderson - Darling: ") + str(test_anderson_darling(data, distribution_class)))
        print(Style.RESET_ALL)