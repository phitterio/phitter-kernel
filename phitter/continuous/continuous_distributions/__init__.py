from .alpha import ALPHA
from .arcsine import ARCSINE
from .argus import ARGUS
from .beta import BETA
from .beta_prime import BETA_PRIME
from .beta_prime_4p import BETA_PRIME_4P
from .bradford import BRADFORD
from .burr import BURR
from .burr_4p import BURR_4P
from .cauchy import CAUCHY
from .chi_square import CHI_SQUARE
from .chi_square_3p import CHI_SQUARE_3P
from .dagum import DAGUM
from .dagum_4p import DAGUM_4P
from .erlang import ERLANG
from .erlang_3p import ERLANG_3P
from .error_function import ERROR_FUNCTION
from .exponential import EXPONENTIAL
from .exponential_2p import EXPONENTIAL_2P
from .f import F
from .f_4p import F_4P
from .fatigue_life import FATIGUE_LIFE
from .folded_normal import FOLDED_NORMAL
from .frechet import FRECHET
from .gamma import GAMMA
from .gamma_3p import GAMMA_3P
from .generalized_extreme_value import GENERALIZED_EXTREME_VALUE
from .generalized_gamma import GENERALIZED_GAMMA
from .generalized_gamma_4p import GENERALIZED_GAMMA_4P
from .generalized_logistic import GENERALIZED_LOGISTIC
from .generalized_normal import GENERALIZED_NORMAL
from .generalized_pareto import GENERALIZED_PARETO
from .gibrat import GIBRAT
from .gumbel_left import GUMBEL_LEFT
from .gumbel_right import GUMBEL_RIGHT
from .half_normal import HALF_NORMAL
from .hyperbolic_secant import HYPERBOLIC_SECANT
from .inverse_gamma import INVERSE_GAMMA
from .inverse_gamma_3p import INVERSE_GAMMA_3P
from .inverse_gaussian import INVERSE_GAUSSIAN
from .inverse_gaussian_3p import INVERSE_GAUSSIAN_3P
from .johnson_sb import JOHNSON_SB
from .johnson_su import JOHNSON_SU
from .kumaraswamy import KUMARASWAMY
from .laplace import LAPLACE
from .levy import LEVY
from .loggamma import LOGGAMMA
from .logistic import LOGISTIC
from .loglogistic import LOGLOGISTIC
from .loglogistic_3p import LOGLOGISTIC_3P
from .lognormal import LOGNORMAL
from .maxwell import MAXWELL
from .moyal import MOYAL
from .nakagami import NAKAGAMI
from .non_central_chi_square import NON_CENTRAL_CHI_SQUARE
from .non_central_f import NON_CENTRAL_F
from .non_central_t_student import NON_CENTRAL_T_STUDENT
from .normal import NORMAL
from .pareto_first_kind import PARETO_FIRST_KIND
from .pareto_second_kind import PARETO_SECOND_KIND
from .pert import PERT
from .power_function import POWER_FUNCTION
from .rayleigh import RAYLEIGH
from .reciprocal import RECIPROCAL
from .rice import RICE
from .semicircular import SEMICIRCULAR
from .t_student import T_STUDENT
from .t_student_3p import T_STUDENT_3P
from .trapezoidal import TRAPEZOIDAL
from .triangular import TRIANGULAR
from .uniform import UNIFORM
from .weibull import WEIBULL
from .weibull_3p import WEIBULL_3P

CONTINUOUS_DISTRIBUTIONS = {
    "alpha": ALPHA,
    "arcsine": ARCSINE,
    "argus": ARGUS,
    "beta": BETA,
    "beta_prime": BETA_PRIME,
    "beta_prime_4p": BETA_PRIME_4P,
    "bradford": BRADFORD,
    "burr": BURR,
    "burr_4p": BURR_4P,
    "cauchy": CAUCHY,
    "chi_square": CHI_SQUARE,
    "chi_square_3p": CHI_SQUARE_3P,
    "dagum": DAGUM,
    "dagum_4p": DAGUM_4P,
    "erlang": ERLANG,
    "erlang_3p": ERLANG_3P,
    "error_function": ERROR_FUNCTION,
    "exponential": EXPONENTIAL,
    "exponential_2p": EXPONENTIAL_2P,
    "f": F,
    "fatigue_life": FATIGUE_LIFE,
    "folded_normal": FOLDED_NORMAL,
    "frechet": FRECHET,
    "f_4p": F_4P,
    "gamma": GAMMA,
    "gamma_3p": GAMMA_3P,
    "generalized_extreme_value": GENERALIZED_EXTREME_VALUE,
    "generalized_gamma": GENERALIZED_GAMMA,
    "generalized_gamma_4p": GENERALIZED_GAMMA_4P,
    "generalized_logistic": GENERALIZED_LOGISTIC,
    "generalized_normal": GENERALIZED_NORMAL,
    "generalized_pareto": GENERALIZED_PARETO,
    "gibrat": GIBRAT,
    "gumbel_left": GUMBEL_LEFT,
    "gumbel_right": GUMBEL_RIGHT,
    "half_normal": HALF_NORMAL,
    "hyperbolic_secant": HYPERBOLIC_SECANT,
    "inverse_gamma": INVERSE_GAMMA,
    "inverse_gamma_3p": INVERSE_GAMMA_3P,
    "inverse_gaussian": INVERSE_GAUSSIAN,
    "inverse_gaussian_3p": INVERSE_GAUSSIAN_3P,
    "johnson_sb": JOHNSON_SB,
    "johnson_su": JOHNSON_SU,
    "kumaraswamy": KUMARASWAMY,
    "laplace": LAPLACE,
    "levy": LEVY,
    "loggamma": LOGGAMMA,
    "logistic": LOGISTIC,
    "loglogistic": LOGLOGISTIC,
    "loglogistic_3p": LOGLOGISTIC_3P,
    "lognormal": LOGNORMAL,
    "maxwell": MAXWELL,
    "moyal": MOYAL,
    "nakagami": NAKAGAMI,
    "non_central_chi_square": NON_CENTRAL_CHI_SQUARE,
    "non_central_f": NON_CENTRAL_F,
    "non_central_t_student": NON_CENTRAL_T_STUDENT,
    "normal": NORMAL,
    "pareto_first_kind": PARETO_FIRST_KIND,
    "pareto_second_kind": PARETO_SECOND_KIND,
    "pert": PERT,
    "power_function": POWER_FUNCTION,
    "rayleigh": RAYLEIGH,
    "reciprocal": RECIPROCAL,
    "rice": RICE,
    "semicircular": SEMICIRCULAR,
    "trapezoidal": TRAPEZOIDAL,
    "triangular": TRIANGULAR,
    "t_student": T_STUDENT,
    "t_student_3p": T_STUDENT_3P,
    "uniform": UNIFORM,
    "weibull": WEIBULL,
    "weibull_3p": WEIBULL_3P,
}
