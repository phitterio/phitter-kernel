import sys

import joblib
import numpy

sys.path.append("../../continuous")
sys.path.append("../../continuous/measurements")
sys.path.append("../../utilities")

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
from distributions.f_4p import F_4P
from distributions.fatigue_life import FATIGUE_LIFE
from distributions.folded_normal import FOLDED_NORMAL
from distributions.frechet import FRECHET
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
from distributions.non_central_chi_square import NON_CENTRAL_CHI_SQUARE
from distributions.non_central_f import NON_CENTRAL_F
from distributions.non_central_t_student import NON_CENTRAL_T_STUDENT
from distributions.normal import NORMAL
from distributions.pareto_first_kind import PARETO_FIRST_KIND
from distributions.pareto_second_kind import PARETO_SECOND_KIND
from distributions.pert import PERT
from distributions.power_function import POWER_FUNCTION
from distributions.rayleigh import RAYLEIGH
from distributions.reciprocal import RECIPROCAL
from distributions.rice import RICE
from distributions.semicircular import SEMICIRCULAR
from distributions.t_student import T_STUDENT
from distributions.t_student_3p import T_STUDENT_3P
from distributions.trapezoidal import TRAPEZOIDAL
from distributions.triangular import TRIANGULAR
from distributions.uniform import UNIFORM
from distributions.weibull import WEIBULL
from distributions.weibull_3p import WEIBULL_3P
from measurements_continuous import MEASUREMENTS_CONTINUOUS
from test_anderson_darling_continuous import test_anderson_darling_continuous
from test_chi_square_continuous import test_chi_square_continuous
from test_kolmogorov_smirnov_continuous import test_kolmogorov_smirnov_continuous


class PHITTER_CONTINUOUS:
    def __init__(
        self,
        data: list[int | float],
        num_bins: int | None = None,
        confidence_level=0.95,
        minimum_sse=float("inf"),
    ):
        self.data = data
        self.measurements = MEASUREMENTS_CONTINUOUS(self.data, num_bins, confidence_level)
        self.confidence_level = confidence_level
        self.minimum_sse = minimum_sse
        self.distribution_results = {}
        self.none_results = {"test_statistic": None, "critical_value": None, "p_value": None, "rejected": None}

    def test(self, test_function, label: str, distribution):
        validation_test = False
        try:
            test = test_function(distribution, self.measurements, confidence_level=self.confidence_level)
            if numpy.isnan(test["test_statistic"]) == False and numpy.isinf(test["test_statistic"]) == False and test["test_statistic"] > 0:
                self.distribution_results[label] = {
                    "test_statistic": test["test_statistic"],
                    "critical_value": test["critical_value"],
                    "p_value": test["p-value"],
                    "rejected": test["rejected"],
                }
                validation_test = True
            else:
                self.distribution_results[label] = self.none_results
        except:
            self.distribution_results[label] = self.none_results
        return validation_test

    def process_distribution(self, distribution_class):
        distribution_name = distribution_class.__name__.lower()

        validate_estimation = True
        sse = 0
        try:
            distribution = distribution_class(self.measurements)
            pdf_values = [distribution.pdf(c) for c in self.measurements.central_values]
            sse = numpy.sum(numpy.power(self.measurements.densities_frequencies - pdf_values, 2.0))
        except:
            validate_estimation = False

        self.distribution_results = {}
        if validate_estimation and distribution.parameter_restrictions() and not numpy.isnan(sse) and not numpy.isinf(sse) and sse < self.minimum_sse:
            v1 = self.test(test_chi_square_continuous, "chi_square", distribution)
            v2 = self.test(test_kolmogorov_smirnov_continuous, "kolmogorov_smirnov", distribution)
            v3 = self.test(test_anderson_darling_continuous, "anderson_darling", distribution)

            if v1 or v2 or v3:
                self.distribution_results["sse"] = sse
                self.distribution_results["parameters"] = str(distribution.parameters)
                self.distribution_results["n_test_passed"] = (
                    +int(self.distribution_results["chi_square"]["rejected"] == False)
                    + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == False)
                    + int(self.distribution_results["anderson_darling"]["rejected"] == False)
                )
                self.distribution_results["n_test_null"] = (
                    +int(self.distribution_results["chi_square"]["rejected"] == None)
                    + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == None)
                    + int(self.distribution_results["anderson_darling"]["rejected"] == None)
                )

                return distribution_name, self.distribution_results
        return None

    def fit(self, n_jobs: int = 1):
        if n_jobs <= 0:
            raise Exception("n_jobs must be greater than 1")

        _ALL_CONTINUOUS_DISTRIBUTIONS = [
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
            NON_CENTRAL_CHI_SQUARE,
            NON_CENTRAL_F,
            NON_CENTRAL_T_STUDENT,
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

        if n_jobs == 1:
            processing_results = [self.process_distribution(distribution_class) for distribution_class in _ALL_CONTINUOUS_DISTRIBUTIONS]
        else:
            processing_results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self.process_distribution)(distribution_class) for distribution_class in _ALL_CONTINUOUS_DISTRIBUTIONS)
        processing_results = [r for r in processing_results if r is not None]

        sorted_results_sse = {distribution: results for distribution, results in sorted(processing_results, key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
        not_rejected_results = {distribution: results for distribution, results in sorted_results_sse.items() if results["n_test_passed"] > 0}

        return sorted_results_sse, not_rejected_results


if __name__ == "__main__":
    path = "../../continuous/data/data_beta.txt"
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    phitter_continuous = PHITTER_CONTINUOUS(data)
    sorted_results_sse, not_rejected_results = phitter_continuous.fit(n_jobs=4)

    for distribution, results in not_rejected_results.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
