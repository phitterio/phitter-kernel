import sys

import joblib
import numpy

sys.path.append("../../discrete")
sys.path.append("../../discrete/measurements")

from distributions.bernoulli import BERNOULLI
from distributions.binomial import BINOMIAL
from distributions.geometric import GEOMETRIC
from distributions.hypergeometric import HYPERGEOMETRIC
from distributions.logarithmic import LOGARITHMIC
from distributions.negative_binomial import NEGATIVE_BINOMIAL
from distributions.poisson import POISSON
from distributions.uniform import UNIFORM
from measurements_discrete import MEASUREMENTS_DISCRETE
from test_chi_square_discrete import test_chi_square_discrete
from test_kolmogorov_smirnov_discrete import test_kolmogorov_smirnov_discrete


class PHITTER_DISCRETE:
    def __init__(
        self,
        data: list[int | float],
        confidence_level=0.95,
        minimum_sse=float("inf"),
    ):
        self.data = data
        self.measurements = MEASUREMENTS_DISCRETE(self.data)
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
            pmf_values = [distribution.pmf(d) for d in self.measurements.domain]
            sse = numpy.sum(numpy.power(numpy.array(pmf_values) - numpy.array(self.measurements.frequencies_pmf), 2.0))
        except:
            validate_estimation = False

        self.distribution_results = {}
        if validate_estimation and distribution.parameter_restrictions() and not numpy.isnan(sse) and not numpy.isinf(sse) and sse < self.minimum_sse:
            v1 = self.test(test_chi_square_discrete, "chi_square", distribution)
            v2 = self.test(test_kolmogorov_smirnov_discrete, "kolmogorov_smirnov", distribution)

            if v1 or v2:
                self.distribution_results["sse"] = sse
                self.distribution_results["parameters"] = str(distribution.parameters)
                self.distribution_results["n_test_passed"] = int(self.distribution_results["chi_square"]["rejected"] == False) + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == False)
                self.distribution_results["n_test_null"] = int(self.distribution_results["chi_square"]["rejected"] == None) + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == None)
                return distribution_name, self.distribution_results
            
        return None

    def fit(self, n_jobs: int = 1):
        if n_jobs <= 0:
            raise Exception("n_jobs must be greater than 1")

        _ALL_DISCRETE_DISTRIBUTIONS = [BERNOULLI, BINOMIAL, GEOMETRIC, HYPERGEOMETRIC, LOGARITHMIC, NEGATIVE_BINOMIAL, POISSON, UNIFORM]

        if n_jobs == 1:
            processing_results = [self.process_distribution(distribution_class) for distribution_class in _ALL_DISCRETE_DISTRIBUTIONS]
        else:
            processing_results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(self.process_distribution)(distribution_class) for distribution_class in _ALL_DISCRETE_DISTRIBUTIONS)
        processing_results = [r for r in processing_results if r is not None]

        sorted_results_sse = {distribution: results for distribution, results in sorted(processing_results, key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
        not_rejected_results = {distribution: results for distribution, results in sorted_results_sse.items() if results["n_test_passed"] > 0}

        return sorted_results_sse, not_rejected_results


if __name__ == "__main__":
    path = "../../discrete/data/data_binomial.txt"
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    phitter_discrete = PHITTER_DISCRETE(data)
    sorted_results_sse, not_rejected_results = phitter_discrete.fit()

    for distribution, results in not_rejected_results.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
