import collections
import math
import numpy
import scipy.optimize
import scipy.special as sc
import scipy.stats

class BERNOULLI:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.p = self.parameters["p"]
    def cdf(self, x: float) -> float:
        if x < 0:
            result = 0
        elif x >= 0 and x < 1:
            result = 1 - self.p
        else:
            result = 1
        return result
    def pmf(self, x: int) -> float:
        result = (self.p**x) * (1 - self.p) ** (1 - x)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0 and self.p < 1
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        p = measurements.mean
        parameters = {"p": p}
        return parameters

class BINOMIAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.n = self.parameters["n"]
        self.p = self.parameters["p"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.binom.cdf(x, self.n, self.p)
        return result
    def pmf(self, x: int) -> float:
        result = sc.comb(self.n, x) * (self.p**x) * ((1 - self.p) ** (self.n - x))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0 and self.p < 1
        v2 = self.n > 0
        v3 = type(self.n) == int
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        p = 1 - measurements.variance / measurements.mean
        n = int(round(measurements.mean / p, 0))
        parameters = {"n": n, "p": p}
        return parameters

class GEOMETRIC:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.p = self.parameters["p"]
    def cdf(self, x: float) -> float:
        result = 1 - (1 - self.p) ** (x + 1)
        return result
    def pmf(self, x: int) -> float:
        result = self.p * (1 - self.p) ** (x - 1)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0 and self.p < 1
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        p = 1 / measurements.mean
        parameters = {"p": p}
        return parameters

class HYPERGEOMETRIC:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.N = self.parameters["N"]
        self.K = self.parameters["K"]
        self.n = self.parameters["n"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.hypergeom.cdf(x, self.N, self.n, self.K)
        return result
    def pmf(self, x: int) -> float:
        result = sc.comb(self.K, x) * sc.comb(self.N - self.K, self.n - x) / sc.comb(self.N, self.n)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.N > 0 and type(self.N) == int
        v2 = self.K > 0 and type(self.K) == int
        v3 = self.n > 0 and type(self.n) == int
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            N, K, n = initial_solution
            parametric_mean = n * K / N
            parametric_variance = (n * K / N) * ((N - K) / N) * ((N - n) / (N - 1))
            parametric_mode = math.floor((n + 1) * (K + 1) / (N + 2))
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_mode - measurements.mode
            return (eq1, eq2, eq3)
        bnds = ((measurements.max, measurements.max, 1), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (measurements.max * 5, measurements.max * 3, measurements.max)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"N": round(solution.x[0]), "K": round(solution.x[1]), "n": round(solution.x[2])}
        return parameters

class LOGARITHMIC:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.p = self.parameters["p"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.logser.cdf(x, self.p)
        return result
    def pmf(self, x: int) -> float:
        result = -(self.p**x) / (math.log(1 - self.p) * x)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0 and self.p < 1
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            p = initial_solution
            parametric_mean = -p / ((1 - p) * math.log(1 - p))
            eq1 = parametric_mean - measurements.mean
            return eq1
        solution = scipy.optimize.least_squares(equations, 0.5, bounds=(0, 1), args=([measurements]))
        parameters = {"p": solution.x[0]}
        return parameters

class NEGATIVE_BINOMIAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.p = self.parameters["p"]
        self.r = self.parameters["r"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.beta.cdf(self.p, self.r, x + 1)
        return result
    def pmf(self, x: int) -> float:
        result = sc.comb(self.r + x - 1, x) * (self.p**self.r) * ((1 - self.p) ** x)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0 and self.p < 1
        v2 = self.r > 0
        v3 = type(self.r) == int
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        p = measurements.mean / measurements.variance
        r = round(measurements.mean * p / (1 - p))
        parameters = {"p": p, "r": r}
        return parameters

class POISSON:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.poisson.cdf(x, self.lambda_)
        return result
    def pmf(self, x: int) -> float:
        result = (self.lambda_**x) * math.exp(-self.lambda_) / math.factorial(x)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        lambda_ = measurements.mean
        parameters = {"lambda": lambda_}
        return parameters

class UNIFORM:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
    def cdf(self, x: float) -> float:
        return (x - self.a + 1) / (self.b - self.a + 1)
    def pmf(self, x: int) -> float:
        return 1 / (self.b - self.a + 1)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        v2 = type(self.b) == int
        v3 = type(self.a) == int
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        a = round(measurements.min)
        b = round(measurements.max)
        parameters = {"a": a, "b": b}
        return parameters

class MEASUREMENTS_DISCRETE:
    def __init__(self, data: list[int]):
        self.data = data
        self.length = len(data)
        self.min = min(data)
        self.max = max(data)
        self.mean = numpy.mean(data)
        self.variance = numpy.var(data, ddof=1)
        self.std = numpy.std(data, ddof=1)
        self.skewness = scipy.stats.moment(data, 3) / pow(self.std, 3)
        self.kurtosis = scipy.stats.moment(data, 4) / pow(self.std, 4)
        self.median = int(numpy.median(self.data))
        self.mode = int(scipy.stats.mode(data, keepdims=True)[0][0])
        self.histogram = self.get_histogram()
        self.domain = list(self.histogram.keys())
        self.frequencies = list(self.histogram.values())
        self.frequencies_pmf = list(map(lambda x: x / self.length, self.histogram.values()))
    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})
    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})
    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}
    def get_histogram(self) -> dict[int, float | int]:
        histogram = collections.Counter(self.data)
        return {k: v for k, v in sorted(histogram.items(), key=lambda item: item[0])}

def test_chi_square_discrete(data, distribution, measurements, confidence_level=0.95):
    N = measurements.length
    freedom_degrees = len(measurements.histogram.items()) - 1
    errors = []
    for i, observed in measurements.histogram.items():
        expected = math.ceil(N * (distribution.pmf(i)))
        errors.append(((observed - expected) ** 2) / expected)
    statistic_chi2 = sum(errors)
    critical_value = scipy.stats.chi2.ppf(confidence_level, freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {"test_statistic": statistic_chi2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_chi2

def test_kolmogorov_smirnov_discrete(data, distribution, measurements, confidence_level=0.95):
    N = measurements.length
    data.sort()
    errors = []
    for i in range(N):
        Sn = (i + 1) / N
        if i < N - 1:
            if data[i] != data[i + 1]:
                Fn = distribution.cdf(data[i])
                errors.append(abs(Sn - Fn))
            else:
                Fn = 0
        else:
            Fn = distribution.cdf(data[i])
            errors.append(abs(Sn - Fn))
    statistic_ks = max(errors)
    critical_value = scipy.stats.kstwo.ppf(confidence_level, N)
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, N)
    rejected = statistic_ks >= critical_value
    result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ks


def phitter_discrete(data, confidence_level=0.95):
    _all_distributions = [BERNOULLI, BINOMIAL, GEOMETRIC, HYPERGEOMETRIC, LOGARITHMIC, NEGATIVE_BINOMIAL, POISSON, UNIFORM]
    measurements = MEASUREMENTS_DISCRETE(data)

    NONE_RESULTS = {
        "test_statistic": None,
        "critical_value": None,
        "p_value": None,
        "rejected": None,
    }

    RESPONSE = {}
    for distribution_class in _all_distributions:
        distribution_name = distribution_class.__name__.lower()

        validate_estimation = True
        sse = 0
        try:
            distribution = distribution_class(measurements)
            pmf_values = [distribution.pmf(d) for d in measurements.domain]
            sse = numpy.sum(numpy.power(numpy.array(pmf_values) - numpy.array(measurements.frequencies_pmf), 2.0))
        except:
            validate_estimation = False

        DISTRIBUTION_RESULTS = {}
        v1, v2 = False, False
        if validate_estimation and distribution.parameter_restrictions() and not math.isnan(sse) and not math.isinf(sse):
            try:
                chi2_test = test_chi_square_discrete(data, distribution, measurements, confidence_level=confidence_level)
                if numpy.isnan(chi2_test["test_statistic"]) == False and math.isinf(chi2_test["test_statistic"]) == False and chi2_test["test_statistic"] > 0:
                    DISTRIBUTION_RESULTS["chi_square"] = {
                        "test_statistic": chi2_test["test_statistic"],
                        "critical_value": chi2_test["critical_value"],
                        "p_value": chi2_test["p-value"],
                        "rejected": chi2_test["rejected"],
                    }
                    v1 = True
                else:
                    DISTRIBUTION_RESULTS["chi_square"] = NONE_RESULTS
            except:
                DISTRIBUTION_RESULTS["chi_square"] = NONE_RESULTS

            try:
                ks_test = test_kolmogorov_smirnov_discrete(data, distribution, measurements, confidence_level=confidence_level)
                if numpy.isnan(ks_test["test_statistic"]) == False and math.isinf(ks_test["test_statistic"]) == False and ks_test["test_statistic"] > 0:
                    DISTRIBUTION_RESULTS["kolmogorov_smirnov"] = {
                        "test_statistic": ks_test["test_statistic"],
                        "critical_value": ks_test["critical_value"],
                        "p_value": ks_test["p-value"],
                        "rejected": ks_test["rejected"],
                    }
                    v2 = True
                else:
                    DISTRIBUTION_RESULTS["anderson_darling"] = NONE_RESULTS
            except:
                DISTRIBUTION_RESULTS["kolmogorov_smirnov"] = NONE_RESULTS

            if v1 or v2:
                DISTRIBUTION_RESULTS["sse"] = sse
                DISTRIBUTION_RESULTS["parameters"] = str(distribution.parameters)
                DISTRIBUTION_RESULTS["n_test_passed"] = int(DISTRIBUTION_RESULTS["chi_square"]["rejected"] == False) + int(DISTRIBUTION_RESULTS["kolmogorov_smirnov"]["rejected"] == False)
                DISTRIBUTION_RESULTS["n_test_null"] = int(DISTRIBUTION_RESULTS["chi_square"]["rejected"] == None) + int(DISTRIBUTION_RESULTS["kolmogorov_smirnov"]["rejected"] == None)

                RESPONSE[distribution_name] = DISTRIBUTION_RESULTS

    sorted_results_sse = {distribution: results for distribution, results in sorted(RESPONSE.items(), key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
    aproved_results = {distribution: results for distribution, results in sorted_results_sse.items() if results["n_test_passed"] > 0}

    return sorted_results_sse, aproved_results

if __name__ == "__main__":
    path = "../../discrete/data/data_binomial.txt"
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    sorted_results_sse, aproved_results = phitter_discrete(data, confidence_level=0.975)

    for distribution, results in aproved_results.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
