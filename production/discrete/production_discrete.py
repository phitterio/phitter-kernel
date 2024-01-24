import collections
import joblib
import numpy
import scipy.optimize
import scipy.stats


class BERNOULLI:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.p = self.parameters["p"]

    def cdf(self, x: float) -> float:
        result = scipy.stats.bernoulli.cdf(x, self.p)
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
        result = scipy.stats.binom.pmf(x, self.n, self.p)
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
        result = scipy.stats.hypergeom.pmf(x, self.N, self.n, self.K)
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
            parametric_mode = numpy.floor((n + 1) * (K + 1) / (N + 2))
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
        result = -(self.p**x) / (numpy.log(1 - self.p) * x)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0 and self.p < 1
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            p = initial_solution
            parametric_mean = -p / ((1 - p) * numpy.log(1 - p))
            eq1 = parametric_mean - measurements.mean
            return eq1

        solution = scipy.optimize.least_squares(equations, 0.5, bounds=(0, 1), args=([measurements]))
        parameters = {"p": solution.x[0]}
        return parameters


class NEGATIVE_BINOMIAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.r = self.parameters["r"]
        self.p = self.parameters["p"]

    def cdf(self, x: float) -> float:
        result = scipy.stats.beta.cdf(self.p, self.r, x + 1)
        return result

    def pmf(self, x: int) -> float:
        result = scipy.stats.nbinom.pmf(x, self.r, self.p)
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
        parameters = {"r": r, "p": p}
        return parameters


class POISSON:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]

    def cdf(self, x: float) -> float:
        result = scipy.stats.poisson.cdf(x, self.lambda_)
        return result

    def pmf(self, x: int) -> float:
        result = scipy.stats.poisson.pmf(x, self.lambda_)
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
        self.data = numpy.sort(data)
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
        self.domain = numpy.fromiter(self.histogram.keys(), dtype=float)
        self.frequencies = numpy.fromiter(self.histogram.values(), dtype=float)
        self.frequencies_pmf = list(map(lambda x: x / self.length, self.histogram.values()))
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.length - 1]])
        self.Sn_ks = (numpy.arange(self.length) + 1) / self.length

    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def get_histogram(self) -> dict[int, float | int]:
        histogram = collections.Counter(self.data)
        return {k: v for k, v in sorted(histogram.items(), key=lambda item: item[0])}


def test_chi_square_discrete(distribution, measurements, confidence_level=0.95):
    N = measurements.length
    freedom_degrees = len(measurements.histogram.items()) - 1
    expected_values = numpy.ceil(N * (distribution.pmf(measurements.domain)))
    errors = ((measurements.frequencies - expected_values) ** 2) / expected_values
    statistic_chi2 = numpy.sum(errors)
    critical_value = scipy.stats.chi2.ppf(confidence_level, freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {"test_statistic": statistic_chi2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_chi2


def test_kolmogorov_smirnov_discrete(distribution, measurements, confidence_level=0.95):
    N = measurements.length
    Fn = distribution.cdf(measurements.data)
    errors = numpy.abs(measurements.Sn_ks[measurements.idx_ks] - Fn[measurements.idx_ks])
    statistic_ks = numpy.max(errors)
    critical_value = scipy.stats.kstwo.ppf(confidence_level, N)
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, N)
    rejected = statistic_ks >= critical_value
    result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ks


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
                self.distribution_results["n_test_passed"] = int(self.distribution_results["chi_square"]["rejected"] == False) + int(
                    self.distribution_results["kolmogorov_smirnov"]["rejected"] == False
                )
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
