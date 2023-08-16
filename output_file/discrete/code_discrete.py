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
        parameters = {"p": p, "n": n}
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
        λ = measurements.mean
        parameters = {"lambda": λ}
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
    def __init__(self, data):
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
        self.frequencies = self.get_frequencies()

    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def get_frequencies(self) -> dict[int, float | int]:
        frequencies = collections.Counter(self.data)
        return {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[0])}


def test_chi_square(data, distribution_class):
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = distribution_class(measurements)
    N = measurements.length
    frequencies = measurements.frequencies
    freedom_degrees = len(frequencies.items()) - 1
    errors = []
    for i, observed in frequencies.items():
        expected = math.ceil(N * (distribution.pmf(i)))
        errors.append(((observed - expected) ** 2) / expected)
    statistic_chi2 = sum(errors)
    critical_value = scipy.stats.chi2.ppf(0.95, freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {"test_statistic": statistic_chi2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_chi2


def test_kolmogorov_smirnov(data, distribution_class):
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = distribution_class(measurements)
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
    critical_value = scipy.stats.kstwo.ppf(0.95, N)
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, N)
    rejected = statistic_ks >= critical_value
    result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ks


def adinf(z):
    if z < 2:
        return (z**-0.5) * math.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
    return math.exp(-math.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))


def errfix(n, x):
    def g1(t):
        return math.sqrt(t) * (1 - t) * (49 * t - 102)

    def g2(t):
        return -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t

    def g3(t):
        return -130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * t) * t) * t) * t) * t

    c = 0.01265 + 0.1757 / n
    if x < c:
        return (0.0037 / (n**3) + 0.00078 / (n**2) + 0.00006 / n) * g1(x / c)
    elif x > c and x < 0.8:
        return (0.04213 / n + 0.01365 / (n**2)) * g2((x - c) / (0.8 - c))
    else:
        return (g3(x)) / n


def AD(n, z):
    return adinf(z) + errfix(n, adinf(z))


def ad_critical_value(q, n):
    def f(x):
        return AD(n, x) - q

    root = scipy.optimize.newton(f, 2)
    return root


def ad_p_value(n, z):
    return 1 - AD(n, z)


def test_anderson_darling(data, distribution_class):
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = distribution_class(measurements)
    N = measurements.length
    data.sort()
    S = 0
    for k in range(N):
        c1 = math.log(distribution.cdf(data[k]))
        c2 = math.log(1 - distribution.cdf(data[N - k - 1]))
        c3 = (2 * (k + 1) - 1) / N
        S += c3 * (c1 + c2)
    A2 = -N - S
    critical_value = ad_critical_value(0.95, N)
    p_value = ad_p_value(N, A2)
    rejected = A2 >= critical_value
    result_test_ad = {"test_statistic": A2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ad


if __name__ == "__main__":
    path = "../../discrete/data/data_binomial.txt"
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    _all_distributions = [BERNOULLI, BINOMIAL, GEOMETRIC, HYPERGEOMETRIC, LOGARITHMIC, NEGATIVE_BINOMIAL, POISSON, UNIFORM]
    measurements = MEASUREMENTS_DISCRETE(data)

    ## Calculae Histogram
    num_bins = measurements.num_bins
    frequencies, bin_edges = numpy.histogram(data, num_bins, density=True)
    central_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

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
            pdf_values = [distribution.pdf(c) for c in central_values]
            sse = numpy.sum(numpy.power(frequencies - pdf_values, 2.0))
        except:
            validate_estimation = False

        DISTRIBUTION_RESULTS = {}
        v1, v2, v3 = False, False, False
        if validate_estimation and not math.isnan(sse) and not math.isinf(sse):
            try:
                chi2_test = test_chi_square(data, distribution, measurements)
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
                ks_test = test_kolmogorov_smirnov(data, distribution, measurements)
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
            try:
                ad_test = test_anderson_darling(data, distribution, measurements)
                if numpy.isnan(ad_test["test_statistic"]) == False and math.isinf(ad_test["test_statistic"]) == False and ad_test["test_statistic"] > 0:
                    DISTRIBUTION_RESULTS["anderson_darling"] = {
                        "test_statistic": ad_test["test_statistic"],
                        "critical_value": ad_test["critical_value"],
                        "p_value": ad_test["p-value"],
                        "rejected": ad_test["rejected"],
                    }
                    v3 = True
                else:
                    DISTRIBUTION_RESULTS["anderson_darling"] = NONE_RESULTS
            except:
                DISTRIBUTION_RESULTS["anderson_darling"] = NONE_RESULTS

            if v1 or v2 or v3:
                DISTRIBUTION_RESULTS["sse"] = sse
                DISTRIBUTION_RESULTS["parameters"] = str(distribution.parameters)
                DISTRIBUTION_RESULTS["n_test_passed"] = (
                    int(DISTRIBUTION_RESULTS["chi_square"]["rejected"] == False)
                    + int(DISTRIBUTION_RESULTS["kolmogorov_smirnov"]["rejected"] == False)
                    + int(DISTRIBUTION_RESULTS["anderson_darling"]["rejected"] == False)
                )
                DISTRIBUTION_RESULTS["n_test_null"] = (
                    int(DISTRIBUTION_RESULTS["chi_square"]["rejected"] == None)
                    + int(DISTRIBUTION_RESULTS["kolmogorov_smirnov"]["rejected"] == None)
                    + int(DISTRIBUTION_RESULTS["anderson_darling"]["rejected"] == None)
                )

                RESPONSE[distribution_name] = DISTRIBUTION_RESULTS

    sorted_results_sse = {distribution: results for distribution, results in sorted(RESPONSE.items(), key=lambda x: x[1]["sse"])}
    aproved_results = {distribution: results for distribution, results in sorted_results_sse.items() if results["n_test_passed"] > 0}

    for distribution, results in aproved_results.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
