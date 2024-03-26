import concurrent.futures
import numpy
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats
import typing
import warnings

warnings.filterwarnings("ignore")


class ALPHA:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "alpha"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.stats.norm.cdf(self.alpha - (1 / z(x))) / scipy.stats.norm.cdf(self.alpha)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.alpha.pdf(x, self.alpha, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.scale / (self.alpha - scipy.stats.norm.ppf(u * scipy.stats.norm.cdf(self.alpha)))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        f = lambda x: x**k * scipy.stats.alpha.pdf(x, self.alpha, loc=0, scale=1)
        return scipy.integrate.quad(f, 0, 2)[0]

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale * self.scale * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / (µ2 - µ1**2) ** 1.5

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / (µ2 - µ1**2) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (self.scale * (numpy.sqrt(self.alpha * self.alpha + 8) - self.alpha)) / 4 + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.alpha.fit(continuous_measures.data)
        parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters


class ARCSINE:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "arcsine"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.a) / (self.b - self.a)
        return 2 * numpy.arcsin(numpy.sqrt(z(x))) / numpy.pi

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 / (numpy.pi * numpy.sqrt((x - self.a) * (self.b - x)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a + (self.b - self.a) * numpy.sin((u * numpy.pi) / 2) ** 2
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (scipy.special.gamma(0.5) * scipy.special.gamma(k + 0.5)) / (numpy.pi * scipy.special.gamma(k + 1))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1 * (self.b - self.a) + self.a

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (µ2 - µ1**2) * (self.b - self.a) ** 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-3
        b = continuous_measures.max + 1e-3
        parameters = {"a": a, "b": b}
        return parameters


class ARGUS:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.chi = self.parameters["chi"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "argus"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = 1 - scipy.special.gammainc(1.5, self.chi * self.chi * (1 - z(x) ** 2) / 2) / scipy.special.gammainc(1.5, self.chi * self.chi / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        result = (1 / self.scale) * ((self.chi**3) / (numpy.sqrt(2 * numpy.pi) * Ψ(self.chi))) * z(x) * numpy.sqrt(1 - z(x) * z(x)) * numpy.exp(-0.5 * self.chi**2 * (1 - z(x) * z(x)))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        y1 = (1 - u) * scipy.special.gammainc(1.5, (self.chi * self.chi) / 2)
        y2 = (2 * scipy.special.gammaincinv(1.5, y1)) / (self.chi * self.chi)
        result = self.loc + self.scale * numpy.sqrt(1 - y2)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.loc + self.scale * numpy.sqrt(numpy.pi / 8) * (
            (self.chi * numpy.exp((-self.chi * self.chi) / 4) * scipy.special.iv(1, (self.chi * self.chi) / 4)) / (scipy.stats.norm.cdf(self.chi) - self.chi * scipy.stats.norm.pdf(self.chi) - 0.5)
        )

    @property
    def variance(self) -> float:
        return (
            self.scale * self.scale * (1 - 3 / (self.chi * self.chi) + (self.chi * scipy.stats.norm.pdf(self.chi)) / (scipy.stats.norm.cdf(self.chi) - self.chi * scipy.stats.norm.pdf(self.chi) - 0.5))
            - (self.mean - self.loc) ** 2
        )

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return None

    @property
    def kurtosis(self) -> float:
        return None

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.scale * (1 / (numpy.sqrt(2) * self.chi)) * numpy.sqrt(self.chi * self.chi - 2 + numpy.sqrt(self.chi * self.chi * self.chi * self.chi + 4))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.chi > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.argus.fit(continuous_measures.data)
        parameters = {"chi": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters


class BETA:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]

    @property
    def name(self):
        return "beta"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.beta.cdf(x, self.alpha, self.beta, loc=self.A, scale=self.B - self.A)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.beta.pdf(x, self.alpha, self.beta, loc=self.A, scale=self.B - self.A)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.A + (self.B - self.A) * scipy.special.betaincinv(self.alpha, self.beta, u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.A + (self.alpha / (self.alpha + self.beta)) * (self.B - self.A)

    @property
    def variance(self) -> float:
        return ((self.alpha * self.beta) / ((self.alpha + self.beta + 1) * (self.alpha + self.beta) ** 2)) * (self.B - self.A) ** 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 2 * ((self.beta - self.alpha) / (self.alpha + self.beta + 2)) * numpy.sqrt((self.alpha + self.beta + 1) / (self.alpha * self.beta))

    @property
    def kurtosis(self) -> float:
        return 3 + (6 * ((self.alpha + self.beta + 1) * (self.alpha - self.beta) ** 2 - self.alpha * self.beta * (self.alpha + self.beta + 2))) / (
            self.alpha * self.beta * (self.alpha + self.beta + 2) * (self.alpha + self.beta + 3)
        )

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.A + ((self.alpha - 1) / (self.alpha + self.beta - 2)) * (self.B - self.A)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.A < self.B
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, A, B = initial_solution
            parametric_mean = A + (alpha / (alpha + beta)) * (B - A)
            parametric_variance = ((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))) * (B - A) ** 2
            parametric_skewness = 2 * ((beta - alpha) / (alpha + beta + 2)) * numpy.sqrt((alpha + beta + 1) / (alpha * beta))
            parametric_kurtosis = 3 * (((alpha + beta + 1) * (2 * (alpha + beta) ** 2 + (alpha * beta) * (alpha + beta - 6))) / ((alpha * beta) * (alpha + beta + 2) * (alpha + beta + 3)))
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        bnds = ((0, 0, -numpy.inf, continuous_measures.mean), (numpy.inf, numpy.inf, continuous_measures.mean, numpy.inf))
        x0 = (1, 1, continuous_measures.min, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "A": solution.x[2], "B": solution.x[3]}
        v1 = parameters["alpha"] > 0
        v2 = parameters["beta"] > 0
        v3 = parameters["A"] < parameters["B"]
        if (v1 and v2 and v3) == False:
            scipy_params = scipy.stats.beta.fit(continuous_measures.data)
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "A": scipy_params[2], "B": scipy_params[3]}
        return parameters


warnings.filterwarnings("ignore")


class BETA_PRIME:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "beta_prime"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.betainc(self.alpha, self.beta, x / (1 + x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (x ** (self.alpha - 1) * (1 + x) ** (-self.alpha - self.beta)) / (scipy.special.beta(self.alpha, self.beta))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.betaincinv(self.alpha, self.beta, u) / (1 - scipy.special.betaincinv(self.alpha, self.beta, u))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (scipy.special.gamma(k + self.alpha) * scipy.special.gamma(self.beta - k)) / (scipy.special.gamma(self.alpha) * scipy.special.gamma(self.beta))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (self.alpha - 1) / (self.beta + 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta = initial_solution
            parametric_mean = alpha / (beta - 1)
            parametric_variance = alpha * (alpha + beta - 1) / ((beta - 1) ** 2 * (beta - 2))
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        scipy_params = scipy.stats.betaprime.fit(continuous_measures.data)
        try:
            bnds = ((0, 0), (numpy.inf, numpy.inf))
            x0 = (scipy_params[0], scipy_params[1])
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
        except:
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1]}
        return parameters


warnings.filterwarnings("ignore")


class BETA_PRIME_4P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "beta_prime_4p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.betaprime.cdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.betaprime.pdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.betaprime.ppf(u, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (scipy.special.gamma(k + self.alpha) * scipy.special.gamma(self.beta - k)) / (scipy.special.gamma(self.alpha) * scipy.special.gamma(self.beta))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + (self.scale * (self.alpha - 1)) / (self.beta + 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.scale > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, scale, loc = initial_solution
            parametric_mean = scale * alpha / (beta - 1) + loc
            parametric_variance = (scale**2) * alpha * (alpha + beta - 1) / ((beta - 1) ** 2 * (beta - 2))
            parametric_median = loc + scale * scipy.stats.beta.ppf(0.5, alpha, beta) / (1 - scipy.stats.beta.ppf(0.5, alpha, beta))
            parametric_mode = scale * (alpha - 1) / (beta + 1) + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3, eq4)

        scipy_params = scipy.stats.betaprime.fit(continuous_measures.data)
        try:
            bnds = ((0, 0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            x0 = (continuous_measures.mean, continuous_measures.mean, scipy_params[3], continuous_measures.mean)
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1], "loc": solution.x[3], "scale": solution.x[2]}
        except:
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "loc": scipy_params[2], "scale": scipy_params[3]}
        return parameters


class BRADFORD:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.c = self.parameters["c"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "bradford"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.log(1 + self.c * (x - self.min) / (self.max - self.min)) / numpy.log(self.c + 1)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.c / ((self.c * (x - self.min) + self.max - self.min) * numpy.log(self.c + 1))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.min + ((numpy.exp(u * numpy.log(1 + self.c)) - 1) * (self.max - self.min)) / self.c
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (self.c * (self.max - self.min) + numpy.log(1 + self.c) * (self.min * (self.c + 1) - self.max)) / (numpy.log(1 + self.c) * self.c)

    @property
    def variance(self) -> float:
        return ((self.max - self.min) ** 2 * ((self.c + 2) * numpy.log(1 + self.c) - 2 * self.c)) / (2 * self.c * numpy.log(1 + self.c) ** 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.sqrt(2) * (12 * self.c * self.c - 9 * numpy.log(1 + self.c) * self.c * (self.c + 2) + 2 * numpy.log(1 + self.c) * numpy.log(1 + self.c) * (self.c * (self.c + 3) + 3))) / (
            numpy.sqrt(self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c))) * (3 * self.c * (numpy.log(1 + self.c) - 2) + 6 * numpy.log(1 + self.c))
        )

    @property
    def kurtosis(self) -> float:
        return (
            self.c**3 * (numpy.log(1 + self.c) - 3) * (numpy.log(1 + self.c) * (3 * numpy.log(1 + self.c) - 16) + 24)
            + 12 * numpy.log(1 + self.c) * self.c * self.c * (numpy.log(1 + self.c) - 4) * (numpy.log(1 + self.c) - 3)
            + 6 * self.c * numpy.log(1 + self.c) ** 2 * (3 * numpy.log(1 + self.c) - 14)
            + 12 * numpy.log(1 + self.c) ** 3
        ) / (3 * self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c)) ** 2) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.min

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.max > self.min
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        _min = continuous_measures.min - 1e-3
        _max = continuous_measures.max + 1e-3

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            c = initial_solution
            parametric_mean = (c * (_max - _min) + numpy.log(c + 1) * (_min * (c + 1) - _max)) / (c * numpy.log(c + 1))
            eq1 = parametric_mean - continuous_measures.mean
            return eq1

        solution = scipy.optimize.fsolve(equations, (1), continuous_measures)
        parameters = {"c": solution[0], "min": _min, "max": _max}
        return parameters


warnings.filterwarnings("ignore")


class BURR:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]

    @property
    def name(self):
        return "burr"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.cdf(x, self.B, self.C, scale=self.A)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.pdf(x, self.B, self.C, scale=self.A)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.A * ((1 - u) ** (-1 / self.C) - 1) ** (1 / self.B)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (
            self.A**k
            * self.C
            * ((scipy.special.gamma((self.B * self.C - k) / self.B) * scipy.special.gamma((self.B + k) / self.B)) / scipy.special.gamma((self.B * self.C - k) / self.B + (self.B + k) / self.B))
        )

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.A * ((self.B - 1) / (self.B * self.C + 1)) ** (1 / self.B)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.burr12.fit(continuous_measures.data)
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1]}
        return parameters


warnings.filterwarnings("ignore")


class BURR_4P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "burr_4p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.cdf(x, self.B, self.C, loc=self.loc, scale=self.A)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.pdf(x, self.B, self.C, loc=self.loc, scale=self.A)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.A * ((1 - u) ** (-1 / self.C) - 1) ** (1 / self.B) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (
            self.A**k
            * self.C
            * ((scipy.special.gamma((self.B * self.C - k) / self.B) * scipy.special.gamma((self.B + k) / self.B)) / scipy.special.gamma((self.B * self.C - k) / self.B + (self.B + k) / self.B))
        )

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.A * ((self.B - 1) / (self.B * self.C + 1)) ** (1 / self.B)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.burr12.fit(continuous_measures.data)
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1], "loc": scipy_params[2]}
        return parameters


class CAUCHY:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.x0 = self.parameters["x0"]
        self.gamma = self.parameters["gamma"]

    @property
    def name(self):
        return "cauchy"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 / numpy.pi) * numpy.arctan(((x - self.x0) / self.gamma)) + (1 / 2)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 / (numpy.pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.x0 + self.gamma * numpy.tan(numpy.pi * (u - 0.5))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return None

    @property
    def variance(self) -> float:
        return None

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return None

    @property
    def kurtosis(self) -> float:
        return None

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.x0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.gamma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.cauchy.fit(continuous_measures.data)
        parameters = {"x0": scipy_params[0], "gamma": scipy_params[1]}
        return parameters


class CHI_SQUARE:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.df = self.parameters["df"]

    @property
    def name(self):
        return "chi_square"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.df / 2, x / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.chi2.pdf(x, self.df)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = 2 * scipy.special.gammaincinv(self.df / 2, u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.df

    @property
    def variance(self) -> float:
        return self.df * 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return numpy.sqrt(8 / self.df)

    @property
    def kurtosis(self) -> float:
        return 12 / self.df + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.df - 2

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = type(self.df) == int
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        parameters = {"df": round(continuous_measures.mean)}
        return parameters


class CHI_SQUARE_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "chi_square_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.special.gammainc(self.df / 2, z(x) / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * (1 / (2 ** (self.df / 2) * scipy.special.gamma(self.df / 2))) * (z(x) ** ((self.df / 2) - 1)) * (numpy.exp(-z(x) / 2))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = 2 * self.scale * scipy.special.gammaincinv(self.df / 2, u) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.df * self.scale + self.loc

    @property
    def variance(self) -> float:
        return self.df * 2 * (self.scale * self.scale)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return numpy.sqrt(8 / self.df)

    @property
    def kurtosis(self) -> float:
        return 12 / self.df + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (self.df - 2) * self.scale + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = type(self.df) == int
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.chi2.fit(continuous_measures.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters


class DAGUM:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "dagum"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 + (x / self.b) ** (-self.a)) ** (-self.p)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.a * self.p / x) * (((x / self.b) ** (self.a * self.p)) / ((((x / self.b) ** (self.a)) + 1) ** (self.p + 1)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.b * (u ** (-1 / self.p) - 1) ** (-1 / self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (
            self.b**k
            * self.p
            * ((scipy.special.gamma((self.a * self.p + k) / self.a) * scipy.special.gamma((self.a - k) / self.a)) / scipy.special.gamma((self.a * self.p + k) / self.a + (self.a - k) / self.a))
        )

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.b * ((self.a * self.p - 1) / (self.a + 1)) ** (1 / self.a)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def sse(parameters: dict) -> float:
            def __pdf(x: float, params: dict) -> float:
                return (params["a"] * params["p"] / x) * (((x / params["b"]) ** (params["a"] * params["p"])) / ((((x / params["b"]) ** (params["a"])) + 1) ** (params["p"] + 1)))

            frequencies, bin_edges = numpy.histogram(continuous_measures.data, density=True)
            central_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            pdf_values = [__pdf(c, parameters) for c in central_values]
            sse = numpy.sum(numpy.power(frequencies - pdf_values, 2))
            return sse

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, p = initial_solution
            mu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = mu(1)
            parametric_variance = -(mu(1) ** 2) + mu(2)
            parametric_median = b * ((2 ** (1 / p)) - 1) ** (-1 / a)
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3)

        s0_burr3_sc = scipy.stats.burr.fit(continuous_measures.data)
        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1]}
        a0 = s0_burr3_sc[0]
        b0 = s0_burr3_sc[3]
        x0 = [a0, b0, 1]
        b = ((1e-5, 1e-5, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([continuous_measures]))
        parameters_ls = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2]}
        sse_sc = sse(parameters_sc)
        sse_ls = sse(parameters_ls)
        if a0 <= 2:
            return parameters_sc
        else:
            if sse_sc < sse_ls:
                return parameters_sc
            else:
                return parameters_ls


warnings.filterwarnings("ignore")


class DAGUM_4P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "dagum_4p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 + ((x - self.loc) / self.b) ** (-self.a)) ** (-self.p)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.a * self.p / x) * ((((x - self.loc) / self.b) ** (self.a * self.p)) / (((((x - self.loc) / self.b) ** (self.a)) + 1) ** (self.p + 1)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.b * (u ** (-1 / self.p) - 1) ** (-1 / self.a) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (
            self.b**k
            * self.p
            * ((scipy.special.gamma((self.a * self.p + k) / self.a) * scipy.special.gamma((self.a - k) / self.a)) / scipy.special.gamma((self.a * self.p + k) / self.a + (self.a - k) / self.a))
        )

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.b * ((self.a * self.p - 1) / (self.a + 1)) ** (1 / self.a)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def sse(parameters: dict) -> float:
            def __pdf(x: float, params: dict) -> float:
                return (params["a"] * params["p"] / (x - params["loc"])) * (
                    (((x - params["loc"]) / params["b"]) ** (params["a"] * params["p"])) / ((((x / params["b"]) ** (params["a"])) + 1) ** (params["p"] + 1))
                )

            frequencies, bin_edges = numpy.histogram(continuous_measures.data, density=True)
            central_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            pdf_values = [__pdf(c, parameters) for c in central_values]
            sse = numpy.sum(numpy.power(frequencies - pdf_values, 2))
            return sse

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, p, loc = initial_solution
            mu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = mu(1) + loc
            parametric_variance = -(mu(1) ** 2) + mu(2)
            parametric_median = b * ((2 ** (1 / p)) - 1) ** (-1 / a) + loc
            parametric_mode = b * ((a * p - 1) / (a + 1)) ** (1 / a) + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3, eq4)

        s0_burr3_sc = scipy.stats.burr.fit(continuous_measures.data)
        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1], "loc": s0_burr3_sc[2]}
        if s0_burr3_sc[0] <= 2:
            return parameters_sc
        else:
            a0 = s0_burr3_sc[0]
            x0 = [a0, 1, 1, continuous_measures.mean]
            b = ((1e-5, 1e-5, 1e-5, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([continuous_measures]))
            parameters_ls = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2], "loc": solution.x[3]}
            sse_sc = sse(parameters_sc)
            sse_ls = sse(parameters_ls)
            if sse_sc < sse_ls:
                return parameters_sc
            else:
                return parameters_ls


class ERLANG:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "erlang"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.k, x / self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.erlang.pdf(x, self.k, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.k, u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**self.k * (scipy.special.gamma(self.k + k) / scipy.special.factorial(k - 1))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.k - 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.k > 0
        v2 = self.beta > 0
        v3 = type(self.k) == int
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        k = round(continuous_measures.mean**2 / continuous_measures.variance)
        beta = continuous_measures.variance / continuous_measures.mean
        parameters = {"k": k, "beta": beta}
        return parameters


class ERLANG_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "erlang_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.k, (x - self.loc) / self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.erlang.pdf(x, self.k, scale=self.beta, loc=self.loc)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.k, u) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**self.k * (scipy.special.gamma(k + self.k) / scipy.special.factorial(k - 1))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.k - 1) + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.k > 0
        v2 = self.beta > 0
        v3 = type(self.k) == int
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        k = round((2 / continuous_measures.skewness) ** 2)
        beta = numpy.sqrt(continuous_measures.variance / ((2 / continuous_measures.skewness) ** 2))
        loc = continuous_measures.mean - ((2 / continuous_measures.skewness) ** 2) * beta
        parameters = {"k": k, "beta": beta, "loc": loc}
        return parameters


class ERROR_FUNCTION:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.h = self.parameters["h"]

    @property
    def name(self):
        return "error_function"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return scipy.stats.norm.cdf((2**0.5) * self.h * x)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.h * numpy.exp(-((self.h * x) ** 2)) / numpy.sqrt(numpy.pi)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.norm.ppf(u) / (self.h * numpy.sqrt(2))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return 0

    @property
    def variance(self) -> float:
        return 1 / (2 * self.h**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.h > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        h = numpy.sqrt(1 / (2 * continuous_measures.variance))
        parameters = {"h": h}
        return parameters


class EXPONENTIAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.lambda_ = self.parameters["lambda"]

    @property
    def name(self):
        return "exponential"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 - numpy.exp(-self.lambda_ * x)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.lambda_ * numpy.exp(-self.lambda_ * x)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = -numpy.log(1 - u) / self.lambda_
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return 1 / self.lambda_

    @property
    def variance(self) -> float:
        return 1 / (self.lambda_ * self.lambda_)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 2

    @property
    def kurtosis(self) -> float:
        return 9

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        lambda_ = 1 / continuous_measures.mean
        parameters = {"lambda": lambda_}
        return parameters


class EXPONENTIAL_2P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "exponential_2p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 - numpy.exp(-self.lambda_ * (x - self.loc))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.lambda_ * numpy.exp(-self.lambda_ * (x - self.loc))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc - numpy.log(1 - u) / self.lambda_
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return 1 / self.lambda_ + self.loc

    @property
    def variance(self) -> float:
        return 1 / (self.lambda_ * self.lambda_)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 2

    @property
    def kurtosis(self) -> float:
        return 9

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        lambda_ = (1 - numpy.log(2)) / (continuous_measures.mean - continuous_measures.median)
        loc = continuous_measures.min - 1e-4
        parameters = {"lambda": lambda_, "loc": loc}
        return parameters


class F:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]

    @property
    def name(self):
        return "f"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.cdf(x, self.df1, self.df2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.pdf(x, self.df1, self.df2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        t = scipy.special.betaincinv(self.df1 / 2, self.df2 / 2, u)
        result = (self.df2 * t) / (self.df1 * (1 - t))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.df2 / self.df1) ** k * (scipy.special.gamma(self.df1 / 2 + k) / scipy.special.gamma(self.df1 / 2)) * (scipy.special.gamma(self.df2 / 2 - k) / scipy.special.gamma(self.df2 / 2))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (self.df2 * (self.df1 - 2)) / (self.df1 * (self.df2 + 2))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.f.fit(continuous_measures.data)
        parameters = {"df1": scipy_params[0], "df2": scipy_params[1]}
        return parameters


class FATIGUE_LIFE:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.gamma = self.parameters["gamma"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "fatigue_life"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.sqrt((t - self.loc) / self.scale)
        result = scipy.stats.norm.cdf((z(x) - 1 / z(x)) / (self.gamma))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.sqrt((t - self.loc) / self.scale)
        result = (z(x) + 1 / z(x)) / (2 * self.gamma * (x - self.loc)) * scipy.stats.norm.pdf((z(x) - 1 / z(x)) / (self.gamma))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + (self.scale * (self.gamma * scipy.stats.norm.ppf(u) + numpy.sqrt((self.gamma * scipy.stats.norm.ppf(u)) ** 2 + 4)) ** 2) / 4
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.loc + self.scale * (1 + self.gamma**2 / 2)

    @property
    def variance(self) -> float:
        return self.scale**2 * self.gamma**2 * (1 + (5 * self.gamma**2) / 4)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (4 * self.gamma**2 * (11 * self.gamma**2 + 6)) / ((5 * self.gamma**2 + 4) * numpy.sqrt(self.gamma**2 * (5 * self.gamma**2 + 4)))

    @property
    def kurtosis(self) -> float:
        return 3 + (6 * self.gamma * self.gamma * (93 * self.gamma * self.gamma + 40)) / (5 * self.gamma**2 + 4) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        v2 = self.gamma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.fatiguelife.fit(continuous_measures.data)
        parameters = {"gamma": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters


class FOLDED_NORMAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "folded_normal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z1 = lambda t: (t + self.mu) / self.sigma
        z2 = lambda t: (t - self.mu) / self.sigma
        result = 0.5 * (scipy.special.erf(z1(x) / numpy.sqrt(2)) + scipy.special.erf(z2(x) / numpy.sqrt(2)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(2 / (numpy.pi * self.sigma**2)) * numpy.exp(-(x**2 + self.mu**2) / (2 * self.sigma**2)) * numpy.cosh(self.mu * x / (self.sigma**2))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.foldnorm.ppf(u, self.mu / self.sigma, loc=0, scale=self.sigma)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        f = lambda x: x**k * self.pdf(x)
        return scipy.integrate.quad(f, 0, self.mu + 4 * self.sigma)[0]

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        return self.sigma * numpy.sqrt(2 / numpy.pi) * numpy.exp((-self.mu * self.mu) / (2 * self.sigma * self.sigma)) - self.mu * (2 * scipy.stats.norm.cdf(-self.mu / self.sigma) - 1)

    @property
    def variance(self) -> float:
        return self.mu * self.mu + self.sigma * self.sigma - self.mean**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = sigma * numpy.sqrt(2 / numpy.pi) * numpy.exp(-(mu**2) / (2 * sigma**2)) + mu * scipy.special.erf(mu / numpy.sqrt(2 * sigma**2))
            parametric_variance = mu**2 + sigma**2 - parametric_mean**2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        x0 = [continuous_measures.mean, continuous_measures.standard_deviation]
        b = ((-numpy.inf, 0), (numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([continuous_measures]))
        parameters = {"mu": solution.x[0], "sigma": solution.x[1]}
        return parameters


class FRECHET:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "frechet"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * self.alpha * z(x) ** (-self.alpha - 1) * numpy.exp(-z(x) ** -self.alpha)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.alpha / self.scale) * (((x - self.loc) / self.scale) ** (-1 - self.alpha)) * numpy.exp(-(((x - self.loc) / self.scale) ** (-self.alpha)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.scale * (-numpy.log(u)) ** (-1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return scipy.special.gamma(1 - k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.scale * (self.alpha / (self.alpha + 1)) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.invweibull.fit(continuous_measures.data)
        parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters


class F_4P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "f_4p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.cdf(x, self.df1, self.df2, self.loc, self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.pdf(x, self.df1, self.df2, self.loc, self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        t = scipy.special.betaincinv(self.df1 / 2, self.df2 / 2, u)
        result = self.loc + (self.scale * (self.df2 * t)) / (self.df1 * (1 - t))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.df2 / self.df1) ** k * (scipy.special.gamma(self.df1 / 2 + k) / scipy.special.gamma(self.df1 / 2)) * (scipy.special.gamma(self.df2 / 2 - k) / scipy.special.gamma(self.df2 / 2))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return ((self.df2 * (self.df1 - 2)) / (self.df1 * (self.df2 + 2))) * self.scale + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        v3 = self.scale > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            df1, df2, loc, scale = initial_solution
            E = lambda k: (df2 / df1) ** k * (scipy.special.gamma(df1 / 2 + k) * scipy.special.gamma(df2 / 2 - k)) / (scipy.special.gamma(df1 / 2) * scipy.special.gamma(df2 / 2))
            parametric_mean = E(1) * scale + loc
            parametric_variance = (E(2) - E(1) ** 2) * (scale) ** 2
            parametric_median = scipy.stats.f.ppf(0.5, df1, df2) * scale + loc
            parametric_mode = ((df2 * (df1 - 2)) / (df1 * (df2 + 2))) * scale + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3, eq4)

        try:
            bnds = ((0, 0, -numpy.inf, 0), (numpy.inf, numpy.inf, continuous_measures.min, numpy.inf))
            x0 = (1, continuous_measures.standard_deviation, continuous_measures.min, continuous_measures.standard_deviation)
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"df1": solution.x[0], "df2": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        except:
            scipy_params = scipy.stats.f.fit(continuous_measures.data)
            parameters = {"df1": scipy_params[0], "df2": scipy_params[1], "loc": scipy_params[2], "scale": scipy_params[3]}
        return parameters


class GAMMA:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "gamma"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.alpha, x / self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gamma.pdf(x, self.alpha, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.alpha, u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * (scipy.special.gamma(k + self.alpha) / scipy.special.gamma(self.alpha))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.alpha - 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        alpha = continuous_measures.mean**2 / continuous_measures.variance
        beta = continuous_measures.variance / continuous_measures.mean
        parameters = {"alpha": alpha, "beta": beta}
        return parameters


class GAMMA_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "gamma_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.alpha, (x - self.loc) / self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gamma.pdf(x, self.alpha, loc=self.loc, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.alpha, u) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * (scipy.special.gamma(k + self.alpha) / scipy.special.factorial(self.alpha - 1))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.alpha - 1) + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        alpha = (2 / continuous_measures.skewness) ** 2
        beta = numpy.sqrt(continuous_measures.variance / alpha)
        loc = continuous_measures.mean - alpha * beta
        parameters = {"alpha": alpha, "loc": loc, "beta": beta}
        return parameters


class GENERALIZED_EXTREME_VALUE:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.xi = self.parameters["xi"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "generalized_extreme_value"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return numpy.exp(-numpy.exp(-z(x)))
        else:
            return numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi)))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))
        else:
            return (1 / self.sigma) * numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi))) * (1 + self.xi * z(x)) ** (-1 - 1 / self.xi)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        if self.xi == 0:
            result = self.mu - self.sigma * numpy.log(-numpy.log(u))
        else:
            result = self.mu + (self.sigma * ((-numpy.log(u)) ** -self.xi - 1)) / self.xi
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return scipy.special.gamma(1 - self.xi * k)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        if self.xi == 0:
            return self.mu + self.sigma * 0.5772156649
        µ1 = self.non_central_moments(1)
        return self.mu + (self.sigma * (µ1 - 1)) / self.xi

    @property
    def variance(self) -> float:
        if self.xi == 0:
            return self.sigma**2 * (numpy.pi**2 / 6)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.sigma**2 * (µ2 - µ1**2)) / self.xi**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        if self.xi == 0:
            return (12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        if self.xi == 0:
            return 5.4
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.xi == 0:
            return self.mu
        return self.mu + (self.sigma * ((1 + self.xi) ** -self.xi - 1)) / self.xi

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.genextreme.fit(continuous_measures.data)
        parameters = {"xi": -scipy_params[0], "mu": scipy_params[1], "sigma": scipy_params[2]}
        return parameters


warnings.filterwarnings("ignore")


class GENERALIZED_GAMMA:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "generalized_gamma"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.d / self.p, (x / self.a) ** self.p)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.p / (self.a**self.d)) * (x ** (self.d - 1)) * numpy.exp(-((x / self.a) ** self.p)) / scipy.special.gamma(self.d / self.p)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a * scipy.special.gammaincinv(self.d / self.p, u) ** (1 / self.p)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.a**k * scipy.special.gamma((self.d + k) / self.p)) / scipy.special.gamma(self.d / self.p)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.a * ((self.d - 1) / self.p) ** (1 / self.p)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, d, p = initial_solution
            E = lambda r: a**r * (scipy.special.gamma((d + r) / p) / scipy.special.gamma(d / p))
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        try:
            solution = scipy.optimize.fsolve(equations, (1, 1, 1), continuous_measures)
            if all(x > 0 for x in solution) is False or all(x == 1 for x in solution) is True:
                bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
                x0 = (1, 1, 1)
                args = [continuous_measures]
                response = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
                solution = response.x
            parameters = {"a": solution[0], "d": solution[1], "p": solution[2]}
        except:
            scipy_params = scipy.stats.gengamma.fit(continuous_measures.data)
            parameters = {"a": scipy_params[0], "c": scipy_params[1], "mu": scipy_params[2]}
        return parameters


class GENERALIZED_GAMMA_4P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "generalized_gamma_4p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.d / self.p, ((x - self.loc) / self.a) ** self.p)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.p / (self.a**self.d)) * ((x - self.loc) ** (self.d - 1)) * numpy.exp(-(((x - self.loc) / self.a) ** self.p)) / scipy.special.gamma(self.d / self.p)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a * scipy.special.gammaincinv(self.d / self.p, u) ** (1 / self.p)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.a**k * scipy.special.gamma((self.d + k) / self.p)) / scipy.special.gamma(self.d / self.p)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.a * ((self.d - 1) / self.p) ** (1 / self.p)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, d, p, loc = initial_solution
            E = lambda r: a**r * (scipy.special.gamma((d + r) / p) / scipy.special.gamma(d / p))
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            parametric_median = a * scipy.stats.gamma.ppf(0.5, a=d / p, scale=1) ** (1 / p) + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), continuous_measures)
        if all(x > 0 for x in solution) is False or all(x == 1 for x in solution) is True:
            try:
                bnds = ((0, 0, 0, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
                if continuous_measures.mean < 0:
                    bnds = ((0, 0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, 0))
                x0 = (1, 1, 1, continuous_measures.mean)
                args = [continuous_measures]
                response = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
                solution = response.x
            except:
                scipy_params = scipy.stats.gengamma.fit(continuous_measures.data)
                solution = [scipy_params[3], scipy_params[0], scipy_params[1], scipy_params[2]]
        parameters = {"a": solution[0], "d": solution[1], "p": solution[2], "loc": solution[3]}
        return parameters


class GENERALIZED_LOGISTIC:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "generalized_logistic"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        return 1 / ((1 + numpy.exp(-z(x))) ** self.c)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        return (self.c / self.scale) * numpy.exp(-z(x)) * ((1 + numpy.exp(-z(x))) ** (-self.c - 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.scale * -numpy.log(u ** (-1 / self.c) - 1)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.loc + self.scale * (0.57721 + scipy.special.digamma(self.c))

    @property
    def variance(self) -> float:
        return self.scale * self.scale * ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c))

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2.40411380631918 + scipy.special.polygamma(2, self.c)) / ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c)) ** 1.5

    @property
    def kurtosis(self) -> float:
        return 3 + (6.49393940226682 + scipy.special.polygamma(3, self.c)) / ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c)) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.scale * numpy.log(self.c)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        v2 = self.c > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            c, loc, scale = initial_solution
            parametric_mean = loc + scale * (0.57721 + scipy.special.digamma(c))
            parametric_variance = scale**2 * (numpy.pi**2 / 6 + scipy.special.polygamma(1, c))
            parametric_median = loc + scale * (-numpy.log(0.5 ** (-1 / c) - 1))
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3)

        x0 = [continuous_measures.mean, continuous_measures.mean, continuous_measures.mean]
        b = ((1e-5, -numpy.inf, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([continuous_measures]))
        parameters = {"c": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}
        return parameters


class GENERALIZED_NORMAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.beta = self.parameters["beta"]
        self.mu = self.parameters["mu"]
        self.alpha = self.parameters["alpha"]

    @property
    def name(self):
        return "generalized_normal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 0.5 + (numpy.sign(x - self.mu) / 2) * scipy.special.gammainc(1 / self.beta, abs((x - self.mu) / self.alpha) ** self.beta)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.beta / (2 * self.alpha * scipy.special.gamma(1 / self.beta)) * numpy.exp(-((abs(x - self.mu) / self.alpha) ** self.beta))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + numpy.sign(u - 0.5) * (self.alpha**self.beta * scipy.special.gammaincinv(1 / self.beta, 2 * numpy.abs(u - 0.5))) ** (1 / self.beta)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return (self.mu**2 * scipy.special.gamma(3 / self.alpha)) / scipy.special.gamma(1 / self.alpha)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return (scipy.special.gamma(5 / self.alpha) * scipy.special.gamma(1 / self.alpha)) / scipy.special.gamma(3 / self.alpha) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.gennorm.fit(continuous_measures.data)
        parameters = {"beta": scipy_params[0], "mu": scipy_params[1], "alpha": scipy_params[2]}
        return parameters


class GENERALIZED_PARETO:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "generalized_pareto"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = 1 - (1 + self.c * z(x)) ** (-1 / self.c)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = (1 / self.sigma) * (1 + self.c * z(x)) ** (-1 / self.c - 1)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + (self.sigma * ((1 - u) ** -self.c - 1)) / self.c
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu + self.sigma / (1 - self.c)

    @property
    def variance(self) -> float:
        return (self.sigma * self.sigma) / ((1 - self.c) * (1 - self.c) * (1 - 2 * self.c))

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 * (1 + self.c) * numpy.sqrt(1 - 2 * self.c)) / (1 - 3 * self.c)

    @property
    def kurtosis(self) -> float:
        return (3 * (1 - 2 * self.c) * (2 * self.c * self.c + self.c + 3)) / ((1 - 3 * self.c) * (1 - 4 * self.c))

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            c, mu, sigma = initial_solution
            parametric_mean = mu + sigma / (1 - c)
            parametric_variance = sigma * sigma / ((1 - c) * (1 - c) * (1 - 2 * c))
            parametric_median = mu + sigma * (2**c - 1) / c
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - numpy.percentile(continuous_measures.data, 50)
            return (eq1, eq2, eq3)

        scipy_params = scipy.stats.genpareto.fit(continuous_measures.data)
        parameters = {"c": scipy_params[0], "mu": scipy_params[1], "sigma": scipy_params[2]}
        if parameters["c"] < 0:
            scipy_params = scipy.stats.genpareto.fit(continuous_measures.data)
            c0 = scipy_params[0]
            x0 = [c0, continuous_measures.min, 1]
            b = ((-numpy.inf, -numpy.inf, 0), (numpy.inf, continuous_measures.min, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([continuous_measures]))
            parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
            parameters["mu"] = min(parameters["mu"], continuous_measures.min - 1e-3)
            delta_sigma = parameters["c"] * (parameters["mu"] - continuous_measures.max) - parameters["sigma"]
            parameters["sigma"] = parameters["sigma"] + delta_sigma + 1e-8
        return parameters


class GIBRAT:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "gibrat"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gibrat.cdf(x, self.loc, self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gibrat.pdf(x, self.loc, self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.exp(scipy.stats.norm.ppf(u)) * self.scale + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.loc + self.scale * numpy.sqrt(numpy.exp(1))

    @property
    def variance(self) -> float:
        return numpy.exp(1) * (numpy.exp(1) - 1) * self.scale * self.scale

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 + numpy.exp(1)) * numpy.sqrt(numpy.exp(1) - 1)

    @property
    def kurtosis(self) -> float:
        return numpy.exp(1) ** 4 + 2 * numpy.exp(1) ** 3 + 3 * numpy.exp(1) ** 2 - 6

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (1 / numpy.exp(1)) * self.scale + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.gibrat.fit(continuous_measures.data)
        parameters = {"loc": scipy_params[0], "scale": scipy_params[1]}
        return parameters


class GUMBEL_LEFT:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "gumbel_left"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return 1 - numpy.exp(-numpy.exp(z(x)))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return (1 / self.sigma) * numpy.exp(z(x) - numpy.exp(z(x)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * numpy.log(-numpy.log(1 - u))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu - 0.5772156649 * self.sigma

    @property
    def variance(self) -> float:
        return self.sigma**2 * (numpy.pi**2 / 6)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (-12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3

    @property
    def kurtosis(self) -> float:
        return 3 + 12 / 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = mu - sigma * 0.5772156649
            parametric_variance = (sigma**2) * (numpy.pi**2) / 6
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        solution = scipy.optimize.fsolve(equations, (1, 1), continuous_measures)
        parameters = {"mu": solution[0], "sigma": solution[1]}
        return parameters


class GUMBEL_RIGHT:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "gumbel_right"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return numpy.exp(-numpy.exp(-z(x)))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu - self.sigma * numpy.log(-numpy.log(u))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu + 0.5772156649 * self.sigma

    @property
    def variance(self) -> float:
        return self.sigma**2 * (numpy.pi**2 / 6)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3

    @property
    def kurtosis(self) -> float:
        return 3 + 12 / 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = mu + sigma * 0.5772156649
            parametric_variance = (sigma**2) * (numpy.pi**2) / 6
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        solution = scipy.optimize.fsolve(equations, (1, 1), continuous_measures)
        parameters = {"mu": solution[0], "sigma": solution[1]}
        return parameters


class HALF_NORMAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "half_normal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = scipy.special.erf(z(x) / numpy.sqrt(2))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = (1 / self.sigma) * numpy.sqrt(2 / numpy.pi) * numpy.exp(-(z(x) ** 2) / 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.norm.ppf((1 + u) / 2) * self.sigma + self.mu
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu + self.sigma * numpy.sqrt(2 / numpy.pi)

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma * (1 - 2 / numpy.pi)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.sqrt(2) * (4 - numpy.pi)) / (numpy.pi - 2) ** 1.5

    @property
    def kurtosis(self) -> float:
        return 3 + (8 * (numpy.pi - 3)) / (numpy.pi - 2) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        sigma = numpy.sqrt(continuous_measures.variance / (1 - 2 / numpy.pi))
        mu = continuous_measures.mean - sigma * numpy.sqrt(2) / numpy.sqrt(numpy.pi)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class HYPERBOLIC_SECANT:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "hyperbolic_secant"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (2 / numpy.pi) * numpy.arctan(numpy.exp((z(x))))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (1 / numpy.cosh(z(x))) / (2 * self.sigma)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.log(numpy.tan((u * numpy.pi) / 2)) * ((2 * self.sigma) / numpy.pi) + self.mu
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        sigma = numpy.sqrt(continuous_measures.variance)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class INVERSE_GAMMA:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "inverse_gamma"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.cdf(x, a=self.alpha, scale=self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.pdf(x, a=self.alpha, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta / scipy.special.gammaincinv(self.alpha, 1 - u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return self.beta**k / (self.alpha - 1)
        if k == 2:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2))
        if k == 3:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3))
        if k == 4:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3) * (self.alpha - 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta / (self.alpha + 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.invgamma.fit(continuous_measures.data)
        parameters = {"alpha": scipy_params[0], "beta": scipy_params[2]}
        return parameters


class INVERSE_GAMMA_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "inverse_gamma_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.cdf(x, a=self.alpha, loc=self.loc, scale=self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.pdf(x, a=self.alpha, loc=self.loc, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.beta / scipy.special.gammaincinv(self.alpha, 1 - u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return self.beta**k / (self.alpha - 1)
        if k == 2:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2))
        if k == 3:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3))
        if k == 4:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3) * (self.alpha - 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta / (self.alpha + 1) + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, loc = initial_solution
            E = lambda k: (beta**k) / numpy.prod(numpy.array([(alpha - i) for i in range(1, k + 1)]))
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        try:
            bnds = ((0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
            x0 = (2, 1, continuous_measures.mean)
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1], "loc": solution.x[2]}
        except:
            scipy_params = scipy.stats.invgamma.fit(continuous_measures.data)
            parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "beta": scipy_params[2]}
        return parameters


class INVERSE_GAUSSIAN:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.lambda_ = self.parameters["lambda"]

    @property
    def name(self):
        return "inverse_gaussian"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.cdf(x, self.mu / self.lambda_, scale=self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.pdf(x, self.mu / self.lambda_, scale=self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.ppf(u, self.mu / self.lambda_, scale=self.lambda_)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.mu**3 / self.lambda_

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 3 * numpy.sqrt(self.mu / self.lambda_)

    @property
    def kurtosis(self) -> float:
        return 15 * (self.mu / self.lambda_) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu * (numpy.sqrt(1 + (9 * self.mu * self.mu) / (4 * self.lambda_ * self.lambda_)) - (3 * self.mu) / (2 * self.lambda_))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        lambda_ = mu**3 / continuous_measures.variance
        parameters = {"mu": mu, "lambda": lambda_}
        return parameters


class INVERSE_GAUSSIAN_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "inverse_gaussian_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.cdf(x, self.mu / self.lambda_, loc=self.loc, scale=self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.pdf(x, self.mu / self.lambda_, loc=self.loc, scale=self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.ppf(u, self.mu / self.lambda_, loc=self.loc, scale=self.lambda_)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu + self.loc

    @property
    def variance(self) -> float:
        return self.mu**3 / self.lambda_

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 3 * numpy.sqrt(self.mu / self.lambda_)

    @property
    def kurtosis(self) -> float:
        return 15 * (self.mu / self.lambda_) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.mu * (numpy.sqrt(1 + (9 * self.mu * self.mu) / (4 * self.lambda_ * self.lambda_)) - (3 * self.mu) / (2 * self.lambda_))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = 3 * numpy.sqrt(continuous_measures.variance / (continuous_measures.skewness**2))
        lambda_ = mu**3 / continuous_measures.variance
        loc = continuous_measures.mean - mu
        parameters = {"mu": mu, "lambda": lambda_, "loc": loc}
        return parameters


class JOHNSON_SB:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]

    @property
    def name(self):
        return "johnson_sb"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * numpy.log(z(x) / (1 - z(x))))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * numpy.sqrt(2 * numpy.pi) * z(x) * (1 - z(x)))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.log(z(x) / (1 - z(x)))) ** 2)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (self.lambda_ * numpy.exp((scipy.stats.norm.ppf(u) - self.gamma_) / self.delta_)) / (1 + numpy.exp((scipy.stats.norm.ppf(u) - self.gamma_) / self.delta_)) + self.xi_
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        f = lambda x: x**k * (self.delta_ / (numpy.sqrt(2 * numpy.pi) * x * (1 - x))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.log(x / (1 - x))) ** 2)
        return scipy.integrate.quad(f, 0, 1)[0]

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.xi_ + self.lambda_ * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.lambda_ * self.lambda_ * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / (µ2 - µ1**2) ** 1.5

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / (µ2 - µ1**2) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.johnsonsb.fit(continuous_measures.data)
        parameters = {"xi": scipy_params[2], "lambda": scipy_params[3], "gamma": scipy_params[0], "delta": scipy_params[1]}
        return parameters


class JOHNSON_SU:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]

    @property
    def name(self):
        return "johnson_su"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * numpy.arcsinh(z(x)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * numpy.sqrt(2 * numpy.pi) * numpy.sqrt(z(x) ** 2 + 1))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.arcsinh(z(x))) ** 2)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.lambda_ * numpy.sinh((scipy.stats.norm.ppf(u) - self.gamma_) / self.delta_) + self.xi_
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.xi_ - self.lambda_ * numpy.exp(self.delta_**-2 / 2) * numpy.sinh(self.gamma_ / self.delta_)

    @property
    def variance(self) -> float:
        return (self.lambda_**2 / 2) * (numpy.exp(self.delta_**-2) - 1) * (numpy.exp(self.delta_**-2) * numpy.cosh((2 * self.gamma_) / self.delta_) + 1)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return -(
            self.lambda_**3
            * numpy.sqrt(numpy.exp(self.delta_**-2))
            * (numpy.exp(self.delta_**-2) - 1) ** 2
            * (numpy.exp(self.delta_**-2) * (numpy.exp(self.delta_**-2) + 2) * numpy.sinh(3 * (self.gamma_ / self.delta_)) + 3 * numpy.sinh(self.gamma_ / self.delta_))
        ) / (4 * self.standard_deviation**3)

    @property
    def kurtosis(self) -> float:
        return (
            self.lambda_**4
            * (numpy.exp(self.delta_**-2) - 1) ** 2
            * (
                numpy.exp(self.delta_**-2) ** 2
                * (numpy.exp(self.delta_**-2) ** 4 + 2 * numpy.exp(self.delta_**-2) ** 3 + 3 * numpy.exp(self.delta_**-2) ** 2 - 3)
                * numpy.cosh(4 * (self.gamma_ / self.delta_))
                + 4 * numpy.exp(self.delta_**-2) ** 2 * (numpy.exp(self.delta_**-2) + 2) * numpy.cosh(2 * (self.gamma_ / self.delta_))
                + 3 * (2 * numpy.exp(self.delta_**-2) + 1)
            )
        ) / (8 * self.standard_deviation**4)

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            xi_, lambda_, gamma_, delta_ = initial_solution
            w = numpy.exp(1 / delta_**2)
            omega = gamma_ / delta_
            A = w**2 * (w**4 + 2 * w**3 + 3 * w**2 - 3) * numpy.cosh(4 * omega)
            B = 4 * w**2 * (w + 2) * numpy.cosh(2 * omega)
            C = 3 * (2 * w + 1)
            parametric_mean = xi_ - lambda_ * numpy.sqrt(w) * numpy.sinh(omega)
            parametric_variance = (lambda_**2 / 2) * (w - 1) * (w * numpy.cosh(2 * omega) + 1)
            parametric_kurtosis = ((lambda_**4) * (w - 1) ** 2 * (A + B + C)) / (8 * numpy.sqrt(parametric_variance) ** 4)
            parametric_median = xi_ + lambda_ * numpy.sinh(-omega)
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_kurtosis - continuous_measures.kurtosis
            eq4 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3, eq4)

        solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), continuous_measures)
        parameters = {"xi": solution[0], "lambda": solution[1], "gamma": solution[2], "delta": solution[3]}
        return parameters


class KUMARASWAMY:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "kumaraswamy"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        result = 1 - (1 - z(x) ** self.alpha) ** self.beta
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (self.alpha * self.beta) * (z(x) ** (self.alpha - 1)) * ((1 - z(x) ** self.alpha) ** (self.beta - 1)) / (self.max - self.min)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (1 - (1 - u) ** (1 / self.beta)) ** (1 / self.alpha) * (self.max - self.min) + self.min
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.beta * scipy.special.gamma(1 + k / self.alpha) * scipy.special.gamma(self.beta)) / scipy.special.gamma(1 + self.beta + k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.min + (self.max - self.min) * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.max - self.min) ** 2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.min + (self.max - self.min) * ((self.alpha - 1) / (self.alpha * self.beta - 1)) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.min < self.max
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha_, beta_, min_, max_ = initial_solution
            E = lambda r: beta_ * scipy.special.gamma(1 + r / alpha_) * scipy.special.gamma(beta_) / scipy.special.gamma(1 + beta_ + r / alpha_)
            parametric_mean = E(1) * (max_ - min_) + min_
            parametric_variance = (E(2) - E(1) ** 2) * (max_ - min_) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            parametric_median = ((1 - 2 ** (-1 / beta_)) ** (1 / alpha_)) * (max_ - min_) + min_
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        l = continuous_measures.min - 3 * abs(continuous_measures.min)
        bnds = ((0, 0, l, l), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1, 1)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "min": solution.x[2], "max": solution.x[3]}
        return parameters


class LAPLACE:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "laplace"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 0.5 + 0.5 * numpy.sign(x - self.mu) * (1 - numpy.exp(-abs(x - self.mu) / self.b))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 / (2 * self.b)) * numpy.exp(-abs(x - self.mu) / self.b)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu - self.b * numpy.sign(u - 0.5) * numpy.log(1 - 2 * numpy.abs(u - 0.5))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return 2 * self.b**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 6

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        b = numpy.sqrt(continuous_measures.variance / 2)
        parameters = {"mu": mu, "b": b}
        return parameters


class LEVY:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "levy"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        y = lambda x: numpy.sqrt(self.c / ((x - self.mu)))
        result = 2 - 2 * scipy.stats.norm.cdf(y(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(self.c / (2 * numpy.pi)) * numpy.exp(-self.c / (2 * (x - self.mu))) / ((x - self.mu) ** 1.5)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.c / scipy.stats.norm.ppf((2 - u) / 2) ** 2
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return numpy.inf

    @property
    def variance(self) -> float:
        return numpy.inf

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return None

    @property
    def kurtosis(self) -> float:
        return None

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu + self.c / 3

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.levy.fit(continuous_measures.data)
        parameters = {"mu": scipy_params[0], "c": scipy_params[1]}
        return parameters


class LOGGAMMA:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "loggamma"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        y = lambda x: (x - self.mu) / self.sigma
        result = scipy.special.gammainc(self.c, numpy.exp(y(x)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        y = lambda x: (x - self.mu) / self.sigma
        result = numpy.exp(self.c * y(x) - numpy.exp(y(x)) - scipy.special.gammaln(self.c)) / self.sigma
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * numpy.log(scipy.special.gammaincinv(self.c, u))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return scipy.special.digamma(self.c) * self.sigma + self.mu

    @property
    def variance(self) -> float:
        return scipy.special.polygamma(1, self.c) * self.sigma * self.sigma

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return scipy.special.polygamma(2, self.c) / scipy.special.polygamma(1, self.c) ** 1.5

    @property
    def kurtosis(self) -> float:
        return scipy.special.polygamma(3, self.c) / scipy.special.polygamma(1, self.c) ** 2 + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu + self.sigma * numpy.log(self.c)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution, data_mean, data_variance, data_skewness):
            c, mu, sigma = initial_solution
            parametric_mean = scipy.special.digamma(c) * sigma + mu
            parametric_variance = scipy.special.polygamma(1, c) * (sigma**2)
            parametric_skewness = scipy.special.polygamma(2, c) / (scipy.special.polygamma(1, c) ** 1.5)
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            eq3 = parametric_skewness - data_skewness
            return (eq1, eq2, eq3)

        bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1)
        args = (continuous_measures.mean, continuous_measures.variance, continuous_measures.skewness)
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
        return parameters


class LOGISTIC:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "logistic"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.exp(-(t - self.mu) / self.sigma)
        result = 1 / (1 + z(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.exp(-(t - self.mu) / self.sigma)
        result = z(x) / (self.sigma * (1 + z(x)) ** 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * numpy.log(u / (1 - u))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return (self.sigma * self.sigma * numpy.pi * numpy.pi) / 3

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 4.2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        sigma = numpy.sqrt(3 * continuous_measures.variance / (numpy.pi**2))
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class LOGLOGISTIC:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "loglogistic"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = x**self.beta / (self.alpha**self.beta + x**self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta / self.alpha * (x / self.alpha) ** (self.beta - 1) / ((1 + (x / self.alpha) ** self.beta) ** 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.alpha * (u / (1 - u)) ** (1 / self.beta)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.alpha**k * ((k * numpy.pi) / self.beta)) / numpy.sin((k * numpy.pi) / self.beta)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.alpha * ((self.beta - 1) / (self.beta + 1)) ** (1 / self.beta)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.fisk.fit(continuous_measures.data)
        parameters = {"alpha": scipy_params[2], "beta": scipy_params[0]}
        return parameters


class LOGLOGISTIC_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "loglogistic_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (x - self.loc) ** self.beta / (self.alpha**self.beta + (x - self.loc) ** self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta / self.alpha * ((x - self.loc) / self.alpha) ** (self.beta - 1) / ((1 + ((x - self.loc) / self.alpha) ** self.beta) ** 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.alpha * (u / (1 - u)) ** (1 / self.beta) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.alpha**k * ((k * numpy.pi) / self.beta)) / numpy.sin((k * numpy.pi) / self.beta)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.alpha * ((self.beta - 1) / (self.beta + 1)) ** (1 / self.beta)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.fisk.fit(continuous_measures.data)
        parameters = {"loc": scipy_params[1], "alpha": scipy_params[2], "beta": scipy_params[0]}
        return parameters


class LOGNORMAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "lognormal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.norm.cdf((numpy.log(x) - self.mu) / self.sigma)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 / (x * self.sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((numpy.log(x) - self.mu) ** 2) / (2 * self.sigma**2)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.exp(self.mu + self.sigma * scipy.stats.norm.ppf(u))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return numpy.exp(self.mu + self.sigma**2 / 2)

    @property
    def variance(self) -> float:
        return (numpy.exp(self.sigma**2) - 1) * numpy.exp(2 * self.mu + self.sigma**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.exp(self.sigma * self.sigma) + 2) * numpy.sqrt(numpy.exp(self.sigma * self.sigma) - 1)

    @property
    def kurtosis(self) -> float:
        return numpy.exp(4 * self.sigma * self.sigma) + 2 * numpy.exp(3 * self.sigma * self.sigma) + 3 * numpy.exp(2 * self.sigma * self.sigma) - 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.exp(self.mu - self.sigma * self.sigma)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = numpy.log(continuous_measures.mean**2 / numpy.sqrt(continuous_measures.mean**2 + continuous_measures.variance))
        sigma = numpy.sqrt(numpy.log((continuous_measures.mean**2 + continuous_measures.variance) / (continuous_measures.mean**2)))
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class MAXWELL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "maxwell"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.alpha
        result = scipy.special.erf(z(x) / (numpy.sqrt(2))) - numpy.sqrt(2 / numpy.pi) * z(x) * numpy.exp(-z(x) ** 2 / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.alpha
        result = 1 / self.alpha * numpy.sqrt(2 / numpy.pi) * z(x) ** 2 * numpy.exp(-z(x) ** 2 / 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.alpha * numpy.sqrt(2 * scipy.special.gammaincinv(1.5, u)) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return 2 * numpy.sqrt(2 / numpy.pi) * self.alpha + self.loc

    @property
    def variance(self) -> float:
        return (self.alpha * self.alpha * (3 * numpy.pi - 8)) / numpy.pi

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 * numpy.sqrt(2) * (16 - 5 * numpy.pi)) / (3 * numpy.pi - 8) ** 1.5

    @property
    def kurtosis(self) -> float:
        return (4 * (-96 + 40 * numpy.pi - 3 * numpy.pi * numpy.pi)) / (3 * numpy.pi - 8) ** 2 + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.sqrt(2) * self.alpha + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        alpha = numpy.sqrt(continuous_measures.variance * numpy.pi / (3 * numpy.pi - 8))
        loc = continuous_measures.mean - 2 * alpha * numpy.sqrt(2 / numpy.pi)
        parameters = {"alpha": alpha, "loc": loc}
        return parameters


class MOYAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "moyal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = scipy.special.erfc(numpy.exp(-0.5 * z(x)) / numpy.sqrt(2))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = numpy.exp(-0.5 * (z(x) + numpy.exp(-z(x)))) / (self.sigma * numpy.sqrt(2 * numpy.pi))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu - self.sigma * numpy.log(scipy.stats.norm.ppf(1 - u / 2) ** 2)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu + self.sigma * (numpy.log(2) + 0.577215664901532)

    @property
    def variance(self) -> float:
        return (self.sigma * self.sigma * numpy.pi * numpy.pi) / 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 1.5351415907229

    @property
    def kurtosis(self) -> float:
        return 7

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        sigma = numpy.sqrt(2 * continuous_measures.variance / (numpy.pi * numpy.pi))
        mu = continuous_measures.mean - sigma * (numpy.log(2) + 0.577215664901532)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class NAKAGAMI:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.m = self.parameters["m"]
        self.omega = self.parameters["omega"]

    @property
    def name(self):
        return "nakagami"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.m, (self.m / self.omega) * x**2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (2 * self.m**self.m) / (scipy.special.gamma(self.m) * self.omega**self.m) * (x ** (2 * self.m - 1) * numpy.exp(-(self.m / self.omega) * x**2))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(scipy.special.gammaincinv(self.m, u) * (self.omega / self.m))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(self.omega / self.m)

    @property
    def variance(self) -> float:
        return self.omega * (1 - (1 / self.m) * (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) ** 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (
            (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m))
            * numpy.sqrt(1 / self.m)
            * (1 - 4 * self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2))
        ) / (2 * self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2) ** 1.5)

    @property
    def kurtosis(self) -> float:
        return 3 + (
            -6 * ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 4 * self.m
            + (8 * self.m - 2) * ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2
            - 2 * self.m
            + 1
        ) / (self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2) ** 2)

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (numpy.sqrt(2) / 2) * numpy.sqrt((self.omega * (2 * self.m - 1)) / self.m)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.m >= 0.5
        v2 = self.omega > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        d = numpy.array(continuous_measures.data)
        E_x2 = sum(d * d) / len(d)
        E_x4 = sum(d * d * d * d) / len(d)
        omega = E_x2
        m = E_x2**2 / (E_x4 - E_x2**2)
        parameters = {"m": m, "omega": omega}
        return parameters


class NON_CENTRAL_CHI_SQUARE:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]

    @property
    def name(self):
        return "non_central_chi_square"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncx2.cdf(x, self.lambda_, self.n)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncx2.pdf(x, self.lambda_, self.n)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncx2.ppf(u, self.lambda_, self.n)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.lambda_ + self.n

    @property
    def variance(self) -> float:
        return 2 * (self.n + 2 * self.lambda_)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2**1.5 * (self.n + 3 * self.lambda_)) / (self.n + 2 * self.lambda_) ** 1.5

    @property
    def kurtosis(self) -> float:
        return 3 + (12 * (self.n + 4 * self.lambda_)) / (self.n + 2 * self.lambda_) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        v2 = self.n > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        lambda_ = continuous_measures.variance / 2 - continuous_measures.mean
        n = 2 * continuous_measures.mean - continuous_measures.variance / 2
        parameters = {"lambda": lambda_, "n": n}
        return parameters


class NON_CENTRAL_F:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.lambda_ = self.parameters["lambda"]
        self.n1 = self.parameters["n1"]
        self.n2 = self.parameters["n2"]

    @property
    def name(self):
        return "non_central_f"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncf.cdf(x, self.n1, self.n2, self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncf.pdf(x, self.n1, self.n2, self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncf.ppf(u, self.n1, self.n2, self.lambda_)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (self.n2 / self.n1) * ((self.n1 + self.lambda_) / (self.n2 - 2))
        if k == 2:
            return (self.n2 / self.n1) ** 2 * (1 / ((self.n2 - 2) * (self.n2 - 4))) * (self.lambda_**2 + (2 * self.lambda_ + self.n1) * (self.n1 + 2))
        if k == 3:
            return (
                (self.n2 / self.n1) ** 3
                * (1 / ((self.n2 - 2) * (self.n2 - 4) * (self.n2 - 6)))
                * (self.lambda_**3 + 3 * (self.n1 + 4) * self.lambda_**2 + (3 * self.lambda_ + self.n1) * (self.n1 + 4) * (self.n1 + 2))
            )
        if k == 4:
            return (
                (self.n2 / self.n1) ** 4
                * (1 / ((self.n2 - 2) * (self.n2 - 4) * (self.n2 - 6) * (self.n2 - 8)))
                * (
                    self.lambda_**4
                    + 4 * (self.n1 + 6) * self.lambda_**3
                    + 6 * (self.n1 + 6) * (self.n1 + 4) * self.lambda_**2
                    + (4 * self.lambda_ + self.n1) * (self.n1 + 2) * (self.n1 + 4) * (self.n1 + 6)
                )
            )
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        v2 = self.n1 > 0
        v3 = self.n2 > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            lambda_, n1, n2 = initial_solution
            E_1 = (n2 / n1) * ((n1 + lambda_) / (n2 - 2))
            E_2 = (n2 / n1) ** 2 * ((lambda_**2 + (2 * lambda_ + n1) * (n1 + 2)) / ((n2 - 2) * (n2 - 4)))
            E_3 = (n2 / n1) ** 3 * ((lambda_**3 + 3 * (n1 + 4) * lambda_**2 + (3 * lambda_ + n1) * (n1 + 2) * (n1 + 4)) / ((n2 - 2) * (n2 - 4) * (n2 - 6)))
            parametric_mean = E_1
            parametric_variance = E_2 - E_1**2
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, 1, 10)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"lambda": solution.x[0], "n1": solution.x[1], "n2": solution.x[2]}
        return parameters


class NON_CENTRAL_T_STUDENT:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "non_central_t_student"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nct.cdf(x, self.n, self.lambda_, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nct.pdf(x, self.n, self.lambda_, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nct.ppf(u, self.n, self.lambda_, loc=self.loc, scale=self.scale)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (self.lambda_ * numpy.sqrt(self.n / 2) * scipy.special.gamma((self.n - 1) / 2)) / scipy.special.gamma(self.n / 2)
        if k == 2:
            return (self.n * (1 + self.lambda_ * self.lambda_)) / (self.n - 2)
        if k == 3:
            return (self.n**1.5 * numpy.sqrt(2) * scipy.special.gamma((self.n - 3) / 2) * self.lambda_ * (3 + self.lambda_ * self.lambda_)) / (4 * scipy.special.gamma(self.n / 2))
        if k == 4:
            return (self.n * self.n * (self.lambda_**4 + 6 * self.lambda_**2 + 3)) / ((self.n - 2) * (self.n - 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance())

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.n > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            lambda_, n, loc, scale = initial_solution
            E_1 = lambda_ * numpy.sqrt(n / 2) * scipy.special.gamma((n - 1) / 2) / scipy.special.gamma(n / 2)
            E_2 = (1 + lambda_**2) * n / (n - 2)
            E_3 = lambda_ * (3 + lambda_**2) * n**1.5 * numpy.sqrt(2) * scipy.special.gamma((n - 3) / 2) / (4 * scipy.special.gamma(n / 2))
            E_4 = (lambda_**4 + 6 * lambda_**2 + 3) * n**2 / ((n - 2) * (n - 4))
            parametric_mean = E_1 * scale + loc
            parametric_variance = (E_2 - E_1**2) * (scale**2)
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            parametric_kurtosis = (E_4 - 4 * E_1 * E_3 + 6 * E_1**2 * E_2 - 3 * E_1**4) / ((E_2 - E_1**2)) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        bnds = ((0, 0, 0, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 5, continuous_measures.mean, 1)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"lambda": solution.x[0], "n": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        return parameters


class NORMAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "normal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = 0.5 * (1 + scipy.special.erf(z(x) / numpy.sqrt(2)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (1 / (self.sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((x - self.mu) ** 2) / (2 * self.sigma**2)))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * scipy.stats.norm.ppf(u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma * 3

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        sigma = continuous_measures.standard_deviation
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class PARETO_FIRST_KIND:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "pareto_first_kind"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.pareto.cdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.pareto.pdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.xm * (1 - u) ** -(1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.alpha * self.xm**k) / (self.alpha - k)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.xm + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.pareto.fit(continuous_measures.data)
        parameters = {"xm": scipy_params[2], "alpha": scipy_params[0], "loc": scipy_params[1]}
        return parameters


class PARETO_SECOND_KIND:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "pareto_second_kind"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.lomax.cdf(x, self.alpha, scale=self.xm, loc=self.loc)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.alpha * self.xm**self.alpha) / (((x - self.loc) + self.xm) ** (self.alpha + 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.xm / (1 - u) ** (1 / self.alpha) - self.xm
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.xm**k * scipy.special.gamma(self.alpha - k) * scipy.special.gamma(1 + k)) / scipy.special.gamma(self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        m = continuous_measures.mean
        v = continuous_measures.variance
        loc = scipy.stats.lomax.fit(continuous_measures.data)[1]
        xm = -((m - loc) * ((m - loc) ** 2 + v)) / ((m - loc) ** 2 - v)
        alpha = -(2 * v) / ((m - loc) ** 2 - v)
        parameters = {"xm": xm, "alpha": alpha, "loc": loc}
        return parameters


class PERT:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.alpha1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        self.alpha2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)

    @property
    def name(self):
        return "pert"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.a) / (self.c - self.a)
        result = scipy.special.betainc(self.alpha1, self.alpha2, z(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (x - self.a) ** (self.alpha1 - 1) * (self.c - x) ** (self.alpha2 - 1) / (scipy.special.beta(self.alpha1, self.alpha2) * (self.c - self.a) ** (self.alpha1 + self.alpha2 - 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a + (self.c - self.a) * scipy.special.betaincinv(self.alpha1, self.alpha2, u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (self.a + 4 * self.b + self.c) / 6

    @property
    def variance(self) -> float:
        return ((self.mean - self.a) * (self.c - self.mean)) / 7

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        α1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        α2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)
        return (2 * (α2 - α1) * numpy.sqrt(α1 + α2 + 1)) / ((α1 + α2 + 2) * numpy.sqrt(α1 * α2))

    @property
    def kurtosis(self) -> float:
        α1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        α2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)
        return (6 * ((α2 - α1) ** 2 * (α1 + α2 + 1) - α1 * α2 * (α1 + α2 + 2))) / (α1 * α2 * (α1 + α2 + 2) * (α1 + α2 + 3)) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.b

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.b
        v2 = self.b < self.c
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, c = initial_solution
            self.alpha1 = (4 * b + c - 5 * a) / (c - a)
            self.alpha2 = (5 * c - a - 4 * b) / (c - a)
            parametric_mean = (a + 4 * b + c) / 6
            parametric_variance = ((parametric_mean - a) * (c - parametric_mean)) / 7
            parametric_median = scipy.special.betaincinv(self.alpha1, self.alpha2, 0.5) * (c - a) + a
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq5 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq5)

        bnds = ((-numpy.inf, continuous_measures.mean, continuous_measures.min), (continuous_measures.mean, numpy.inf, continuous_measures.max))
        x0 = (continuous_measures.min, continuous_measures.mean, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"a": solution.x[0], "b": solution.x[1], "c": solution.x[2]}
        parameters["a"] = min(continuous_measures.min - 1e-3, parameters["a"])
        parameters["c"] = max(continuous_measures.max + 1e-3, parameters["c"])
        return parameters


class POWER_FUNCTION:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "power_function"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return ((x - self.a) / (self.b - self.a)) ** self.alpha

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.alpha * ((x - self.a) ** (self.alpha - 1)) / ((self.b - self.a) ** self.alpha)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = u ** (1 / self.alpha) * (self.b - self.a) + self.a
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (self.a + self.b * self.alpha) / (self.alpha + 1)
        if k == 2:
            return (2 * self.a**2 + 2 * self.alpha * self.a * self.b + self.alpha * (self.alpha + 1) * self.b**2) / ((self.alpha + 1) * (self.alpha + 2))
        if k == 3:
            return (
                6 * self.a**3 + 6 * self.a**2 * self.b * self.alpha + 3 * self.a * self.b**2 * self.alpha * (1 + self.alpha) + self.b**3 * self.alpha * (1 + self.alpha) * (2 + self.alpha)
            ) / ((1 + self.alpha) * (2 + self.alpha) * (3 + self.alpha))
        if k == 4:
            return (
                24 * self.a**4
                + 24 * self.alpha * self.a**3 * self.b
                + 12 * self.alpha * (self.alpha + 1) * self.a**2 * self.b**2
                + 4 * self.alpha * (self.alpha + 1) * (self.alpha + 2) * self.a * self.b**3
                + self.alpha * (self.alpha + 1) * (self.alpha + 2) * (self.alpha + 3) * self.b**4
            ) / ((self.alpha + 1) * (self.alpha + 2) * (self.alpha + 3) * (self.alpha + 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.max([self.a, self.b])

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.b > self.a
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, a, b = initial_solution
            E1 = (a + b * alpha) / (1 + alpha)
            E2 = (2 * a**2 + 2 * a * b * alpha + b**2 * alpha * (1 + alpha)) / ((1 + alpha) * (2 + alpha))
            E3 = (6 * a**3 + 6 * a**2 * b * alpha + 3 * a * b**2 * alpha * (1 + alpha) + b**3 * alpha * (1 + alpha) * (2 + alpha)) / ((1 + alpha) * (2 + alpha) * (3 + alpha))
            parametric_mean = E1
            parametric_variance = E2 - E1**2
            parametric_skewness = (E3 - 3 * E2 * E1 + 2 * E1**3) / ((E2 - E1**2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        bnds = ((0, -numpy.inf, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "a": solution.x[1], "b": continuous_measures.max + 1e-3}
        return parameters


class RAYLEIGH:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.gamma = self.parameters["gamma"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "rayleigh"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.gamma) / self.sigma
        return 1 - numpy.exp(-0.5 * (z(x) ** 2))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.gamma) / self.sigma
        return z(x) * numpy.exp(-0.5 * (z(x) ** 2)) / self.sigma

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(-2 * numpy.log(1 - u)) * self.sigma + self.gamma
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.sigma * numpy.sqrt(numpy.pi / 2) + self.gamma

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma * (2 - numpy.pi / 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0.6311

    @property
    def kurtosis(self) -> float:
        return (24 * numpy.pi - 6 * numpy.pi * numpy.pi - 16) / ((4 - numpy.pi) * (4 - numpy.pi)) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.gamma + self.sigma

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        sigma = numpy.sqrt(continuous_measures.variance * 2 / (4 - numpy.pi))
        gamma = continuous_measures.mean - sigma * numpy.sqrt(numpy.pi / 2)
        parameters = {"gamma": gamma, "sigma": sigma}
        return parameters


class RECIPROCAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "reciprocal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (numpy.log(x) - numpy.log(self.a)) / (numpy.log(self.b) - numpy.log(self.a))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 / (x * (numpy.log(self.b) - numpy.log(self.a)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.exp(u * (numpy.log(self.b) - numpy.log(self.a)) + numpy.log(self.a))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.b**k - self.a**k) / (k * (numpy.log(self.b) - numpy.log(self.a)))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.a

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-8
        b = continuous_measures.max + 1e-8
        parameters = {"a": a, "b": b}
        return parameters


class RICE:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.v = self.parameters["v"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "rice"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.rice.cdf(x, self.v / self.sigma, scale=self.sigma)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.rice.pdf(x, self.v / self.sigma, scale=self.sigma)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.rice.ppf(u, self.v / self.sigma, scale=self.sigma)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (
                self.sigma
                * numpy.sqrt(numpy.pi / 2)
                * numpy.exp((-self.v * self.v) / (2 * self.sigma * self.sigma) / 2)
                * (
                    (1 - (-self.v * self.v) / (2 * self.sigma * self.sigma)) * scipy.special.iv(0, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                    + ((-self.v * self.v) / (2 * self.sigma * self.sigma)) * scipy.special.iv(1, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                )
            )
        if k == 2:
            return 2 * self.sigma * self.sigma + self.v * self.v
        if k == 3:
            return (
                3
                * self.sigma**3
                * numpy.sqrt(numpy.pi / 2)
                * numpy.exp((-self.v * self.v) / (2 * self.sigma * self.sigma) / 2)
                * (
                    (2 * ((-self.v * self.v) / (2 * self.sigma * self.sigma)) ** 2 - 6 * ((-self.v * self.v) / (2 * self.sigma * self.sigma)) + 3)
                    * scipy.special.iv(0, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                    - 2
                    * ((-self.v * self.v) / (2 * self.sigma * self.sigma) - 2)
                    * ((-self.v * self.v) / (2 * self.sigma * self.sigma))
                    * scipy.special.iv(1, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                )
            ) / 3
        if k == 4:
            return 8 * self.sigma**4 + 8 * self.sigma * self.sigma * self.v * self.v + self.v**4
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.v > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            v, sigma = initial_solution
            E = lambda k: sigma**k * 2 ** (k / 2) * scipy.special.gamma(1 + k / 2) * scipy.special.eval_laguerre(k / 2, -v * v / (2 * sigma * sigma))
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        bnds = ((0, 0), (numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, numpy.sqrt(continuous_measures.variance))
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"v": solution.x[0], "sigma": solution.x[1]}
        return parameters


class SEMICIRCULAR:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.loc = self.parameters["loc"]
        self.R = self.parameters["R"]

    @property
    def name(self):
        return "semicircular"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: t - self.loc
        result = 0.5 + z(x) * numpy.sqrt(self.R**2 - z(x) ** 2) / (numpy.pi * self.R**2) + numpy.arcsin(z(x) / self.R) / numpy.pi
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: t - self.loc
        result = 2 * numpy.sqrt(self.R**2 - z(x) ** 2) / (numpy.pi * self.R**2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.R * (2 * scipy.special.betaincinv(1.5, 1.5, u) - 1)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.loc

    @property
    def variance(self) -> float:
        return (self.R * self.R) / 4

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.R > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        loc = continuous_measures.mean
        R = numpy.sqrt(4 * continuous_measures.variance)
        d1 = (loc - R) - continuous_measures.min
        d2 = continuous_measures.max - (loc + R)
        delta = max(max(d1, 0), max(d2, 0)) + 1e-2
        R = R + delta
        parameters = {"loc": loc, "R": R}
        return parameters


class TRAPEZOIDAL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.d = self.parameters["d"]

    @property
    def name(self):
        return "trapezoidal"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.trapezoid.cdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.trapezoid.pdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.trapezoid.ppf(u, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (
            (2 / (self.d + self.c - self.b - self.a))
            * (1 / ((k + 1) * (k + 2)))
            * ((self.d ** (k + 2) - self.c ** (k + 2)) / (self.d - self.c) - (self.b ** (k + 2) - self.a ** (k + 2)) / (self.b - self.a))
        )

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.b
        v2 = self.b < self.c
        v3 = self.c < self.d
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution, continuous_measures, a, d):
            b, c = initial_solution
            parametric_mean = (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            parametric_variance = (1 / (6 * (d + c - a - b))) * ((d**4 - c**4) / (d - c) - (b**4 - a**4) / (b - a)) - (
                (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            ) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        a = continuous_measures.min - 1e-3
        d = continuous_measures.max + 1e-3
        x0 = [(d + a) * 0.25, (d + a) * 0.75]
        bnds = ((a, a), (d, d))
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=([continuous_measures, a, d]))
        parameters = {"a": a, "b": solution.x[0], "c": solution.x[1], "d": d}
        return parameters


class TRIANGULAR:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "triangular"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.triang.cdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.triang.pdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.triang.ppf(u, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (self.a + self.b + self.c) / 3

    @property
    def variance(self) -> float:
        return (self.a**2 + self.b**2 + self.c**2 - self.a * self.b - self.a * self.c - self.b * self.c) / 18

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.sqrt(2) * (self.a + self.b - 2 * self.c) * (2 * self.a - self.b - self.c) * (self.a - 2 * self.b + self.c)) / (
            5 * (self.a**2 + self.b**2 + self.c**2 - self.a * self.b - self.a * self.c - self.b * self.c) ** (3 / 2)
        )

    @property
    def kurtosis(self) -> float:
        return 3 - 3 / 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.c

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.c
        v2 = self.c < self.b
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-3
        b = continuous_measures.max + 1e-3
        c = 3 * continuous_measures.mean - a - b
        parameters = {"a": a, "b": b, "c": c}
        return parameters


class T_STUDENT:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.df = self.parameters["df"]

    @property
    def name(self):
        return "t_student"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.betainc(self.df / 2, self.df / 2, (x + numpy.sqrt(x * x + self.df)) / (2 * numpy.sqrt(x * x + self.df)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (1 / (numpy.sqrt(self.df) * scipy.special.beta(0.5, self.df / 2))) * (1 + x * x / self.df) ** (-(self.df + 1) / 2)
        return result

    def ppf(self, u):
        if u >= 0.5:
            result = numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
        else:
            result = -numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u]))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u])))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return 0

    @property
    def variance(self) -> float:
        return self.df / (self.df - 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 6 / (self.df - 4) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        df = 2 * continuous_measures.variance / (continuous_measures.variance - 1)
        parameters = {"df": df}
        return parameters


class T_STUDENT_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "t_student_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.special.betainc(self.df / 2, self.df / 2, (z(x) + numpy.sqrt(z(x) ** 2 + self.df)) / (2 * numpy.sqrt(z(x) ** 2 + self.df)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / (numpy.sqrt(self.df) * scipy.special.beta(0.5, self.df / 2))) * (1 + z(x) * z(x) / self.df) ** (-(self.df + 1) / 2)
        return result

    def ppf(self, u):
        result = self.loc + self.scale * numpy.sign(u - 0.5) * numpy.sqrt(
            self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u]))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u]))
        )
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.loc

    @property
    def variance(self) -> float:
        return (self.scale * self.scale * self.df) / (self.df - 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 6 / (self.df - 4) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_params = scipy.stats.t.fit(continuous_measures.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters


class UNIFORM:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "uniform"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (x - self.a) / (self.b - self.a)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 / (self.b - self.a)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a + u * (self.b - self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (self.a + self.b) / 2

    @property
    def variance(self) -> float:
        return (self.b - self.a) ** 2 / 12

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 3 - 6 / 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-8
        b = continuous_measures.max + 1e-8
        parameters = {"a": a, "b": b}
        return parameters


class WEIBULL:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "weibull"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 - numpy.exp(-((x / self.beta) ** self.alpha))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.alpha / self.beta) * ((x / self.beta) ** (self.alpha - 1)) * numpy.exp(-((x / self.beta) ** self.alpha))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * (-numpy.log(1 - u)) ** (1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * scipy.special.gamma(1 + k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.alpha <= 1:
            return 0
        return self.beta * ((self.alpha - 1) / self.alpha) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta = initial_solution
            E = lambda k: (beta**k) * scipy.special.gamma(1 + k / alpha)
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        solution = scipy.optimize.fsolve(equations, (1, 1), continuous_measures)
        parameters = {"alpha": solution[0], "beta": solution[1]}
        return parameters


class WEIBULL_3P:
    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        if continuous_measures is None and parameters is None:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        else:
            self.parameters = parameters
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "weibull_3p"

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.beta
        return 1 - numpy.exp(-(z(x) ** self.alpha))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.beta
        return (self.alpha / self.beta) * (z(x) ** (self.alpha - 1)) * numpy.exp(-z(x) ** self.alpha)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.beta * (-numpy.log(1 - u)) ** (1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * scipy.special.gamma(1 + k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.alpha <= 1:
            return 0
        return self.loc + self.beta * ((self.alpha - 1) / self.alpha) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, loc = initial_solution
            E = lambda k: (beta**k) * scipy.special.gamma(1 + k / alpha)
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        bnds = ((0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, continuous_measures.mean)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "loc": solution.x[2], "beta": solution.x[1]}
        return parameters


ALL_CONTINUOUS_DISTRIBUTIONS = {
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


class CONTINUOUS_MEASURES:
    def __init__(
        self,
        data: list[float | int],
        num_bins: int | None = None,
        confidence_level: float = 0.95,
    ):
        self.data = numpy.sort(data)
        self.data_unique = numpy.unique(self.data)
        self.length = len(self.data)
        self.min = self.data[0]
        self.max = self.data[-1]
        self.mean = numpy.mean(self.data)
        self.variance = numpy.var(self.data, ddof=1)
        self.standard_deviation = numpy.sqrt(self.variance)
        self.skewness = scipy.stats.moment(self.data, 3) / pow(self.standard_deviation, 3)
        self.kurtosis = scipy.stats.moment(self.data, 4) / pow(self.standard_deviation, 4)
        self.median = numpy.median(self.data)
        self.mode = self.calculate_mode()
        self.num_bins = num_bins if num_bins != None else len(numpy.histogram_bin_edges(self.data, bins="doane"))
        self.absolutes_frequencies, self.bin_edges = numpy.histogram(self.data, self.num_bins)
        self.densities_frequencies, _ = numpy.histogram(self.data, self.num_bins, density=True)
        self.central_values = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.length - 1]])
        self.Sn_ks = (numpy.arange(self.length) + 1) / self.length
        self.confidence_level = confidence_level
        self.critical_value_ks = scipy.stats.kstwo.ppf(self.confidence_level, self.length)
        self.critical_value_ad = self.ad_critical_value(self.confidence_level, self.length)
        self.ecdf_frequencies = numpy.arange(1, len(self.data_unique) + 1) / len(self.data_unique)
        self.qq_arr = (numpy.arange(1, self.length + 1) - 0.5) / self.length

    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def calculate_mode(self) -> float:
        distribution = scipy.stats.gaussian_kde(self.data)
        solution = scipy.optimize.minimize(lambda x: -distribution.pdf(x)[0], x0=[self.mean], bounds=[(self.min, self.max)])
        return solution.x[0]

    def critical_value_chi2(self, freedom_degrees: int):
        return scipy.stats.chi2.ppf(self.confidence_level, freedom_degrees)

    def adinf(self, z: float):
        if z < 2:
            return (z**-0.5) * numpy.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
        return numpy.exp(-numpy.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))

    def errfix(self, n: float, x: float) -> float:
        def g1(t: float) -> float:
            return numpy.sqrt(t) * (1 - t) * (49 * t - 102)

        def g2(t: float) -> float:
            return -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t

        def g3(t: float) -> float:
            return -130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * t) * t) * t) * t) * t

        c = 0.01265 + 0.1757 / n
        if x < c:
            return (0.0037 / (n**3) + 0.00078 / (n**2) + 0.00006 / n) * g1(x / c)
        elif x > c and x < 0.8:
            return (0.04213 / n + 0.01365 / (n**2)) * g2((x - c) / (0.8 - c))
        else:
            return (g3(x)) / n

    def AD(self, n: float, z: float) -> float:
        return self.adinf(z) + self.errfix(n, self.adinf(z))

    def ad_critical_value(self, q: float, n: float) -> float:
        f = lambda x: self.AD(n, x) - q
        root = scipy.optimize.newton(f, 2)
        return root

    def ad_p_value(self, n: float, z: float) -> float:
        return 1 - self.AD(n, z)


def evaluate_continuous_test_chi_square(distribution, continuous_measures):
    N = continuous_measures.length
    freedom_degrees = max(continuous_measures.num_bins - 1 - distribution.num_parameters, 1)
    expected_values = N * (distribution.cdf(continuous_measures.bin_edges[1:]) - distribution.cdf(continuous_measures.bin_edges[:-1]))
    errors = ((continuous_measures.absolutes_frequencies - expected_values) ** 2) / expected_values
    statistic_chi2 = numpy.sum(errors)
    critical_value = continuous_measures.critical_value_chi2(freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {"test_statistic": statistic_chi2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_chi2


def evaluate_continuous_test_kolmogorov_smirnov(distribution, continuous_measures):
    N = continuous_measures.length
    Fn = distribution.cdf(continuous_measures.data)
    errors = numpy.abs(continuous_measures.Sn_ks[continuous_measures.idx_ks] - Fn[continuous_measures.idx_ks])
    statistic_ks = numpy.max(errors)
    critical_value = continuous_measures.critical_value_ks
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, N)
    rejected = statistic_ks >= critical_value
    result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ks


def evaluate_continuous_test_anderson_darling(distribution, continuous_measures):
    N = continuous_measures.length
    S = numpy.sum(((2 * (numpy.arange(N) + 1) - 1) / N) * (numpy.log(distribution.cdf(continuous_measures.data)) + numpy.log(1 - distribution.cdf(continuous_measures.data[::-1]))))
    A2 = -N - S
    critical_value = continuous_measures.critical_value_ad
    p_value = continuous_measures.ad_p_value(N, A2)
    rejected = A2 >= critical_value
    result_test_ad = {"test_statistic": A2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ad


class PHITTER_CONTINUOUS:
    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        num_bins: int | None = None,
        confidence_level=0.95,
        minimum_sse=numpy.inf,
        distributions_to_fit: list[str] | typing.Literal["all"] = "all",
    ):
        if distributions_to_fit != "all":
            not_distributions_ids = [dist for dist in distributions_to_fit if dist not in ALL_CONTINUOUS_DISTRIBUTIONS.keys()]
            if len(not_distributions_ids) > 0:
                raise Exception(f"{not_distributions_ids} not founded in continuous disributions")
        self.data = data
        self.continuous_measures = CONTINUOUS_MEASURES(self.data, num_bins, confidence_level)
        self.confidence_level = confidence_level
        self.minimum_sse = minimum_sse
        self.distribution_results = {}
        self.none_results = {"test_statistic": None, "critical_value": None, "p_value": None, "rejected": None}
        self.distributions_to_fit = list(ALL_CONTINUOUS_DISTRIBUTIONS.keys()) if distributions_to_fit == "all" else distributions_to_fit
        self.sorted_distributions_sse = None
        self.not_rejected_distributions = None
        self.distribution_instances = None

    def test(self, test_function, label: str, distribution):
        validation_test = False
        try:
            test = test_function(distribution, self.continuous_measures)
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

    def process_distribution(self, distribution_name: str) -> tuple[str, dict, typing.Any] | None:
        distribution_class = ALL_CONTINUOUS_DISTRIBUTIONS[distribution_name]
        validate_estimation = True
        sse = 0
        try:
            distribution = distribution_class(self.continuous_measures)
            pdf_values = distribution.pdf(self.continuous_measures.central_values)
            sse = numpy.sum(numpy.power(self.continuous_measures.densities_frequencies - pdf_values, 2.0))
        except:
            validate_estimation = False
        self.distribution_results = {}
        if validate_estimation and distribution.parameter_restrictions() and not numpy.isnan(sse) and not numpy.isinf(sse) and sse < self.minimum_sse:
            v1 = self.test(evaluate_continuous_test_chi_square, "chi_square", distribution)
            v2 = self.test(evaluate_continuous_test_kolmogorov_smirnov, "kolmogorov_smirnov", distribution)
            v3 = self.test(evaluate_continuous_test_anderson_darling, "anderson_darling", distribution)
            if v1 or v2 or v3:
                self.distribution_results["sse"] = sse
                self.distribution_results["parameters"] = distribution.parameters
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
                return distribution_name, self.distribution_results, distribution
        return None

    def fit(self, n_workers: int = 1):
        if n_workers <= 0:
            raise Exception("n_workers must be greater than 1")
        if n_workers == 1:
            processing_results = [self.process_distribution(distribution_name) for distribution_name in self.distributions_to_fit]
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
            processing_results = list(executor.map(self.process_distribution, self.distributions_to_fit))
        processing_results = [r for r in processing_results if r is not None]
        self.sorted_distributions_sse = {distribution: results for distribution, results, _ in sorted(processing_results, key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
        self.not_rejected_distributions = {distribution: results for distribution, results in self.sorted_distributions_sse.items() if results["n_test_passed"] > 0}
        self.distribution_instances = {distribution: instance for distribution, _, instance in processing_results}
