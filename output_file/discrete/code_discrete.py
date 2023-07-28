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
        if ( x < 0 ):
            result = 0
        elif( x >= 0 and x < 1 ):
            result = 1 - self.p
        else:
            result = 1
        return result
    def pmf(self, x: int) -> float:
        result = (self.p ** x) * (1 - self.p) ** (1 - x)
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
        result = sc.comb(self.n, x) * (self.p ** x) * ((1 - self.p) ** (self.n - x))
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
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
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
        result = -(self.p ** x) / (math.log(1 - self.p) * x)
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
            return (eq1)
        solution = scipy.optimize.least_squares(equations, 0.5, bounds = (0, 1), args=([measurements]))
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
        result = sc.comb(self.r + x - 1, x) * (self.p ** self.r) * ((1 - self.p) ** x)
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
        result = (self.lambda_ ** x) * math.exp(-self.lambda_) / math.factorial(x)
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

