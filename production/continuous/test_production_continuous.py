import concurrent.futures
import math
import numpy
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats
import warnings

class ALPHA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.stats.norm.cdf(self.alpha - (1 / z(x))) / scipy.stats.norm.cdf(self.alpha)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.alpha.pdf(x, self.alpha, loc=self.loc, scale=self.scale)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.scale > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.alpha.fit(measurements.data)
        parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters

class ARCSINE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
    def cdf(self, x: float) -> float:
        z = lambda t: (x - self.a) / (self.b - self.a)
        return 2 * numpy.arcsin(numpy.sqrt(z(x))) / numpy.pi
    def pdf(self, x: float) -> float:
        return 1 / (numpy.pi * numpy.sqrt((x - self.a) * (self.b - x)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        _a = measurements.min - 1e-3
        _b = measurements.max + 1e-3
        parameters = {"a": _a, "b": _b}
        return parameters

class ARGUS:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.chi = self.parameters["chi"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = 1 - scipy.special.gammainc(1.5, self.chi * self.chi * (1 - z(x) ** 2) / 2) / scipy.special.gammainc(1.5, self.chi * self.chi / 2)
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        result = (1 / self.scale) * ((self.chi**3) / (numpy.sqrt(2 * numpy.pi) * Ψ(self.chi))) * z(x) * numpy.sqrt(1 - z(x) * z(x)) * numpy.exp(-0.5 * self.chi**2 * (1 - z(x) * z(x)))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.chi > 0
        v2 = self.scale > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.argus.fit(measurements.data)
        parameters = {"chi": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters

class BETA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.A) / (self.B - self.A)
        result = scipy.stats.beta.cdf(z(x), self.alpha, self.beta)
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.A) / (self.B - self.A)
        result = scipy.stats.beta.pdf(z(x), self.alpha, self.beta)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.A < self.B
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha, beta, A, B = initial_solution
            parametric_mean = A + (alpha / (alpha + beta)) * (B - A)
            parametric_variance = ((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))) * (B - A) ** 2
            parametric_skewness = 2 * ((beta - alpha) / (alpha + beta + 2)) * numpy.sqrt((alpha + beta + 1) / (alpha * beta))
            parametric_kurtosis = 3 * (((alpha + beta + 1) * (2 * (alpha + beta) ** 2 + (alpha * beta) * (alpha + beta - 6))) / ((alpha * beta) * (alpha + beta + 2) * (alpha + beta + 3)))
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            eq4 = parametric_kurtosis - measurements.kurtosis
            return (eq1, eq2, eq3, eq4)
        bnds = ((0, 0, -numpy.inf, measurements.mean), (numpy.inf, numpy.inf, measurements.mean, numpy.inf))
        x0 = (1, 1, measurements.min, measurements.max)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "A": solution.x[2], "B": solution.x[3]}
        v1 = parameters["alpha"] > 0
        v2 = parameters["beta"] > 0
        v3 = parameters["A"] < parameters["B"]
        if (v1 and v2 and v3) == False:
            scipy_params = scipy.stats.beta.fit(measurements.data)
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "A": scipy_params[2], "B": scipy_params[3]}
        return parameters

class BETA_PRIME:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.betaprime.cdf(x, self.alpha, self.beta)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.betaprime.pdf(x, self.alpha, self.beta)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha, beta = initial_solution
            parametric_mean = alpha / (beta - 1)
            parametric_variance = alpha * (alpha + beta - 1) / ((beta - 1) ** 2 * (beta - 2))
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)
        scipy_params = scipy.stats.betaprime.fit(measurements.data)
        try:
            bnds = ((0, 0), (numpy.inf, numpy.inf))
            x0 = (scipy_params[0], scipy_params[1])
            args = [measurements]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
        except:
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1]}
        return parameters

class BETA_PRIME_4P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.betaprime.cdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.betaprime.pdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.scale > 0
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha, beta, scale, loc = initial_solution
            parametric_mean = scale * alpha / (beta - 1) + loc
            parametric_variance = (scale**2) * alpha * (alpha + beta - 1) / ((beta - 1) ** 2 * (beta - 2))
            parametric_median = loc + scale * scipy.stats.beta.ppf(0.5, alpha, beta) / (1 - scipy.stats.beta.ppf(0.5, alpha, beta))
            parametric_mode = scale * (alpha - 1) / (beta + 1) + loc
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - measurements.median
            eq4 = parametric_mode - measurements.mode
            return (eq1, eq2, eq3, eq4)
        scipy_params = scipy.stats.betaprime.fit(measurements.data)
        try:
            bnds = ((0, 0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            x0 = (measurements.mean, measurements.mean, scipy_params[3], measurements.mean)
            args = [measurements]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1], "loc": solution.x[3], "scale": solution.x[2]}
        except:
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[1], "loc": scipy_params[2], "scale": scipy_params[3]}
        return parameters

class BRADFORD:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.min_ = self.parameters["min"]
        self.max_ = self.parameters["max"]
    def cdf(self, x: float) -> float:
        result = numpy.log(1 + self.c * (x - self.min_) / (self.max_ - self.min_)) / numpy.log(self.c + 1)
        return result
    def pdf(self, x: float) -> float:
        result = self.c / ((self.c * (x - self.min_) + self.max_ - self.min_) * numpy.log(self.c + 1))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.max_ > self.min_
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        _min = measurements.min - 1e-3
        _max = measurements.max + 1e-3
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            c = initial_solution
            parametric_mean = (c * (_max - _min) + numpy.log(c + 1) * (_min * (c + 1) - _max)) / (c * numpy.log(c + 1))
            eq1 = parametric_mean - measurements.mean
            return eq1
        solution = scipy.optimize.fsolve(equations, (1), measurements)
        parameters = {"c": solution[0], "min": _min, "max": _max}
        return parameters

warnings.filterwarnings("ignore")
class BURR:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.burr12.cdf(x, self.B, self.C, scale=self.A)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.burr12.pdf(x, self.B, self.C, scale=self.A)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.burr12.fit(measurements.data)
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1]}
        return parameters

warnings.filterwarnings("ignore")
class BURR_4P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.burr12.cdf(x, self.B, self.C, loc=self.loc, scale=self.A)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.burr12.pdf(x, self.B, self.C, loc=self.loc, scale=self.A)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.burr12.fit(measurements.data)
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1], "loc": scipy_params[2]}
        return parameters

class CAUCHY:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.x0 = self.parameters["x0"]
        self.gamma = self.parameters["gamma"]
    def cdf(self, x: float) -> float:
        return (1 / numpy.pi) * numpy.arctan(((x - self.x0) / self.gamma)) + (1 / 2)
    def pdf(self, x: float) -> float:
        return 1 / (numpy.pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.gamma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.cauchy.fit(measurements.data)
        parameters = {"x0": scipy_params[0], "gamma": scipy_params[1]}
        return parameters

class CHI_SQUARE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.df / 2, x / 2)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.chi2.pdf(x, self.df)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = type(self.df) == int
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        parameters = {"df": round(measurements.mean)}
        return parameters

class CHI_SQUARE_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.special.gammainc(self.df / 2, z(x) / 2)
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * (1 / (2 ** (self.df / 2) * scipy.special.gamma(self.df / 2))) * (z(x) ** ((self.df / 2) - 1)) * (numpy.exp(-z(x) / 2))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = type(self.df) == int
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.chi2.fit(measurements.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters

class DAGUM:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]
    def cdf(self, x: float) -> float:
        return (1 + (x / self.b) ** (-self.a)) ** (-self.p)
    def pdf(self, x: float) -> float:
        return (self.a * self.p / x) * (((x / self.b) ** (self.a * self.p)) / ((((x / self.b) ** (self.a)) + 1) ** (self.p + 1)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def sse(parameters: dict) -> float:
            def __pdf(x: float, params: dict) -> float:
                return (params["a"] * params["p"] / x) * (((x / params["b"]) ** (params["a"] * params["p"])) / ((((x / params["b"]) ** (params["a"])) + 1) ** (params["p"] + 1)))
            frequencies, bin_edges = numpy.histogram(measurements.data, density=True)
            central_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            pdf_values = [__pdf(c, parameters) for c in central_values]
            sse = numpy.sum(numpy.power(frequencies - pdf_values, 2))
            return sse
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, b, p = initial_solution
            mu = lambda k: (b ** k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = mu(1)
            parametric_variance = -(mu(1) ** 2) + mu(2)
            parametric_median = b * ((2 ** (1 / p)) - 1) ** (-1 / a)
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - measurements.median
            return (eq1, eq2, eq3)
        s0_burr3_sc = scipy.stats.burr.fit(measurements.data)
        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1]}
        a0 = s0_burr3_sc[0]
        b0 = s0_burr3_sc[3]
        x0 = [a0, b0, 1]
        b = ((1e-5, 1e-5, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(
            equations, x0, bounds=b, args=([measurements]))
        parameters_ls = {"a": solution.x[0],
                         "b": solution.x[1], "p": solution.x[2]}
        sse_sc = sse(parameters_sc)
        sse_ls = sse(parameters_ls)
        if a0 <= 2:
            return(parameters_sc)
        else:
            if sse_sc < sse_ls:
                return(parameters_sc)
            else:
                return(parameters_ls)

class DAGUM_4P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        return (1 + ((x - self.loc) / self.b) ** (-self.a)) ** (-self.p)
    def pdf(self, x: float) -> float:
        return (self.a * self.p / x) * ((((x - self.loc) / self.b) ** (self.a * self.p)) / (((((x - self.loc) / self.b) ** (self.a)) + 1) ** (self.p + 1)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def sse(parameters: dict) -> float:
            def __pdf(x: float, params: dict) -> float:
                return (params["a"] * params["p"] / (x - params["loc"])) * (
                    (((x - params["loc"]) / params["b"]) ** (params["a"] * params["p"])) / ((((x / params["b"]) ** (params["a"])) + 1) ** (params["p"] + 1))
                )
            frequencies, bin_edges = numpy.histogram(measurements.data, density=True)
            central_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            pdf_values = [__pdf(c, parameters) for c in central_values]
            sse = numpy.sum(numpy.power(frequencies - pdf_values, 2))
            return sse
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, b, p, loc = initial_solution
            mu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = mu(1) + loc
            parametric_variance = -(mu(1) ** 2) + mu(2)
            parametric_median = b * ((2 ** (1 / p)) - 1) ** (-1 / a) + loc
            parametric_mode = b * ((a * p - 1) / (a + 1)) ** (1 / a) + loc
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - measurements.median
            eq4 = parametric_mode - measurements.mode
            return (eq1, eq2, eq3, eq4)
        s0_burr3_sc = scipy.stats.burr.fit(measurements.data)
        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1], "loc": s0_burr3_sc[2]}
        if s0_burr3_sc[0] <= 2:
            return parameters_sc
        else:
            a0 = s0_burr3_sc[0]
            x0 = [a0, 1, 1, measurements.mean]
            b = ((1e-5, 1e-5, 1e-5, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
            parameters_ls = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2], "loc": solution.x[3]}
            sse_sc = sse(parameters_sc)
            sse_ls = sse(parameters_ls)
            if sse_sc < sse_ls:
                return parameters_sc
            else:
                return parameters_ls

class ERLANG:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.k, x / self.beta)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.erlang.pdf(x, self.k, scale=self.beta)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.k > 0
        v2 = self.beta > 0
        v3 = type(self.k) == int
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        k = round(measurements.mean**2 / measurements.variance)
        beta = measurements.variance / measurements.mean
        parameters = {"k": k, "beta": beta}
        return parameters

class ERLANG_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.k, (x - self.loc) / self.beta)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.erlang.pdf(x, self.k, scale=self.beta, loc=self.loc)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.k > 0
        v2 = self.beta > 0
        v3 = type(self.k) == int
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        k = round((2 / measurements.skewness) ** 2)
        beta = numpy.sqrt(measurements.variance / ((2 / measurements.skewness) ** 2))
        loc = measurements.mean - ((2 / measurements.skewness) ** 2) * beta
        parameters = {"k": k, "beta": beta, "loc": loc}
        return parameters

class ERROR_FUNCTION:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.h = self.parameters["h"]
    def cdf(self, x: float) -> float:
        return scipy.stats.norm.cdf((2**0.5) * self.h * x)
    def pdf(self, x: float) -> float:
        return self.h * numpy.exp(-((self.h * x) ** 2)) / numpy.sqrt(numpy.pi)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.h > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        h = numpy.sqrt(1 / (2 * measurements.variance))
        parameters = {"h": h}
        return parameters

class EXPONENTIAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
    def cdf(self, x: float) -> float:
        return 1 - numpy.exp(-self.lambda_ * x)
    def pdf(self, x: float) -> float:
        return self.lambda_ * numpy.exp(-self.lambda_ * x)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        lambda_ = 1 / measurements.mean
        parameters = {"lambda": lambda_}
        return parameters

class EXPONENTIAL_2P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        return 1 - numpy.exp(-self.lambda_ * (x - self.loc))
    def pdf(self, x: float) -> float:
        return self.lambda_ * numpy.exp(-self.lambda_ * (x - self.loc))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        lambda_ = (1 - numpy.log(2)) / (measurements.mean - measurements.median)
        loc = measurements.min - 1e-4
        parameters = {"lambda": lambda_, "loc": loc}
        return parameters

class F:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.f.cdf(x, self.df1, self.df2)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.f.pdf(x, self.df1, self.df2)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.f.fit(measurements.data)
        parameters = {"df1": scipy_params[0], "df2": scipy_params[1]}
        return parameters

class FATIGUE_LIFE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.gamma = self.parameters["gamma"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        z = lambda t: numpy.sqrt((t - self.loc) / self.scale)
        result = scipy.stats.norm.cdf((z(x) - 1 / z(x)) / (self.gamma))
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: numpy.sqrt((t - self.loc) / self.scale)
        result = (z(x) + 1 / z(x)) / (2 * self.gamma * (x - self.loc)) * scipy.stats.norm.pdf((z(x) - 1 / z(x)) / (self.gamma))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        v2 = self.gamma > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.fatiguelife.fit(measurements.data)
        parameters = {"gamma": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters

class FOLDED_NORMAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z1 = lambda t: (t + self.mu) / self.sigma
        z2 = lambda t: (t - self.mu) / self.sigma
        result = 0.5 * (scipy.special.erf(z1(x) / numpy.sqrt(2)) + scipy.special.erf(z2(x) / numpy.sqrt(2)))
        return result
    def pdf(self, x: float) -> float:
        result = numpy.sqrt(2 / (numpy.pi * self.sigma**2)) * numpy.exp(-(x**2 + self.mu**2) / (2 * self.sigma**2)) * numpy.cosh(self.mu * x / (self.sigma**2))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = sigma * numpy.sqrt(2 / numpy.pi) * numpy.exp(-(mu**2) / (2 * sigma**2)) + mu * scipy.special.erf(mu / numpy.sqrt(2 * sigma**2))
            parametric_variance = mu**2 + sigma**2 - parametric_mean**2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)
        x0 = [measurements.mean, measurements.standard_deviation]
        b = ((-numpy.inf, 0), (numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
        parameters = {"mu": solution.x[0], "sigma": solution.x[1]}
        return parameters

class FRECHET:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * self.alpha * z(x) ** (-self.alpha - 1) * numpy.exp(-z(x) ** -self.alpha)
        return result
    def pdf(self, x: float) -> float:
        return (self.alpha / self.scale) * (((x - self.loc) / self.scale) ** (-1 - self.alpha)) * numpy.exp(-(((x - self.loc) / self.scale) ** (-self.alpha)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.scale > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.invweibull.fit(measurements.data)
        parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters

class F_4P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.f.cdf(x, self.df1, self.df2, self.loc, self.scale)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.f.pdf(x, self.df1, self.df2, self.loc, self.scale)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        v3 = self.scale > 0
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            df1, df2, loc, scale = initial_solution
            E = lambda k: (df2 / df1) ** k * (scipy.special.gamma(df1 / 2 + k) * scipy.special.gamma(df2 / 2 - k)) / (scipy.special.gamma(df1 / 2) * scipy.special.gamma(df2 / 2))
            parametric_mean = E(1) * scale + loc
            parametric_variance = (E(2) - E(1) ** 2) * (scale) ** 2
            parametric_median = scipy.stats.f.ppf(0.5, df1, df2) * scale + loc
            parametric_mode = ((df2 * (df1 - 2)) / (df1 * (df2 + 2))) * scale + loc
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - measurements.median
            eq4 = parametric_mode - measurements.mode
            return (eq1, eq2, eq3, eq4)
        try:
            bnds = ((0, 0, -numpy.inf, 0), (numpy.inf, numpy.inf, measurements.min, numpy.inf))
            x0 = (1, measurements.standard_deviation, measurements.min, measurements.standard_deviation)
            args = [measurements]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"df1": solution.x[0], "df2": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        except:
            scipy_params = scipy.stats.f.fit(measurements.data)
            parameters = {"df1": scipy_params[0], "df2": scipy_params[1], "loc": scipy_params[2], "scale": scipy_params[3]}
        return parameters

class GAMMA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.alpha, x / self.beta)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.gamma.pdf(x, self.alpha, scale=self.beta)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mean = measurements.mean
        variance = measurements.variance
        alpha = mean**2 / variance
        beta = variance / mean
        parameters = {"alpha": alpha, "beta": beta}
        return parameters

class GAMMA_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.alpha, (x - self.loc) / self.beta)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.gamma.pdf(x, self.alpha, loc=self.loc, scale=self.beta)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        alpha = (2 / measurements.skewness) ** 2
        beta = numpy.sqrt(measurements.variance / alpha)
        loc = measurements.mean - alpha * beta
        parameters = {"alpha": alpha, "loc": loc, "beta": beta}
        return parameters

class GENERALIZED_EXTREME_VALUE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xi = self.parameters["xi"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return numpy.exp(-numpy.exp(-z(x)))
        else:
            return numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi)))
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))
        else:
            return (1 / self.sigma) * numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi))) * (1 + self.xi * z(x)) ** (-1 - 1 / self.xi)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.genextreme.fit(measurements.data)
        parameters = {"xi": -scipy_params[0], "mu": scipy_params[1], "sigma": scipy_params[2]}
        return parameters

class GENERALIZED_GAMMA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.d / self.p, (x / self.a) ** self.p)
        return result
    def pdf(self, x: float) -> float:
        return (self.p / (self.a**self.d)) * (x ** (self.d - 1)) * numpy.exp(-((x / self.a) ** self.p)) / scipy.special.gamma(self.d / self.p)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, d, p = initial_solution
            E = lambda r: a**r * (scipy.special.gamma((d + r) / p) / scipy.special.gamma(d / p))
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            return (eq1, eq2, eq3)
        try:
            solution = scipy.optimize.fsolve(equations, (1, 1, 1), measurements)
            if all(x > 0 for x in solution) is False or all(x == 1 for x in solution) is True:
                bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
                x0 = (1, 1, 1)
                args = [measurements]
                response = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
                solution = response.x
            parameters = {"a": solution[0], "d": solution[1], "p": solution[2]}
        except:
            scipy_params = scipy.stats.gengamma.fit(measurements.data)
            parameters = {"a": scipy_params[0], "c": scipy_params[1], "mu": scipy_params[2], "sigma": scipy_params[3]}
        return parameters

class GENERALIZED_GAMMA_4P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.d / self.p, ((x - self.loc) / self.a) ** self.p)
        return result
    def pdf(self, x: float) -> float:
        return (self.p / (self.a**self.d)) * ((x - self.loc) ** (self.d - 1)) * numpy.exp(-(((x - self.loc) / self.a) ** self.p)) / scipy.special.gamma(self.d / self.p)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, d, p, loc = initial_solution
            E = lambda r: a**r * (scipy.special.gamma((d + r) / p) / scipy.special.gamma(d / p))
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            parametric_median = a * scipy.stats.gamma.ppf(0.5, a=d / p, scale=1) ** (1 / p) + loc
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - measurements.median
            eq4 = parametric_kurtosis - measurements.kurtosis
            return (eq1, eq2, eq3, eq4)
        solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), measurements)
        if all(x > 0 for x in solution) is False or all(x == 1 for x in solution) is True:
            try:
                bnds = ((0, 0, 0, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
                if measurements.mean < 0:
                    bnds = ((0, 0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, 0))
                x0 = (1, 1, 1, measurements.mean)
                args = [measurements]
                response = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
                solution = response.x
            except:
                scipy_params = scipy.stats.gengamma.fit(measurements.data)
                solution = [scipy_params[3], scipy_params[0], scipy_params[1], scipy_params[2]]
        parameters = {"a": solution[0], "d": solution[1], "p": solution[2], "loc": solution[3]}
        return parameters

class GENERALIZED_LOGISTIC:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        self.c = self.parameters["c"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        return 1 / ((1 + numpy.exp(-z(x))) ** self.c)
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        return (self.c / self.scale) * numpy.exp(-z(x)) * ((1 + numpy.exp(-z(x))) ** (-self.c - 1))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        v2 = self.c > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            c, loc, scale = initial_solution
            parametric_mean = loc + scale * (0.57721 + scipy.special.digamma(c))
            parametric_variance = scale**2 * (numpy.pi**2 / 6 + scipy.special.polygamma(1, c))
            parametric_median = loc + scale * (-numpy.log(0.5 ** (-1 / c) - 1))
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - measurements.median
            return (eq1, eq2, eq3)
        x0 = [measurements.mean, measurements.mean, measurements.mean]
        b = ((1e-5, -numpy.inf, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
        parameters = {"c": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}
        return parameters

class GENERALIZED_NORMAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.beta = self.parameters["beta"]
        self.mu = self.parameters["mu"]
        self.alpha = self.parameters["alpha"]
    def cdf(self, x: float) -> float:
        return 0.5 + (numpy.sign(x - self.mu) / 2) * scipy.special.gammainc(1 / self.beta, abs((x - self.mu) / self.alpha) ** self.beta)
    def pdf(self, x: float) -> float:
        return self.beta / (2 * self.alpha * scipy.special.gamma(1 / self.beta)) * numpy.exp(-((abs(x - self.mu) / self.alpha) ** self.beta))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.gennorm.fit(measurements.data)
        parameters = {"beta": scipy_params[0], "mu": scipy_params[1], "alpha": scipy_params[2]}
        return parameters

class GENERALIZED_PARETO:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        result = 1 - (1 + self.c * z(x)) ** (-1 / self.c)
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        result = (1 / self.sigma) * (1 + self.c * z(x)) ** (-1 / self.c - 1)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            c, mu, sigma = initial_solution
            parametric_mean = mu + sigma / (1 - c)
            parametric_variance = sigma * sigma / ((1 - c) * (1 - c) * (1 - 2 * c))
            parametric_median = mu + sigma * (2**c - 1) / c
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - numpy.percentile(measurements.data, 50)
            return (eq1, eq2, eq3)
        scipy_params = scipy.stats.genpareto.fit(measurements.data)
        parameters = {"c": scipy_params[0], "mu": scipy_params[1], "sigma": scipy_params[2]}
        if parameters["c"] < 0:
            scipy_params = scipy.stats.genpareto.fit(measurements.data)
            c0 = scipy_params[0]
            x0 = [c0, measurements.min, 1]
            b = ((-numpy.inf, -numpy.inf, 0), (numpy.inf, measurements.min, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
            parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
            parameters["mu"] = min(parameters["mu"], measurements.min - 1e-3)
            delta_sigma = parameters["c"] * (parameters["mu"] - measurements.max) - parameters["sigma"]
            parameters["sigma"] = parameters["sigma"] + delta_sigma + 1e-8
        return parameters

class GIBRAT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.gibrat.cdf(x, self.loc, self.scale)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.gibrat.pdf(x, self.loc, self.scale)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.gibrat.fit(measurements.data)
        parameters = {"loc": scipy_params[0], "scale": scipy_params[1]}
        return parameters

class GUMBEL_LEFT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        return 1 - numpy.exp(-numpy.exp(z(x)))
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        return (1 / self.sigma) * numpy.exp(z(x) - numpy.exp(z(x)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = mu - sigma * 0.5772156649
            parametric_variance = (sigma**2) * (numpy.pi**2) / 6
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)
        solution = scipy.optimize.fsolve(equations, (1, 1), measurements)
        parameters = {"mu": solution[0], "sigma": solution[1]}
        return parameters

class GUMBEL_RIGHT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        return numpy.exp(-numpy.exp(-z(x)))
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = mu + sigma * 0.5772156649
            parametric_variance = (sigma**2) * (numpy.pi**2) / 6
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (
                eq1,
                eq2,
            )
        solution = scipy.optimize.fsolve(equations, (1, 1), measurements)
        parameters = {"mu": solution[0], "sigma": solution[1]}
        return parameters

class HALF_NORMAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        result = scipy.special.erf(z(x) / numpy.sqrt(2))
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        result = (1 / self.sigma) * numpy.sqrt(2 / numpy.pi) * numpy.exp(-(z(x) ** 2) / 2)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        sigma = numpy.sqrt(measurements.variance / (1 - 2 / numpy.pi))
        mu = measurements.mean - sigma * numpy.sqrt(2) / numpy.sqrt(numpy.pi)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters

class HYPERBOLIC_SECANT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (2 / numpy.pi) * numpy.arctan(numpy.exp((z(x))))
    def pdf(self, x: float) -> float:
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (1 / numpy.cosh(z(x))) / (2 * self.sigma)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mu = measurements.mean
        sigma = numpy.sqrt(measurements.variance)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters

class INVERSE_GAMMA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
    def cdf(self, x: float) -> float:
        upper_inc_gamma = lambda a, x: scipy.special.gammaincc(a, x) * scipy.special.gamma(a)
        result = upper_inc_gamma(self.alpha, self.beta / x) / scipy.special.gamma(self.alpha)
        return result
    def pdf(self, x: float) -> float:
        return ((self.beta**self.alpha) * (x ** (-self.alpha - 1)) * numpy.exp(-(self.beta / x))) / scipy.special.gamma(self.alpha)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.invgamma.fit(measurements.data)
        parameters = {"alpha": scipy_params[0], "beta": scipy_params[2]}
        return parameters

class INVERSE_GAMMA_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        upper_inc_gamma = lambda a, x: scipy.special.gammaincc(a, x) * scipy.special.gamma(a)
        result = upper_inc_gamma(self.alpha, self.beta / (x - self.loc)) / scipy.special.gamma(self.alpha)
        return result
    def pdf(self, x: float) -> float:
        return ((self.beta**self.alpha) * ((x - self.loc) ** (-self.alpha - 1)) * numpy.exp(-(self.beta / (x - self.loc)))) / scipy.special.gamma(self.alpha)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha, beta, loc = initial_solution
            E = lambda k: (beta**k) / numpy.prod(numpy.array([(alpha - i) for i in range(1, k + 1)]))
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            return (eq1, eq2, eq3)
        try:
            bnds = ((0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
            x0 = (2, 1, measurements.mean)
            args = [measurements]
            solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1], "loc": solution.x[2]}
        except:
            scipy_params = scipy.stats.invgamma.fit(measurements.data)
            parameters = {"alpha": scipy_params[0], "loc": scipy_params[1], "beta": scipy_params[2]}
        return parameters

class INVERSE_GAUSSIAN:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.lambda_ = self.parameters["lambda"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.norm.cdf(numpy.sqrt(self.lambda_ / x) * ((x / self.mu) - 1)) + numpy.exp(2 * self.lambda_ / self.mu) * scipy.stats.norm.cdf(
            -numpy.sqrt(self.lambda_ / x) * ((x / self.mu) + 1)
        )
        return result
    def pdf(self, x: float) -> float:
        result = numpy.sqrt(self.lambda_ / (2 * numpy.pi * x**3)) * numpy.exp(-(self.lambda_ * (x - self.mu) ** 2) / (2 * self.mu**2 * x))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.lambda_ > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mu = measurements.mean
        lambda_ = mu**3 / measurements.variance
        parameters = {"mu": mu, "lambda": lambda_}
        return parameters

class INVERSE_GAUSSIAN_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.norm.cdf(numpy.sqrt(self.lambda_ / (x - self.loc)) * (((x - self.loc) / self.mu) - 1)) + numpy.exp(2 * self.lambda_ / self.mu) * scipy.stats.norm.cdf(
            -numpy.sqrt(self.lambda_ / (x - self.loc)) * (((x - self.loc) / self.mu) + 1)
        )
        return result
    def pdf(self, x: float) -> float:
        result = numpy.sqrt(self.lambda_ / (2 * numpy.pi * (x - self.loc) ** 3)) * numpy.exp(-(self.lambda_ * ((x - self.loc) - self.mu) ** 2) / (2 * self.mu**2 * (x - self.loc)))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.lambda_ > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mu = 3 * numpy.sqrt(measurements.variance / (measurements.skewness**2))
        lambda_ = mu**3 / measurements.variance
        loc = measurements.mean - mu
        parameters = {"mu": mu, "lambda": lambda_, "loc": loc}
        return parameters

class JOHNSON_SB:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * numpy.log(z(x) / (1 - z(x))))
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * numpy.sqrt(2 * numpy.pi) * z(x) * (1 - z(x)))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.log(z(x) / (1 - z(x)))) ** 2)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.johnsonsb.fit(measurements.data)
        parameters = {"xi": scipy_params[2], "lambda": scipy_params[3], "gamma": scipy_params[0], "delta": scipy_params[1]}
        return parameters

class JOHNSON_SU:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * numpy.arcsinh(z(x)))
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * numpy.sqrt(2 * numpy.pi) * numpy.sqrt(z(x) ** 2 + 1))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.arcsinh(z(x))) ** 2)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
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
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_kurtosis - measurements.kurtosis
            eq4 = parametric_median - measurements.median
            return (eq1, eq2, eq3, eq4)
        solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), measurements)
        parameters = {"xi": solution[0], "lambda": solution[1], "gamma": solution[2], "delta": solution[3]}
        return parameters

class KUMARASWAMY:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha_ = self.parameters["alpha"]
        self.beta_ = self.parameters["beta"]
        self.min_ = self.parameters["min"]
        self.max_ = self.parameters["max"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.min_) / (self.max_ - self.min_)
        result = 1 - ( 1 - z(x) ** self.alpha_) ** self.beta_
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.min_) / (self.max_ - self.min_)
        return (self.alpha_ * self.beta_) * (z(x) ** (self.alpha_ - 1)) * ((1 - z(x) ** self.alpha_) ** (self.beta_ - 1)) /  (self.max_ - self.min_)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha_ > 0
        v2 = self.beta_ > 0
        v3 = self.min_ < self.max_
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha_, beta_, min_, max_ = initial_solution
            E = lambda r: beta_ * scipy.special.gamma(1 + r / alpha_) * scipy.special.gamma(beta_) / scipy.special.gamma(1 + beta_ + r / alpha_)
            parametric_mean = E(1) * (max_ - min_) + min_
            parametric_variance = (E(2) - E(1) ** 2) * (max_ - min_) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4)-4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) /  ((E(2) - E(1) ** 2)) ** 2
            parametric_median = ((1 - 2 ** (-1 / beta_)) ** (1 / alpha_)) * (max_ - min_) + min_
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            eq4 = parametric_kurtosis  - measurements.kurtosis
            return (eq1, eq2, eq3, eq4)
        l = measurements.min - 3 * abs(measurements.min)
        bnds = ((0, 0, l, l), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1, 1)
        args = ([measurements])
        solution = scipy.optimize.least_squares(equations, x0, bounds = bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "min": solution.x[2], "max": solution.x[3]}
        return parameters

class LAPLACE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.b = self.parameters["b"]
    def cdf(self, x: float) -> float:
        return 0.5 + 0.5 * numpy.sign(x - self.mu) * (1 - numpy.exp(-abs(x - self.mu) / self.b))
    def pdf(self, x: float) -> float:
        return (1 / (2 * self.b)) * numpy.exp(-abs(x - self.mu) / self.b)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.b > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mu = measurements.mean
        b = numpy.sqrt(measurements.variance / 2)
        parameters = {"mu": mu, "b": b}
        return parameters

class LEVY:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.c = self.parameters["c"]
    def cdf(self, x: float) -> float:
        y = lambda x: numpy.sqrt(self.c / ((x - self.mu)))
        result = 2 - 2 * scipy.stats.norm.cdf(y(x))
        return result
    def pdf(self, x: float) -> float:
        result = numpy.sqrt(self.c / (2 * numpy.pi)) * numpy.exp(-self.c / (2 * (x - self.mu))) / ((x - self.mu) ** 1.5)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.levy.fit(measurements.data)
        parameters = {"mu": scipy_params[0], "c": scipy_params[1]}
        return parameters

class LOGGAMMA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        y = lambda x: (x - self.mu) / self.sigma
        result = scipy.special.gammainc(self.c, numpy.exp(y(x)))
        return result
    def pdf(self, x: float) -> float:
        y = lambda x: (x - self.mu) / self.sigma
        result = numpy.exp(self.c * y(x) - numpy.exp(y(x)) - scipy.special.gammaln(self.c)) / self.sigma
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        v2 = self.sigma > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
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
        args = (measurements.mean, measurements.variance, measurements.skewness)
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
        return parameters

class LOGISTIC:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: numpy.exp(-(t - self.mu) / self.sigma)
        result = 1 / (1 + z(x))
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: numpy.exp(-(t - self.mu) / self.sigma)
        result = z(x) / (self.sigma * (1 + z(x)) ** 2)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mu = measurements.mean
        sigma = numpy.sqrt(3 * measurements.variance / (numpy.pi**2))
        parameters = {"mu": mu, "sigma": sigma}
        return parameters

class LOGLOGISTIC:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
    def cdf(self, x: float) -> float:
        result = x**self.beta / (self.alpha**self.beta + x**self.beta)
        return result
    def pdf(self, x: float) -> float:
        result = self.beta / self.alpha * (x / self.alpha) ** (self.beta - 1) / ((1 + (x / self.alpha) ** self.beta) ** 2)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.fisk.fit(measurements.data)
        parameters = {"alpha": scipy_params[2], "beta": scipy_params[0]}
        return parameters

class LOGLOGISTIC_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = (x - self.loc) ** self.beta / (self.alpha**self.beta + (x - self.loc) ** self.beta)
        return result
    def pdf(self, x: float) -> float:
        result = self.beta / self.alpha * ((x - self.loc) / self.alpha) ** (self.beta - 1) / ((1 + ((x - self.loc) / self.alpha) ** self.beta) ** 2)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.fisk.fit(measurements.data)
        parameters = {"loc": scipy_params[1], "alpha": scipy_params[2], "beta": scipy_params[0]}
        return parameters

class LOGNORMAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.norm.cdf((numpy.log(x) - self.mu) / self.sigma)
        return result
    def pdf(self, x: float) -> float:
        return (1 / (x * self.sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((numpy.log(x) - self.mu) ** 2) / (2 * self.sigma**2)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.sigma > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mu = numpy.log(measurements.mean**2 / numpy.sqrt(measurements.mean**2 + measurements.variance))
        sigma = numpy.sqrt(numpy.log((measurements.mean**2 + measurements.variance) / (measurements.mean**2)))
        parameters = {"mu": mu, "sigma": sigma}
        return parameters

class MAXWELL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        z = lambda t: (x - self.loc) / self.alpha
        result = scipy.special.erf(z(x) / (numpy.sqrt(2))) - numpy.sqrt(2 / numpy.pi) * z(x) * numpy.exp(-z(x) ** 2 / 2)
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (x - self.loc) / self.alpha
        result = 1 / self.alpha * numpy.sqrt(2 / numpy.pi) * z(x) ** 2 * numpy.exp(-z(x) ** 2 / 2)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        alpha = numpy.sqrt(measurements.variance * numpy.pi / (3 * numpy.pi - 8))
        loc = measurements.mean - 2 * alpha * numpy.sqrt(2 / numpy.pi)
        parameters = {"alpha": alpha, "loc": loc}
        return parameters

class MOYAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        result = scipy.special.erfc(numpy.exp(-0.5 * z(x)) / numpy.sqrt(2))
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        result = numpy.exp(-0.5 * (z(x) + numpy.exp(-z(x)))) / (self.sigma * numpy.sqrt(2 * numpy.pi))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        sigma = numpy.sqrt(2 * measurements.variance / (numpy.pi * numpy.pi))
        mu = measurements.mean - sigma * (numpy.log(2) + 0.577215664901532)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters

class NAKAGAMI:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.m = self.parameters["m"]
        self.omega = self.parameters["omega"]
    def cdf(self, x: float) -> float:
        result = scipy.special.gammainc(self.m, (self.m / self.omega) * x**2)
        return result
    def pdf(self, x: float) -> float:
        return (2 * self.m**self.m) / (scipy.special.gamma(self.m) * self.omega**self.m) * (x ** (2 * self.m - 1) * numpy.exp(-(self.m / self.omega) * x**2))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.m >= 0.5
        v2 = self.omega > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        d = numpy.array(measurements.data)
        E_x2 = sum(d * d) / len(d)
        E_x4 = sum(d * d * d * d) / len(d)
        omega = E_x2
        m = E_x2**2 / (E_x4 - E_x2**2)
        parameters = {"m": m, "omega": omega}
        return parameters

class NON_CENTRAL_CHI_SQUARE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.ncx2.cdf(x, self.lambda_, self.n)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.ncx2.pdf(x, self.lambda_, self.n)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        v2 = self.n > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        lambda_ = measurements.variance / 2 - measurements.mean
        n = 2 * measurements.mean - measurements.variance / 2
        parameters = {"lambda": lambda_, "n": n}
        return parameters

class NON_CENTRAL_F:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.n1 = self.parameters["n1"]
        self.n2 = self.parameters["n2"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.ncf.cdf(x, self.n1, self.n2, self.lambda_)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.ncf.pdf(x, self.n1, self.n2, self.lambda_)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        v2 = self.n1 > 0
        v3 = self.n2 > 0
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            lambda_, n1, n2 = initial_solution
            E_1 = (n2 / n1) * ((n1 + lambda_) / (n2 - 2))
            E_2 = (n2 / n1) ** 2 * ((lambda_**2 + (2 * lambda_ + n1) * (n1 + 2)) / ((n2 - 2) * (n2 - 4)))
            E_3 = (n2 / n1) ** 3 * ((lambda_**3 + 3 * (n1 + 4) * lambda_**2 + (3 * lambda_ + n1) * (n1 + 2) * (n1 + 4)) / ((n2 - 2) * (n2 - 4) * (n2 - 6)))
            parametric_mean = E_1
            parametric_variance = E_2 - E_1**2
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            return (eq1, eq2, eq3)
        bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (measurements.mean, 1, 10)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"lambda": solution.x[0], "n1": solution.x[1], "n2": solution.x[2]}
        return parameters

class NON_CENTRAL_T_STUDENT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        z = lambda x: (x - self.loc) / self.scale
        result = scipy.stats.nct.cdf(z(x), self.n, self.lambda_)
        return result
    def pdf(self, x: float) -> float:
        z = lambda x: (x - self.loc) / self.scale
        result = scipy.stats.nct.pdf(z(x), self.n, self.lambda_) / self.scale
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.n > 0
        v2 = self.scale > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            lambda_, n, loc, scale = initial_solution
            E_1 = lambda_ * numpy.sqrt(n / 2) * scipy.special.gamma((n - 1) / 2) / scipy.special.gamma(n / 2)
            E_2 = (1 + lambda_**2) * n / (n - 2)
            E_3 = lambda_ * (3 + lambda_**2) * n**1.5 * numpy.sqrt(2) * scipy.special.gamma((n - 3) / 2) / (4 * scipy.special.gamma(n / 2))
            E_4 = (lambda_**4 + 6 * lambda_**2 + 3) * n**2 / ((n - 2) * (n - 4))
            parametric_mean = E_1 * scale + loc
            parametric_variance = (E_2 - E_1**2) * (scale**2)
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            parametric_kurtosis = (E_4 - 4 * E_1 * E_3 + 6 * E_1**2 * E_2 - 3 * E_1**4) / ((E_2 - E_1**2)) ** 2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            eq4 = parametric_kurtosis - measurements.kurtosis
            return (eq1, eq2, eq3, eq4)
        bnds = ((0, 0, 0, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 5, measurements.mean, 1)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"lambda": solution.x[0], "n": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        return parameters

class NORMAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.mu) / self.sigma
        result = 0.5 * (1 + scipy.special.erf(z(x) / numpy.sqrt(2)))
        return result
    def pdf(self, x: float) -> float:
        result = (1 / (self.sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((x - self.mu) ** 2) / (2 * self.sigma**2)))
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        mu = measurements.mean
        sigma = measurements.standard_deviation
        parameters = {"mu": mu, "sigma": sigma}
        return parameters

class PARETO_FIRST_KIND:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.pareto.cdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.pareto.pdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.pareto.fit(measurements.data)
        parameters = {"xm": scipy_params[2], "alpha": scipy_params[0], "loc": scipy_params[1]}
        return parameters

class PARETO_SECOND_KIND:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.lomax.cdf(x, self.alpha, scale=self.xm, loc = self.loc)
        return result
    def pdf(self, x: float) -> float:
        return (self.alpha * self.xm**self.alpha) / (((x - self.loc) + self.xm) ** (self.alpha + 1))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        m = measurements.mean
        v = measurements.variance
        loc = scipy.stats.lomax.fit(measurements.data)[1]
        xm = -((m - loc) * ((m - loc) ** 2 + v)) / ((m - loc) ** 2 - v)
        alpha = -(2 * v) / ((m - loc) ** 2 - v)
        parameters = {"xm": xm, "alpha": alpha, "loc": loc}
        return parameters

class PERT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
    def cdf(self, x: float) -> float:
        alpha1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        alpha2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)
        z = lambda t: (t - self.a) / (self.c - self.a)
        result = scipy.special.betainc(alpha1, alpha2, z(x))
        return result
    def pdf(self, x: float) -> float:
        alpha1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        alpha2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)
        return (x - self.a) ** (alpha1 - 1) * (self.c - x) ** (alpha2 - 1) / (scipy.special.beta(alpha1, alpha2) * (self.c - self.a) ** (alpha1 + alpha2 - 1))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.b
        v2 = self.b < self.c
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, b, c = initial_solution
            alpha1 = (4 * b + c - 5 * a) / (c - a)
            alpha2 = (5 * c - a - 4 * b) / (c - a)
            parametric_mean = (a + 4 * b + c) / 6
            parametric_variance = ((parametric_mean - a) * (c - parametric_mean)) / 7
            parametric_median = scipy.special.betaincinv(alpha1, alpha2, 0.5) * (c - a) + a
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq5 = parametric_median - measurements.median
            return (eq1, eq2, eq5)
        bnds = ((-numpy.inf, measurements.mean, measurements.min), (measurements.mean, numpy.inf, measurements.max))
        x0 = (measurements.min, measurements.mean, measurements.max)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"a": solution.x[0], "b": solution.x[1], "c": solution.x[2]}
        parameters["a"] = min(measurements.min - 1e-3, parameters["a"])
        parameters["c"] = max(measurements.max + 1e-3, parameters["c"])
        return parameters

class POWER_FUNCTION:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
    def cdf(self, x: float) -> float:
        return ((x - self.a) / (self.b - self.a)) ** self.alpha
    def pdf(self, x: float) -> float:
        return self.alpha * ((x - self.a) ** (self.alpha - 1)) / ((self.b - self.a) ** self.alpha)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.b > self.a
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha, a, b = initial_solution
            E1 = (a + b * alpha) / (1 + alpha)
            E2 = (2 * a**2 + 2 * a * b * alpha + b**2 * alpha * (1 + alpha)) / ((1 + alpha) * (2 + alpha))
            E3 = (6 * a**3 + 6 * a**2 * b * alpha + 3 * a * b**2 * alpha * (1 + alpha) + b**3 * alpha * (1 + alpha) * (2 + alpha)) / ((1 + alpha) * (2 + alpha) * (3 + alpha))
            parametric_mean = E1
            parametric_variance = E2 - E1**2
            parametric_skewness = (E3 - 3 * E2 * E1 + 2 * E1**3) / ((E2 - E1**2)) ** 1.5
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            return (eq1, eq2, eq3)
        bnds = ((0, -numpy.inf, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, measurements.max)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "a": solution.x[1], "b": measurements.max + 1e-3}
        return parameters

class RAYLEIGH:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.gamma = self.parameters["gamma"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.gamma) / self.sigma
        return 1 - numpy.exp(-0.5 * (z(x) ** 2))
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.gamma) / self.sigma
        return z(x) * numpy.exp(-0.5 * (z(x) ** 2)) / self.sigma
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        sigma = numpy.sqrt(measurements.variance * 2 / (4 - numpy.pi))
        gamma = measurements.mean - sigma * numpy.sqrt(numpy.pi / 2)
        parameters = {"gamma": gamma, "sigma": sigma}
        return parameters

class RECIPROCAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
    def cdf(self, x: float) -> float:
        return (numpy.log(x) - numpy.log(self.a)) / (numpy.log(self.b) - numpy.log(self.a))
    def pdf(self, x: float) -> float:
        return 1 / (x * (numpy.log(self.b) - numpy.log(self.a)))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        a = measurements.min - 1e-8
        b = measurements.max + 1e-8
        parameters = {"a": a, "b": b}
        return parameters

class RICE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.v = self.parameters["v"]
        self.sigma = self.parameters["sigma"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.rice.cdf(x, self.v / self.sigma, scale=self.sigma)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.rice.pdf(x, self.v / self.sigma, scale=self.sigma)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.v > 0
        v2 = self.sigma > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            v, sigma = initial_solution
            E = lambda k: sigma**k * 2 ** (k / 2) * scipy.special.gamma(1 + k / 2) * scipy.special.eval_laguerre(k / 2, -v * v / (2 * sigma * sigma))
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)
        bnds = ((0, 0), (numpy.inf, numpy.inf))
        x0 = (measurements.mean, numpy.sqrt(measurements.variance))
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"v": solution.x[0], "sigma": solution.x[1]}
        return parameters

class SEMICIRCULAR:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.loc = self.parameters["loc"]
        self.R = self.parameters["R"]
    def cdf(self, x: float) -> float:
        z = lambda t: t - self.loc
        result = 0.5 + z(x) * numpy.sqrt(self.R**2 - z(x) ** 2) / (numpy.pi * self.R**2) + numpy.arcsin(z(x) / self.R) / numpy.pi
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: t - self.loc
        result = 2 * numpy.sqrt(self.R**2 - z(x) ** 2) / (numpy.pi * self.R**2)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.R > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        loc = measurements.mean
        R = numpy.sqrt(4 * measurements.variance)
        d1 = (loc - R) - measurements.min
        d2 = measurements.max - (loc + R)
        delta = max(max(d1, 0), max(d2, 0)) + 1e-2
        R = R + delta
        parameters = {"loc": loc, "R": R}
        return parameters

class TRAPEZOIDAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.d = self.parameters["d"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.trapezoid.cdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.trapezoid.pdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.b
        v2 = self.b < self.c
        v3 = self.c < self.d
        return v1 and v2 and v3
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution, measurements, a, d):
            b, c = initial_solution
            parametric_mean = (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            parametric_variance = (1 / (6 * (d + c - a - b))) * ((d**4 - c**4) / (d - c) - (b**4 - a**4) / (b - a)) - (
                (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            ) ** 2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)
        a = measurements.min - 1e-3
        d = measurements.max + 1e-3
        x0 = [(d + a) * 0.25, (d + a) * 0.75]
        bnds = ((a, a), (d, d))
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=([measurements, a, d]))
        parameters = {"a": a, "b": solution.x[0], "c": solution.x[1], "d": d}
        return parameters

class TRIANGULAR:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
    def cdf(self, x: float) -> float:
        result = scipy.stats.triang.cdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result
    def pdf(self, x: float) -> float:
        result = scipy.stats.triang.pdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.c
        v2 = self.c < self.b
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        a = measurements.min - 1e-3
        b = measurements.max + 1e-3
        c = 3 * measurements.mean - a - b
        parameters = {"a": a, "b": b, "c": c}
        return parameters

class T_STUDENT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
    def cdf(self, x: float) -> float:
        result = scipy.special.betainc(self.df / 2, self.df / 2, (x + numpy.sqrt(x * x + self.df)) / (2 * numpy.sqrt(x * x + self.df)))
        return result
    def pdf(self, x: float) -> float:
        result = (1 / (numpy.sqrt(self.df) * scipy.special.beta(0.5, self.df / 2))) * (1 + x * x / self.df) ** (-(self.df + 1) / 2)
        return result
    def ppf(self, u):
        if u >= 0.5:
            result = numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
        else:
            result = -numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        df = 2 * measurements.variance / (measurements.variance - 1)
        parameters = {"df": df}
        return parameters

class T_STUDENT_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.special.betainc(self.df / 2, self.df / 2, (z(x) + numpy.sqrt(z(x) ** 2 + self.df)) / (2 * numpy.sqrt(z(x) ** 2 + self.df)))
        return result
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / (numpy.sqrt(self.df) * scipy.special.beta(0.5, self.df / 2))) * (1 + z(x) * z(x) / self.df) ** (-(self.df + 1) / 2)
        return result
    def ppf(self, u):
        if u >= 0.5:
            result = self.loc + self.scale * numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
        else:
            result = self.loc - self.scale * numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = self.scale > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.t.fit(measurements.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
        return parameters

class UNIFORM:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
    def cdf(self, x: float) -> float:
        return (x - self.a) / (self.b - self.a)
    def pdf(self, x: float) -> float:
        return 1 / (self.b - self.a)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        return v1
    def get_parameters(self, measurements) -> dict[str, float | int]:
        a = measurements.min - 1e-8
        b = measurements.max + 1e-8
        parameters = {"a": a, "b": b}
        return parameters

class WEIBULL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
    def cdf(self, x: float) -> float:
        return 1 - numpy.exp(-((x / self.beta) ** self.alpha))
    def pdf(self, x: float) -> float:
        return (self.alpha / self.beta) * ((x / self.beta) ** (self.alpha - 1)) * numpy.exp(-((x / self.beta) ** self.alpha))
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha, beta = initial_solution
            E = lambda k: (beta**k) * scipy.special.gamma(1 + k / alpha)
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)
        solution = scipy.optimize.fsolve(equations, (1, 1), measurements)
        parameters = {"alpha": solution[0], "beta": solution[1]}
        return parameters

class WEIBULL_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.beta
        return 1 - numpy.exp(-(z(x) ** self.alpha))
    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.beta
        return (self.alpha / self.beta) * (z(x) ** (self.alpha - 1)) * numpy.exp(-z(x) ** self.alpha)
    def get_num_parameters(self) -> int:
        return len(self.parameters)
    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2
    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha, beta, loc = initial_solution
            E = lambda k: (beta**k) * scipy.special.gamma(1 + k / alpha)
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            return (eq1, eq2, eq3)
        bnds = ((0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, measurements.mean)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "loc": solution.x[2], "beta": solution.x[1]}
        return parameters

class MEASUREMENTS_CONTINUOUS:
    def __init__(
        self,
        data: list[float | int],
        num_bins: int | None = None,
        confidence_level: float = 0.95,
    ):
        self.data = numpy.sort(data)
        self.length = len(self.data)
        self.min = self.data[0]
        self.max = self.data[-1]
        self.mean = numpy.mean(self.data)
        self.variance = numpy.var(self.data, ddof=1)
        self.standard_deviation = numpy.std(self.data, ddof=1)
        self.skewness = scipy.stats.moment(self.data, 3) / pow(self.standard_deviation, 3)
        self.kurtosis = scipy.stats.moment(self.data, 4) / pow(self.standard_deviation, 4)
        self.median = numpy.median(self.data)
        self.mode = self.calculate_mode()
        self.num_bins = num_bins if num_bins != None else self.num_bins_doane(self.data)
        self.absolutes_frequencies, self.bin_edges = numpy.histogram(self.data, self.num_bins)
        self.densities_frequencies, _ = numpy.histogram(self.data, self.num_bins, density=True)
        self.central_values = [(self.bin_edges[i] + self.bin_edges[i + 1]) / 2 for i in range(len(self.bin_edges) - 1)]
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.length - 1]])
        self.Sn_ks = (numpy.arange(self.length) + 1) / self.length
        self.critical_value_ad = self.ad_critical_value(confidence_level, self.length)
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
    def num_bins_doane(self, data):
        N = self.length
        skewness = scipy.stats.skew(data)
        sigma_g1 = numpy.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        num_bins = 1 + numpy.log2(N) + numpy.log2(1 + abs(skewness) / sigma_g1)
        return math.ceil(num_bins)
    def adinf(self, z):
        if z < 2:
            return (z**-0.5) * numpy.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
        return numpy.exp(-numpy.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))
    def errfix(self, n, x):
        def g1(t):
            return numpy.sqrt(t) * (1 - t) * (49 * t - 102)
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
    def AD(self, n, z):
        return self.adinf(z) + self.errfix(n, self.adinf(z))
    def ad_critical_value(self, q, n):
        f = lambda x: self.AD(n, x) - q
        root = scipy.optimize.newton(f, 2)
        return root
    def ad_p_value(self, n, z):
        return 1 - self.AD(n, z)

def test_chi_square_continuous(distribution, measurements, confidence_level=0.95):
    N = measurements.length
    freedom_degrees = measurements.num_bins - 1 - distribution.get_num_parameters()
    expected_values = N * (distribution.cdf(measurements.bin_edges[1:]) - distribution.cdf(measurements.bin_edges[:-1]))
    errors = ((measurements.absolutes_frequencies - expected_values) ** 2) / expected_values
    statistic_chi2 = numpy.sum(errors)
    critical_value = scipy.stats.chi2.ppf(confidence_level, freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {"test_statistic": statistic_chi2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_chi2

def test_kolmogorov_smirnov_continuous(distribution, measurements, confidence_level=0.95):
    N = measurements.length
    Fn = distribution.cdf(measurements.data)
    errors = numpy.abs(measurements.Sn_ks[measurements.idx_ks] - Fn[measurements.idx_ks])
    statistic_ks = numpy.max(errors)
    critical_value = scipy.stats.kstwo.ppf(confidence_level, N)
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, N)
    rejected = statistic_ks >= critical_value
    result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ks

def test_anderson_darling_continuous(distribution, measurements, confidence_level=0.95):
    N = measurements.length
    S = numpy.sum(((2 * (numpy.arange(N) + 1) - 1) / N) * (numpy.log(distribution.cdf(measurements.data)) + numpy.log(1 - distribution.cdf(measurements.data[::-1]))))
    A2 = -N - S
    critical_value = measurements.critical_value_ad
    p_value = measurements.ad_p_value(N, A2)
    rejected = A2 >= critical_value
    result_test_ad = {"test_statistic": A2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ad

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
            processing_results = list(concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs).map(self.process_distribution, _ALL_CONTINUOUS_DISTRIBUTIONS))
        processing_results = [r for r in processing_results if r is not None]
        sorted_results_sse = {distribution: results for distribution, results in sorted(processing_results, key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
        not_rejected_results = {distribution: results for distribution, results in sorted_results_sse.items() if results["n_test_passed"] > 0}
        return sorted_results_sse, not_rejected_results


if __name__ == "__main__":
    path = "../../continuous/data/data_beta.txt"
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    phitter_continuous = PHITTER_CONTINUOUS(data, num_bins=None, confidence_level=0.95, minimum_sse=100)
    sorted_results_sse, not_rejected_results = phitter_continuous.fit(n_jobs=5)

    for distribution, results in not_rejected_results.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
