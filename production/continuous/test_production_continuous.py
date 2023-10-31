import math
import numpy
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.special as sc
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
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / (self.scale * z(x) * z(x) * scipy.stats.norm.cdf(self.alpha) * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (self.alpha - 1 / z(x)) ** 2)
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
        return 2 * math.asin(math.sqrt(z(x))) / math.pi

    def pdf(self, x: float) -> float:
        return 1 / (math.pi * math.sqrt((x - self.a) * (self.b - x)))

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
        result = 1 - sc.gammainc(1.5, self.chi * self.chi * (1 - z(x) ** 2) / 2) / sc.gammainc(1.5, self.chi * self.chi / 2)
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        result = (1 / self.scale) * ((self.chi**3) / (math.sqrt(2 * math.pi) * Ψ(self.chi))) * z(x) * math.sqrt(1 - z(x) * z(x)) * math.exp(-0.5 * self.chi**2 * (1 - z(x) * z(x)))
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
        self.alpha_ = self.parameters["alpha"]
        self.beta_ = self.parameters["beta"]
        self.A = self.parameters["A"]
        self.B_ = self.parameters["B"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.A) / (self.B_ - self.A)
        result = sc.betainc(self.alpha_, self.beta_, z(x))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.A) / (self.B_ - self.A)
        return (1 / (self.B_ - self.A)) * (math.gamma(self.alpha_ + self.beta_) / (math.gamma(self.alpha_) * math.gamma(self.beta_))) * (z(x) ** (self.alpha_ - 1)) * ((1 - z(x)) ** (self.beta_ - 1))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha_ > 0
        v2 = self.beta_ > 0
        v3 = self.A < self.B_
        return v1 and v2 and v3

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            alpha_, beta_, A, B_ = initial_solution
            parametric_mean = A + (alpha_ / (alpha_ + beta_)) * (B_ - A)
            parametric_variance = ((alpha_ * beta_) / ((alpha_ + beta_) ** 2 * (alpha_ + beta_ + 1))) * (B_ - A) ** 2
            parametric_skewness = 2 * ((beta_ - alpha_) / (alpha_ + beta_ + 2)) * math.sqrt((alpha_ + beta_ + 1) / (alpha_ * beta_))
            parametric_kurtosis = 3 * (
                ((alpha_ + beta_ + 1) * (2 * (alpha_ + beta_) ** 2 + (alpha_ * beta_) * (alpha_ + beta_ - 6))) / ((alpha_ * beta_) * (alpha_ + beta_ + 2) * (alpha_ + beta_ + 3))
            )
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
            α, β = initial_solution
            parametric_mean = α / (β - 1)
            parametric_variance = α * (α + β - 1) / ((β - 1) ** 2 * (β - 2))
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
            α, β, scale, loc = initial_solution
            parametric_mean = scale * α / (β - 1) + loc
            parametric_variance = (scale**2) * α * (α + β - 1) / ((β - 1) ** 2 * (β - 2))
            parametric_median = loc + scale * scipy.stats.beta.ppf(0.5, α, β) / (1 - scipy.stats.beta.ppf(0.5, α, β))
            parametric_mode = scale * (α - 1) / (β + 1) + loc
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
        result = math.log(1 + self.c * (x - self.min_) / (self.max_ - self.min_)) / math.log(self.c + 1)
        return result

    def pdf(self, x: float) -> float:
        result = self.c / ((self.c * (x - self.min_) + self.max_ - self.min_) * math.log(self.c + 1))
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
            parametric_mean = (c * (_max - _min) + math.log(c + 1) * (_min * (c + 1) - _max)) / (c * math.log(c + 1))
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
        result = 1 - ((1 + (x / self.A) ** (self.B)) ** (-self.C))
        return result

    def pdf(self, x: float) -> float:
        result = ((self.B * self.C) / self.A) * ((x / self.A) ** (self.B - 1)) * ((1 + (x / self.A) ** (self.B)) ** (-self.C - 1))
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            A, B, C = initial_solution
            miu = lambda r: (A**r) * C * sc.beta((B * C - r) / B, (B + r) / B)
            parametric_mean = miu(1)
            parametric_median = A * ((2 ** (1 / C)) - 1) ** (1 / B)
            parametric_mode = A * ((B - 1) / (B * C + 1)) ** (1 / B)
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_median - measurements.median
            eq3 = parametric_mode - measurements.mode
            return (eq1, eq2, eq3)

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
        result = 1 - ((1 + ((x - self.loc) / self.A) ** (self.B)) ** (-self.C))
        return result

    def pdf(self, x: float) -> float:
        result = ((self.B * self.C) / self.A) * (((x - self.loc) / self.A) ** (self.B - 1)) * ((1 + ((x - self.loc) / self.A) ** (self.B)) ** (-self.C - 1))
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            A, B, C, loc = initial_solution
            miu = lambda r: (A**r) * C * sc.beta((B * C - r) / B, (B + r) / B)
            parametric_mean = miu(1) + loc
            parametric_variance = -(miu(1) ** 2) + miu(2)
            parametric_kurtosis = -3 * miu(1) ** 4 + 6 * miu(1) ** 2 * miu(2) - 4 * miu(1) * miu(3) + miu(4)
            parametric_mode = A * ((B - 1) / (B * C + 1)) ** (1 / B) + loc
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_kurtosis - measurements.kurtosis
            eq4 = parametric_mode - measurements.mode
            return (eq1, eq2, eq3, eq4)

        scipy_params = scipy.stats.burr12.fit(measurements.data)
        parameters = {"A": scipy_params[3], "B": scipy_params[0], "C": scipy_params[1], "loc": scipy_params[2]}
        return parameters


class CAUCHY:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.x0 = self.parameters["x0"]
        self.gamma = self.parameters["gamma"]

    def cdf(self, x: float) -> float:
        return (1 / math.pi) * math.atan(((x - self.x0) / self.gamma)) + (1 / 2)

    def pdf(self, x: float) -> float:
        return 1 / (math.pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))

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
        result = sc.gammainc(self.df / 2, x / 2)
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
        result = sc.gammainc(self.df / 2, z(x) / 2)
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * (1 / (2 ** (self.df / 2) * math.gamma(self.df / 2))) * (z(x) ** ((self.df / 2) - 1)) * (math.exp(-z(x) / 2))
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
            miu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = miu(1)
            parametric_variance = -(miu(1) ** 2) + miu(2)
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
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
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
            miu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = miu(1) + loc
            parametric_variance = -(miu(1) ** 2) + miu(2)
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
        result = sc.gammainc(self.k, x / self.beta)
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
        β = measurements.variance / measurements.mean
        parameters = {"k": k, "beta": β}
        return parameters


class ERLANG_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    def cdf(self, x: float) -> float:
        result = sc.gammainc(self.k, (x - self.loc) / self.beta)
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
        β = math.sqrt(measurements.variance / ((2 / measurements.skewness) ** 2))
        loc = measurements.mean - ((2 / measurements.skewness) ** 2) * β
        parameters = {"k": k, "beta": β, "loc": loc}
        return parameters


class ERROR_FUNCTION:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.h = self.parameters["h"]

    def cdf(self, x: float) -> float:
        return scipy.stats.norm.cdf((2**0.5) * self.h * x)

    def pdf(self, x: float) -> float:
        return self.h * math.exp(-((self.h * x) ** 2)) / math.sqrt(math.pi)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.h > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        h = math.sqrt(1 / (2 * measurements.variance))
        parameters = {"h": h}
        return parameters


class EXPONENTIAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]

    def cdf(self, x: float) -> float:
        return 1 - math.exp(-self.lambda_ * x)

    def pdf(self, x: float) -> float:
        return self.lambda_ * math.exp(-self.lambda_ * x)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        λ = 1 / measurements.mean
        parameters = {"lambda": λ}
        return parameters


class EXPONENTIAL_2P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]

    def cdf(self, x: float) -> float:
        return 1 - math.exp(-self.lambda_ * (x - self.loc))

    def pdf(self, x: float) -> float:
        return self.lambda_ * math.exp(-self.lambda_ * (x - self.loc))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        λ = (1 - math.log(2)) / (measurements.mean - measurements.median)
        loc = measurements.min - 1e-4
        parameters = {"lambda": λ, "loc": loc}
        return parameters


class F:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]

    def cdf(self, x: float) -> float:
        return sc.betainc(self.df1 / 2, self.df2 / 2, x * self.df1 / (self.df1 * x + self.df2))

    def pdf(self, x: float) -> float:
        return (1 / sc.beta(self.df1 / 2, self.df2 / 2)) * ((self.df1 / self.df2) ** (self.df1 / 2)) * (x ** (self.df1 / 2 - 1)) * ((1 + x * self.df1 / self.df2) ** (-1 * (self.df1 + self.df2) / 2))

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
        z = lambda t: math.sqrt((t - self.loc) / self.scale)
        result = scipy.stats.norm.cdf((z(x) - 1 / z(x)) / (self.gamma))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: math.sqrt((t - self.loc) / self.scale)
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
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z1 = lambda t: (t + self.miu) / self.sigma
        z2 = lambda t: (t - self.miu) / self.sigma
        result = 0.5 * (sc.erf(z1(x) / math.sqrt(2)) + sc.erf(z2(x) / math.sqrt(2)))
        return result

    def pdf(self, x: float) -> float:
        result = math.sqrt(2 / (math.pi * self.sigma**2)) * math.exp(-(x**2 + self.miu**2) / (2 * self.sigma**2)) * math.cosh(self.miu * x / (self.sigma**2))
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            miu, sigma = initial_solution
            parametric_mean = sigma * math.sqrt(2 / math.pi) * math.exp(-(miu**2) / (2 * sigma**2)) + miu * sc.erf(miu / math.sqrt(2 * sigma**2))
            parametric_variance = miu**2 + sigma**2 - parametric_mean**2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)

        x0 = [measurements.mean, measurements.standard_deviation]
        b = ((-numpy.inf, 0), (numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
        parameters = {"miu": solution.x[0], "sigma": solution.x[1]}
        return parameters


class FRECHET:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * self.alpha * z(x) ** (-self.alpha - 1) * math.exp(-z(x) ** -self.alpha)
        return result

    def pdf(self, x: float) -> float:
        return (self.alpha / self.scale) * (((x - self.loc) / self.scale) ** (-1 - self.alpha)) * math.exp(-(((x - self.loc) / self.scale) ** (-self.alpha)))

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
        z = lambda t: (t - self.loc) / self.scale
        return sc.betainc(self.df1 / 2, self.df2 / 2, z(x) * self.df1 / (self.df1 * z(x) + self.df2))

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        return (
            (1 / self.scale)
            * (1 / sc.beta(self.df1 / 2, self.df2 / 2))
            * ((self.df1 / self.df2) ** (self.df1 / 2))
            * (z(x) ** (self.df1 / 2 - 1))
            * ((1 + z(x) * self.df1 / self.df2) ** (-1 * (self.df1 + self.df2) / 2))
        )

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
            E = lambda k: (df2 / df1) ** k * (math.gamma(df1 / 2 + k) * math.gamma(df2 / 2 - k)) / (math.gamma(df1 / 2) * math.gamma(df2 / 2))
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
        result = sc.gammainc(self.alpha, x / self.beta)
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
        result = sc.gammainc(self.alpha, (x - self.loc) / self.beta)
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
        α = (2 / measurements.skewness) ** 2
        β = math.sqrt(measurements.variance / α)
        loc = measurements.mean - α * β
        parameters = {"alpha": α, "beta": β, "loc": loc}
        return parameters


class GENERALIZED_EXTREME_VALUE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.ξ = self.parameters["ξ"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        if self.ξ == 0:
            return math.exp(-math.exp(-z(x)))
        else:
            return math.exp(-((1 + self.ξ * z(x)) ** (-1 / self.ξ)))

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        if self.ξ == 0:
            return (1 / self.sigma) * math.exp(-z(x) - math.exp(-z(x)))
        else:
            return (1 / self.sigma) * math.exp(-((1 + self.ξ * z(x)) ** (-1 / self.ξ))) * (1 + self.ξ * z(x)) ** (-1 - 1 / self.ξ)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.genextreme.fit(measurements.data)
        parameters = {"ξ": -scipy_params[0], "miu": scipy_params[1], "sigma": scipy_params[2]}
        return parameters


class GENERALIZED_GAMMA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]

    def cdf(self, x: float) -> float:
        result = sc.gammainc(self.d / self.p, (x / self.a) ** self.p)
        return result

    def pdf(self, x: float) -> float:
        return (self.p / (self.a**self.d)) * (x ** (self.d - 1)) * math.exp(-((x / self.a) ** self.p)) / math.gamma(self.d / self.p)

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
            E = lambda r: a**r * (math.gamma((d + r) / p) / math.gamma(d / p))
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
            parameters = {"a": scipy_params[0], "c": scipy_params[1], "miu": scipy_params[2], "sigma": scipy_params[3]}
        return parameters


class GENERALIZED_GAMMA_4P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]

    def cdf(self, x: float) -> float:
        result = sc.gammainc(self.d / self.p, ((x - self.loc) / self.a) ** self.p)
        return result

    def pdf(self, x: float) -> float:
        return (self.p / (self.a**self.d)) * ((x - self.loc) ** (self.d - 1)) * math.exp(-(((x - self.loc) / self.a) ** self.p)) / math.gamma(self.d / self.p)

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
            E = lambda r: a**r * (math.gamma((d + r) / p) / math.gamma(d / p))
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
        return 1 / ((1 + math.exp(-z(x))) ** self.c)

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        return (self.c / self.scale) * math.exp(-z(x)) * ((1 + math.exp(-z(x))) ** (-self.c - 1))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        v2 = self.c > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            c, loc, scale = initial_solution
            parametric_mean = loc + scale * (0.57721 + sc.digamma(c))
            parametric_variance = scale**2 * (math.pi**2 / 6 + sc.polygamma(1, c))
            parametric_median = loc + scale * (-math.log(0.5 ** (-1 / c) - 1))
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
        self.miu = self.parameters["miu"]
        self.alpha = self.parameters["alpha"]

    def cdf(self, x: float) -> float:
        return 0.5 + (numpy.sign(x - self.miu) / 2) * sc.gammainc(1 / self.beta, abs((x - self.miu) / self.alpha) ** self.beta)

    def pdf(self, x: float) -> float:
        return self.beta / (2 * self.alpha * math.gamma(1 / self.beta)) * math.exp(-((abs(x - self.miu) / self.alpha) ** self.beta))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.gennorm.fit(measurements.data)
        parameters = {"beta": scipy_params[0], "miu": scipy_params[1], "alpha": scipy_params[2]}
        return parameters


class GENERALIZED_PARETO:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        result = 1 - (1 + self.c * z(x)) ** (-1 / self.c)
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        result = (1 / self.sigma) * (1 + self.c * z(x)) ** (-1 / self.c - 1)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            c, miu, sigma = initial_solution
            parametric_mean = miu + sigma / (1 - c)
            parametric_variance = sigma * sigma / ((1 - c) * (1 - c) * (1 - 2 * c))
            parametric_median = miu + sigma * (2**c - 1) / c
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_median - numpy.percentile(measurements.data, 50)
            return (eq1, eq2, eq3)

        scipy_params = scipy.stats.genpareto.fit(measurements.data)
        parameters = {"c": scipy_params[0], "miu": scipy_params[1], "sigma": scipy_params[2]}
        if parameters["c"] < 0:
            scipy_params = scipy.stats.genpareto.fit(measurements.data)
            c0 = scipy_params[0]
            x0 = [c0, measurements.min, 1]
            b = ((-numpy.inf, -numpy.inf, 0), (numpy.inf, measurements.min, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0, bounds=b, args=([measurements]))
            parameters = {"c": solution.x[0], "miu": solution.x[1], "sigma": solution.x[2]}
            parameters["miu"] = min(parameters["miu"], measurements.min - 1e-3)
            delta_sigma = parameters["c"] * (parameters["miu"] - measurements.max) - parameters["sigma"]
            parameters["sigma"] = parameters["sigma"] + delta_sigma + 1e-8
        return parameters


class GIBRAT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = 0.5 * (1 + sc.erf(math.log(z(x)) / math.sqrt(2)))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = 1 / (self.scale * z(x) * math.sqrt(2 * math.pi)) * math.exp(-0.5 * math.log(z(x)) ** 2)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.gilbrat.fit(measurements.data)
        parameters = {"loc": scipy_params[0], "scale": scipy_params[1]}
        return parameters


class GUMBEL_LEFT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        return 1 - math.exp(-math.exp(z(x)))

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        return (1 / self.sigma) * math.exp(z(x) - math.exp(z(x)))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            miu, sigma = initial_solution
            parametric_mean = miu - sigma * 0.5772156649
            parametric_variance = (sigma**2) * (math.pi**2) / 6
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)

        solution = scipy.optimize.fsolve(equations, (1, 1), measurements)
        parameters = {"miu": solution[0], "sigma": solution[1]}
        return parameters


class GUMBEL_RIGHT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        return math.exp(-math.exp(-z(x)))

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        return (1 / self.sigma) * math.exp(-z(x) - math.exp(-z(x)))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            miu, sigma = initial_solution
            parametric_mean = miu + sigma * 0.5772156649
            parametric_variance = (sigma**2) * (math.pi**2) / 6
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (
                eq1,
                eq2,
            )

        solution = scipy.optimize.fsolve(equations, (1, 1), measurements)
        parameters = {"miu": solution[0], "sigma": solution[1]}
        return parameters


class HALF_NORMAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        result = sc.erf(z(x) / math.sqrt(2))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        result = (1 / self.sigma) * math.sqrt(2 / math.pi) * math.exp(-(z(x) ** 2) / 2)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        σ = math.sqrt(measurements.variance / (1 - 2 / math.pi))
        μ = measurements.mean - σ * math.sqrt(2) / math.sqrt(math.pi)
        parameters = {"miu": μ, "sigma": σ}
        return parameters


class HYPERBOLIC_SECANT:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: math.pi * (t - self.miu) / (2 * self.sigma)
        return (2 / math.pi) * math.atan(math.exp((z(x))))

    def pdf(self, x: float) -> float:
        z = lambda t: math.pi * (t - self.miu) / (2 * self.sigma)
        return (1 / math.cosh(z(x))) / (2 * self.sigma)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        miu = measurements.mean
        sigma = math.sqrt(measurements.variance)
        parameters = {"miu": miu, "sigma": sigma}
        return parameters


class INVERSE_GAMMA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    def cdf(self, x: float) -> float:
        upper_inc_gamma = lambda a, x: sc.gammaincc(a, x) * math.gamma(a)
        result = upper_inc_gamma(self.alpha, self.beta / x) / math.gamma(self.alpha)
        return result

    def pdf(self, x: float) -> float:
        return ((self.beta**self.alpha) * (x ** (-self.alpha - 1)) * math.exp(-(self.beta / x))) / math.gamma(self.alpha)

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
        upper_inc_gamma = lambda a, x: sc.gammaincc(a, x) * math.gamma(a)
        result = upper_inc_gamma(self.alpha, self.beta / (x - self.loc)) / math.gamma(self.alpha)
        return result

    def pdf(self, x: float) -> float:
        return ((self.beta**self.alpha) * ((x - self.loc) ** (-self.alpha - 1)) * math.exp(-(self.beta / (x - self.loc)))) / math.gamma(self.alpha)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            α, β, loc = initial_solution
            E = lambda k: (β**k) / numpy.prod(numpy.array([(α - i) for i in range(1, k + 1)]))
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
            parameters = {"alpha": scipy_params[0], "beta": scipy_params[2], "loc": scipy_params[1]}
        return parameters


class INVERSE_GAUSSIAN:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.lambda_ = self.parameters["lambda"]

    def cdf(self, x: float) -> float:
        result = scipy.stats.norm.cdf(math.sqrt(self.lambda_ / x) * ((x / self.miu) - 1)) + math.exp(2 * self.lambda_ / self.miu) * scipy.stats.norm.cdf(
            -math.sqrt(self.lambda_ / x) * ((x / self.miu) + 1)
        )
        return result

    def pdf(self, x: float) -> float:
        result = math.sqrt(self.lambda_ / (2 * math.pi * x**3)) * math.exp(-(self.lambda_ * (x - self.miu) ** 2) / (2 * self.miu**2 * x))
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.miu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        μ = measurements.mean
        λ = μ**3 / measurements.variance
        parameters = {"miu": μ, "lambda": λ}
        return parameters


class INVERSE_GAUSSIAN_3P:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]

    def cdf(self, x: float) -> float:
        result = scipy.stats.norm.cdf(math.sqrt(self.lambda_ / (x - self.loc)) * (((x - self.loc) / self.miu) - 1)) + math.exp(2 * self.lambda_ / self.miu) * scipy.stats.norm.cdf(
            -math.sqrt(self.lambda_ / (x - self.loc)) * (((x - self.loc) / self.miu) + 1)
        )
        return result

    def pdf(self, x: float) -> float:
        result = math.sqrt(self.lambda_ / (2 * math.pi * (x - self.loc) ** 3)) * math.exp(-(self.lambda_ * ((x - self.loc) - self.miu) ** 2) / (2 * self.miu**2 * (x - self.loc)))
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.miu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        μ = 3 * math.sqrt(measurements.variance / (measurements.skewness**2))
        λ = μ**3 / measurements.variance
        loc = measurements.mean - μ
        parameters = {"miu": μ, "lambda": λ, "loc": loc}
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
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * math.log(z(x) / (1 - z(x))))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * math.sqrt(2 * math.pi) * z(x) * (1 - z(x)))) * math.exp(-(1 / 2) * (self.gamma_ + self.delta_ * math.log(z(x) / (1 - z(x)))) ** 2)

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
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * math.asinh(z(x)))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * math.sqrt(2 * math.pi) * math.sqrt(z(x) ** 2 + 1))) * math.exp(-(1 / 2) * (self.gamma_ + self.delta_ * math.asinh(z(x))) ** 2)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            xi_, lambda_, gamma_, delta_ = initial_solution
            w = math.exp(1 / delta_**2)
            omega = gamma_ / delta_
            A = w**2 * (w**4 + 2 * w**3 + 3 * w**2 - 3) * math.cosh(4 * omega)
            B = 4 * w**2 * (w + 2) * math.cosh(2 * omega)
            C = 3 * (2 * w + 1)
            parametric_mean = xi_ - lambda_ * math.sqrt(w) * math.sinh(omega)
            parametric_variance = (lambda_**2 / 2) * (w - 1) * (w * math.cosh(2 * omega) + 1)
            parametric_kurtosis = ((lambda_**4) * (w - 1) ** 2 * (A + B + C)) / (8 * math.sqrt(parametric_variance) ** 4)
            parametric_median = xi_ + lambda_ * math.sinh(-omega)
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
        result = 1 - (1 - z(x) ** self.alpha_) ** self.beta_
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.min_) / (self.max_ - self.min_)
        return (self.alpha_ * self.beta_) * (z(x) ** (self.alpha_ - 1)) * ((1 - z(x) ** self.alpha_) ** (self.beta_ - 1)) / (self.max_ - self.min_)

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
            E = lambda r: beta_ * math.gamma(1 + r / alpha_) * math.gamma(beta_) / math.gamma(1 + beta_ + r / alpha_)
            parametric_mean = E(1) * (max_ - min_) + min_
            parametric_variance = (E(2) - E(1) ** 2) * (max_ - min_) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            parametric_median = ((1 - 2 ** (-1 / beta_)) ** (1 / alpha_)) * (max_ - min_) + min_
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            eq3 = parametric_skewness - measurements.skewness
            eq4 = parametric_kurtosis - measurements.kurtosis
            return (eq1, eq2, eq3, eq4)

        l = measurements.min - 3 * abs(measurements.min)
        bnds = ((0, 0, l, l), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1, 1)
        args = [measurements]
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "min": solution.x[2], "max": solution.x[3]}
        return parameters


class LAPLACE:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.b = self.parameters["b"]

    def cdf(self, x: float) -> float:
        return 0.5 + 0.5 * numpy.sign(x - self.miu) * (1 - math.exp(-abs(x - self.miu) / self.b))

    def pdf(self, x: float) -> float:
        return (1 / (2 * self.b)) * math.exp(-abs(x - self.miu) / self.b)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        miu = measurements.mean
        b = math.sqrt(measurements.variance / 2)
        parameters = {"miu": miu, "b": b}
        return parameters


class LEVY:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.c = self.parameters["c"]

    def cdf(self, x: float) -> float:
        y = lambda x: math.sqrt(self.c / ((x - self.miu)))
        result = 2 - 2 * scipy.stats.norm.cdf(y(x))
        return result

    def pdf(self, x: float) -> float:
        result = math.sqrt(self.c / (2 * math.pi)) * math.exp(-self.c / (2 * (x - self.miu))) / ((x - self.miu) ** 1.5)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        scipy_params = scipy.stats.levy.fit(measurements.data)
        parameters = {"miu": scipy_params[0], "c": scipy_params[1]}
        return parameters


class LOGGAMMA:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.c = self.parameters["c"]
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        y = lambda x: (x - self.miu) / self.sigma
        result = sc.gammainc(self.c, math.exp(y(x)))
        return result

    def pdf(self, x: float) -> float:
        y = lambda x: (x - self.miu) / self.sigma
        result = math.exp(self.c * y(x) - math.exp(y(x)) - sc.gammaln(self.c)) / self.sigma
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution, data_mean, data_variance, data_skewness):
            c, miu, sigma = initial_solution
            parametric_mean = sc.digamma(c) * sigma + miu
            parametric_variance = sc.polygamma(1, c) * (sigma**2)
            parametric_skewness = sc.polygamma(2, c) / (sc.polygamma(1, c) ** 1.5)
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            eq3 = parametric_skewness - data_skewness
            return (eq1, eq2, eq3)

        bnds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1)
        args = (measurements.mean, measurements.variance, measurements.skewness)
        solution = scipy.optimize.least_squares(equations, x0, bounds=bnds, args=args)
        parameters = {"c": solution.x[0], "miu": solution.x[1], "sigma": solution.x[2]}
        return parameters


class LOGISTIC:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: math.exp(-(t - self.miu) / self.sigma)
        result = 1 / (1 + z(x))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: math.exp(-(t - self.miu) / self.sigma)
        result = z(x) / (self.sigma * (1 + z(x)) ** 2)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        μ = measurements.mean
        σ = math.sqrt(3 * measurements.variance / (math.pi**2))
        parameters = {"miu": μ, "sigma": σ}
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
        parameters = {"alpha": scipy_params[2], "beta": scipy_params[0], "loc": scipy_params[1]}
        return parameters


class LOGNORMAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        result = scipy.stats.norm.cdf((math.log(x) - self.miu) / self.sigma)
        return result

    def pdf(self, x: float) -> float:
        return (1 / (x * self.sigma * math.sqrt(2 * math.pi))) * math.exp(-(((math.log(x) - self.miu) ** 2) / (2 * self.sigma**2)))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.miu > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        μ = math.log(measurements.mean**2 / math.sqrt(measurements.mean**2 + measurements.variance))
        σ = math.sqrt(math.log((measurements.mean**2 + measurements.variance) / (measurements.mean**2)))
        parameters = {"miu": μ, "sigma": σ}
        return parameters


class MAXWELL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    def cdf(self, x: float) -> float:
        z = lambda t: (x - self.loc) / self.alpha
        result = sc.erf(z(x) / (math.sqrt(2))) - math.sqrt(2 / math.pi) * z(x) * math.exp(-z(x) ** 2 / 2)
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (x - self.loc) / self.alpha
        result = 1 / self.alpha * math.sqrt(2 / math.pi) * z(x) ** 2 * math.exp(-z(x) ** 2 / 2)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        alpha = math.sqrt(measurements.variance * math.pi / (3 * math.pi - 8))
        loc = measurements.mean - 2 * alpha * math.sqrt(2 / math.pi)
        parameters = {"alpha": alpha, "loc": loc}
        return parameters


class MOYAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        result = sc.erfc(math.exp(-0.5 * z(x)) / math.sqrt(2))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        result = math.exp(-0.5 * (z(x) + math.exp(-z(x)))) / (self.sigma * math.sqrt(2 * math.pi))
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        σ = math.sqrt(2 * measurements.variance / (math.pi * math.pi))
        μ = measurements.mean - σ * (math.log(2) + 0.577215664901532)
        parameters = {"miu": μ, "sigma": σ}
        return parameters


class NAKAGAMI:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.m = self.parameters["m"]
        self.omega = self.parameters["omega"]

    def cdf(self, x: float) -> float:
        result = sc.gammainc(self.m, (self.m / self.omega) * x**2)
        return result

    def pdf(self, x: float) -> float:
        return (2 * self.m**self.m) / (math.gamma(self.m) * self.omega**self.m) * (x ** (2 * self.m - 1) * math.exp(-(self.m / self.omega) * x**2))

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
        def Q(M: float, a: float, b: float) -> float:
            k = 1 - M
            x = (a / b) ** k * sc.iv(k, a * b)
            acum = 0
            while x > 1e-20:
                acum += x
                k += 1
                x = (a / b) ** k * sc.iv(k, a * b)
            res = acum * math.exp(-(a**2 + b**2) / 2)
            return res

        result = 1 - Q(self.n / 2, math.sqrt(self.lambda_), math.sqrt(x))
        return result

    def pdf(self, x: float) -> float:
        result = 1 / 2 * math.exp(-(x + self.lambda_) / 2) * (x / self.lambda_) ** ((self.n - 2) / 4) * sc.iv((self.n - 2) / 2, math.sqrt(self.lambda_ * x))
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
        result = sc.ncfdtr(self.n1, self.n2, self.lambda_, x)
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
        result = sc.nctdtr(self.n, self.lambda_, z(x))
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
            E_1 = lambda_ * math.sqrt(n / 2) * math.gamma((n - 1) / 2) / math.gamma(n / 2)
            E_2 = (1 + lambda_**2) * n / (n - 2)
            E_3 = lambda_ * (3 + lambda_**2) * n**1.5 * math.sqrt(2) * math.gamma((n - 3) / 2) / (4 * math.gamma(n / 2))
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
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        z = lambda t: (t - self.miu) / self.sigma
        result = 0.5 * (1 + sc.erf(z(x) / math.sqrt(2)))
        return result

    def pdf(self, x: float) -> float:
        result = (1 / (self.sigma * math.sqrt(2 * math.pi))) * math.exp(-(((x - self.miu) ** 2) / (2 * self.sigma**2)))
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        μ = measurements.mean
        σ = measurements.standard_deviation
        parameters = {"miu": μ, "sigma": σ}
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
        result = scipy.stats.lomax.cdf(x, self.alpha, scale=self.xm, loc=self.loc)
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
        α1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        α2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)
        z = lambda t: (t - self.a) / (self.c - self.a)
        result = sc.betainc(α1, α2, z(x))
        return result

    def pdf(self, x: float) -> float:
        α1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        α2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)
        return (x - self.a) ** (α1 - 1) * (self.c - x) ** (α2 - 1) / (sc.beta(α1, α2) * (self.c - self.a) ** (α1 + α2 - 1))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.b
        v2 = self.b < self.c
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            a, b, c = initial_solution
            α1 = (4 * b + c - 5 * a) / (c - a)
            α2 = (5 * c - a - 4 * b) / (c - a)
            parametric_mean = (a + 4 * b + c) / 6
            parametric_variance = ((parametric_mean - a) * (c - parametric_mean)) / 7
            parametric_median = sc.betaincinv(α1, α2, 0.5) * (c - a) + a
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
            α, a, b = initial_solution
            E1 = (a + b * α) / (1 + α)
            E2 = (2 * a**2 + 2 * a * b * α + b**2 * α * (1 + α)) / ((1 + α) * (2 + α))
            E3 = (6 * a**3 + 6 * a**2 * b * α + 3 * a * b**2 * α * (1 + α) + b**3 * α * (1 + α) * (2 + α)) / ((1 + α) * (2 + α) * (3 + α))
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
        return 1 - math.exp(-0.5 * (z(x) ** 2))

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.gamma) / self.sigma
        return z(x) * math.exp(-0.5 * (z(x) ** 2)) / self.sigma

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        sigma = math.sqrt(measurements.variance * 2 / (4 - math.pi))
        gamma = measurements.mean - sigma * math.sqrt(math.pi / 2)
        parameters = {"gamma": gamma, "sigma": sigma}
        return parameters


class RECIPROCAL:
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    def cdf(self, x: float) -> float:
        return (math.log(x) - math.log(self.a)) / (math.log(self.b) - math.log(self.a))

    def pdf(self, x: float) -> float:
        return 1 / (x * (math.log(self.b) - math.log(self.a)))

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
        def Q(M: float, a: float, b: float) -> float:
            k = 1 - M
            x = (a / b) ** k * sc.iv(k, a * b)
            acum = 0
            while x > 1e-20:
                acum += x
                k += 1
                x = (a / b) ** k * sc.iv(k, a * b)
            res = acum * math.exp(-(a**2 + b**2) / 2)
            return res

        result = 1 - Q(1, self.v / self.sigma, x / self.sigma)
        return result

    def pdf(self, x: float) -> float:
        result = (x / (self.sigma**2)) * math.exp(-(x**2 + self.v**2) / (2 * self.sigma**2)) * sc.i0(x * self.v / (self.sigma**2))
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
            E = lambda k: sigma**k * 2 ** (k / 2) * math.gamma(1 + k / 2) * sc.eval_laguerre(k / 2, -v * v / (2 * sigma * sigma))
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)

        bnds = ((0, 0), (numpy.inf, numpy.inf))
        x0 = (measurements.mean, math.sqrt(measurements.variance))
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
        result = 0.5 + z(x) * math.sqrt(self.R**2 - z(x) ** 2) / (math.pi * self.R**2) + math.asin(z(x) / self.R) / math.pi
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: t - self.loc
        result = 2 * math.sqrt(self.R**2 - z(x) ** 2) / (math.pi * self.R**2)
        return result

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.R > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        loc = measurements.mean
        R = math.sqrt(4 * measurements.variance)
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
        if x <= self.a:
            return 0
        if self.a <= x and x < self.b:
            return (1 / (self.d + self.c - self.b - self.a)) * (1 / (self.b - self.a)) * (x - self.a) ** 2
        if self.b <= x and x < self.c:
            return (1 / (self.d + self.c - self.b - self.a)) * (2 * x - self.a - self.b)
        if self.c <= x and x <= self.d:
            return 1 - (1 / (self.d + self.c - self.b - self.a)) * (1 / (self.d - self.c)) * (self.d - x) ** 2
        if x >= self.d:
            return 1

    def pdf(self, x: float) -> float:
        if x <= self.a:
            return 0
        if self.a <= x and x < self.b:
            return (2 / (self.d + self.c - self.b - self.a)) * ((x - self.a) / (self.b - self.a))
        if self.b <= x and x < self.c:
            return 2 / (self.d + self.c - self.b - self.a)
        if self.c <= x and x <= self.d:
            return (2 / (self.d + self.c - self.b - self.a)) * ((self.d - x) / (self.d - self.c))
        if x >= self.d:
            return 0

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.b
        v2 = self.b < self.c
        v3 = self.c < self.d
        return v1 and v2 and v3

    def get_parameters(self, measurements) -> dict[str, float | int]:
        a = measurements.min - 1e-3
        d = measurements.max + 1e-3

        def equations(initial_solution, measurements, a, d):
            b, c = initial_solution
            parametric_mean = (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            parametric_variance = (1 / (6 * (d + c - a - b))) * ((d**4 - c**4) / (d - c) - (b**4 - a**4) / (b - a)) - (
                (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            ) ** 2
            eq1 = parametric_mean - measurements.mean
            eq2 = parametric_variance - measurements.variance
            return (eq1, eq2)

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
        if x <= self.a:
            return 0
        if self.a < x and x <= self.c:
            return (x - self.a) ** 2 / ((self.b - self.a) * (self.c - self.a))
        if self.c < x and x < self.b:
            return 1 - ((self.b - x) ** 2 / ((self.b - self.a) * (self.b - self.c)))
        if x > self.b:
            return 1

    def pdf(self, x: float) -> float:
        if x <= self.a:
            return 0
        if self.a <= x and x < self.c:
            return 2 * (x - self.a) / ((self.b - self.a) * (self.c - self.a))
        if x == self.c:
            return 2 / (self.b - self.a)
        if x > self.c and x <= self.b:
            return 2 * (self.b - x) / ((self.b - self.a) * (self.b - self.c))
        if x > self.b:
            return 0

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
        result = sc.betainc(self.df / 2, self.df / 2, (x + math.sqrt(x * x + self.df)) / (2 * math.sqrt(x * x + self.df)))
        return result

    def pdf(self, x: float) -> float:
        result = (1 / (math.sqrt(self.df) * sc.beta(0.5, self.df / 2))) * (1 + x * x / self.df) ** (-(self.df + 1) / 2)
        return result

    def ppf(self, u):
        if u >= 0.5:
            result = math.sqrt(self.df * (1 - sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
        else:
            result = -math.sqrt(self.df * (1 - sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
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
        result = sc.betainc(self.df / 2, self.df / 2, (z(x) + math.sqrt(z(x) ** 2 + self.df)) / (2 * math.sqrt(z(x) ** 2 + self.df)))
        return result

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / (math.sqrt(self.df) * sc.beta(0.5, self.df / 2))) * (1 + z(x) * z(x) / self.df) ** (-(self.df + 1) / 2)
        return result

    def ppf(self, u):
        if u >= 0.5:
            result = self.loc + self.scale * math.sqrt(self.df * (1 - sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
            return result
        else:
            result = self.loc - self.scale * math.sqrt(self.df * (1 - sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u))) / sc.betaincinv(self.df / 2, 0.5, 2 * min(u, 1 - u)))
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
        return 1 - math.exp(-((x / self.beta) ** self.alpha))

    def pdf(self, x: float) -> float:
        return (self.alpha / self.beta) * ((x / self.beta) ** (self.alpha - 1)) * math.exp(-((x / self.beta) ** self.alpha))

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            α, β = initial_solution
            E = lambda k: (β**k) * math.gamma(1 + k / α)
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
        return 1 - math.exp(-(z(x) ** self.alpha))

    def pdf(self, x: float) -> float:
        z = lambda t: (t - self.loc) / self.beta
        return (self.alpha / self.beta) * (z(x) ** (self.alpha - 1)) * math.exp(-z(x) ** self.alpha)

    def get_num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], measurements) -> tuple[float]:
            α, β, loc = initial_solution
            E = lambda k: (β**k) * math.gamma(1 + k / α)
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
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "loc": solution.x[2]}
        return parameters


class MEASUREMENTS_CONTINUOUS:
    def __init__(self, data: list[float | int], num_bins: int | None = None):
        self.data = data
        self.length = len(data)
        self.min = min(data)
        self.max = max(data)
        self.mean = numpy.mean(data)
        self.variance = numpy.var(data, ddof=1)
        self.standard_deviation = numpy.std(data, ddof=1)
        self.skewness = scipy.stats.moment(data, 3) / pow(self.standard_deviation, 3)
        self.kurtosis = scipy.stats.moment(data, 4) / pow(self.standard_deviation, 4)
        self.median = numpy.median(data)
        self.mode = self.calculate_mode()
        self.num_bins = num_bins if num_bins != None else self.num_bins_doane()

    def __str__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"length": self.length, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def calculate_mode(self) -> float:
        def calc_shgo_mode(distribution):
            objective = lambda x: -distribution.pdf(x)[0]
            bnds = [[self.min, self.max]]
            solution = scipy.optimize.shgo(objective, bounds=bnds, n=100 * self.length)
            return solution.x[0]

        distribution = scipy.stats.gaussian_kde(self.data)
        shgo_mode = calc_shgo_mode(distribution)
        return shgo_mode

    def num_bins_doane(self):
        N = self.length
        skewness = scipy.stats.skew(self.data)
        sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
        num_bins = 1 + math.log2(N) + math.log2(1 + abs(skewness) / sigma_g1)
        return math.ceil(num_bins)


def test_chi_square_continuous(data, distribution, measurements, confidence_level=0.95):
    N = measurements.length
    num_bins = measurements.num_bins
    frequencies, bin_edges = numpy.histogram(data, num_bins)
    freedom_degrees = num_bins - 1 - distribution.get_num_parameters()
    errors = []
    for i, observed in enumerate(frequencies):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        expected = N * (distribution.cdf(upper) - distribution.cdf(lower))
        errors.append(((observed - expected) ** 2) / expected)
    statistic_chi2 = sum(errors)
    critical_value = scipy.stats.chi2.ppf(confidence_level, freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {"test_statistic": statistic_chi2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_chi2


def test_kolmogorov_smirnov_continuous(data, distribution, measurements, confidence_level=0.95):
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


def test_anderson_darling_continuous(data, distribution, measurements, confidence_level=0.95):
    N = measurements.length
    data.sort()
    S = 0
    for k in range(N):
        c1 = math.log(distribution.cdf(data[k]))
        c2 = math.log(1 - distribution.cdf(data[N - k - 1]))
        c3 = (2 * (k + 1) - 1) / N
        S += c3 * (c1 + c2)
    A2 = -N - S
    critical_value = ad_critical_value(confidence_level, N)
    p_value = ad_p_value(N, A2)
    rejected = A2 >= critical_value
    result_test_ad = {"test_statistic": A2, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ad


def phitter_continuous(data, num_bins=None, confidence_level=0.95):
    _all_distributions = [
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
    measurements = MEASUREMENTS_CONTINUOUS(data, num_bins)

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
        if validate_estimation and distribution.parameter_restrictions() and not math.isnan(sse) and not math.isinf(sse):
            try:
                chi2_test = test_chi_square_continuous(data, distribution, measurements, confidence_level=confidence_level)
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
                ks_test = test_kolmogorov_smirnov_continuous(data, distribution, measurements, confidence_level=confidence_level)
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
                ad_test = test_anderson_darling_continuous(data, distribution, measurements, confidence_level=confidence_level)
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

    sorted_results_sse = {distribution: results for distribution, results in sorted(RESPONSE.items(), key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
    aproved_results = {distribution: results for distribution, results in sorted_results_sse.items() if results["n_test_passed"] > 0}

    return sorted_results_sse, aproved_results


if __name__ == "__main__":
    path = "../../continuous/data/data_beta.txt"
    sample_distribution_file = open(path, "r")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    sorted_results_sse, aproved_results = phitter_continuous(data, 20, confidence_level=0.99)

    for distribution, results in aproved_results.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
