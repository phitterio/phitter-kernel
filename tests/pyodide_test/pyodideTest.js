async function main() {
    let pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.4/full",
    });
    await pyodide.loadPackage(["scipy"]);

    pyodide.runPython(`
        import math
        import numpy
        import scipy.integrate
        import scipy.optimize
        import scipy.special
        import scipy.stats
        import warnings
        import concurrent.futures
        
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
                z = lambda t: (t - self.a) / (self.b - self.a)
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
                z = lambda t: (t - self.loc) / self.alpha
                result = scipy.special.erf(z(x) / (numpy.sqrt(2))) - numpy.sqrt(2 / numpy.pi) * z(x) * numpy.exp(-z(x) ** 2 / 2)
                return result
            def pdf(self, x: float) -> float:
                z = lambda t: (t - self.loc) / self.alpha
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
        
        class CONTINUOUS_MEASURES:
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
        
        def evaluate_continuous_test_chi_square(distribution, measurements, confidence_level=0.95):
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
        
        def evaluate_continuous_test_kolmogorov_smirnov(distribution, measurements, confidence_level=0.95):
            N = measurements.length
            Fn = distribution.cdf(measurements.data)
            errors = numpy.abs(measurements.Sn_ks[measurements.idx_ks] - Fn[measurements.idx_ks])
            statistic_ks = numpy.max(errors)
            critical_value = scipy.stats.kstwo.ppf(confidence_level, N)
            p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, N)
            rejected = statistic_ks >= critical_value
            result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
            return result_test_ks
        
        def evaluate_continuous_test_anderson_darling(distribution, measurements, confidence_level=0.95):
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
                minimum_sse=numpy.inf,
            ):
                self.data = data
                self.measurements = CONTINUOUS_MEASURES(self.data, num_bins, confidence_level)
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
                    v1 = self.test(evaluate_continuous_test_chi_square, "chi_square", distribution)
                    v2 = self.test(evaluate_continuous_test_kolmogorov_smirnov, "kolmogorov_smirnov", distribution)
                    v3 = self.test(evaluate_continuous_test_anderson_darling, "anderson_darling", distribution)
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
            def fit(self, n_workers: int = 1):
                if n_workers <= 0:
                    raise Exception("n_workers must be greater than 1")
                ALL_CONTINUOUS_DISTRIBUTIONS = [
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
                if n_workers == 1:
                    processing_results = [self.process_distribution(distribution_class) for distribution_class in ALL_CONTINUOUS_DISTRIBUTIONS]
                else:
                    processing_results = list(concurrent.futures.ProcessPoolExecutor(max_workers=n_workers).map(self.process_distribution, ALL_CONTINUOUS_DISTRIBUTIONS))
                processing_results = [r for r in processing_results if r is not None]
                sorted_distributions_sse = {distribution: results for distribution, results in sorted(processing_results, key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
                not_rejected_distributions = {distribution: results for distribution, results in sorted_distributions_sse.items() if results["n_test_passed"] > 0}
                return sorted_distributions_sse, not_rejected_distributions
    `)

    pyodide.runPython(`
        import time
        array = [782.1587476, 795.387086, 799.7345698, 806.9197855, 807.0463942, 807.3649597, 809.8982279, 812.8464141, 813.5454571, 814.0393853, 814.4982938, 814.7235583, 815.2237586, 815.3689112, 815.549188, 816.5616516, 816.6898013, 816.7170744, 816.9565214, 818.545773, 818.5851785, 818.6100861, 818.6707629, 818.8429549, 819.5863531, 820.7975631, 821.2037954, 821.2545802, 821.3866398, 822.7657442, 823.5462453, 823.6373205, 823.8577613, 824.4523423, 824.6145776, 826.8229941, 826.9143961, 827.4739583, 827.7353163, 828.1253071, 828.18333, 828.4066215, 828.4682931, 828.6164161, 828.7162096, 828.9112205, 829.0875069, 829.239806, 829.35716, 830.0010021, 830.1688936, 830.3358744, 831.4524552, 831.5245596, 831.7075856, 832.0860798, 832.2370759, 832.3447862, 833.2832111, 833.4840663, 833.5412179, 833.8488735, 833.939438, 834.0439737, 834.3184225, 834.4691618, 835.0099595, 835.3312188, 835.4768996, 835.6195047, 835.6228222, 835.6312219, 835.6332902, 835.7457027, 836.2510027, 836.2798312, 836.4761627, 836.5636378, 836.7503009, 836.7804539, 836.9120054, 837.2280176, 837.2544554, 837.5386083, 837.663141, 838.0297279, 838.3433105, 838.35716, 838.4609116, 838.6197982, 838.730219, 839.0732597, 839.1237875, 839.1483822, 839.4337945, 839.4762865, 839.6733894, 839.6817123, 839.7688878, 839.8980412, 840.0159731, 840.2424596, 840.3715146, 840.3871349, 841.0504482, 841.1463021, 841.1763338, 841.2826581, 841.7285967, 841.9570781, 841.9581628, 841.9860092, 842.0103145, 842.0312513, 842.0517846, 842.0928257, 842.1650072, 842.2656446, 842.2829051, 842.2834287, 842.3116781, 842.3786753, 842.4788982, 842.4917058, 842.4981138, 842.5453892, 842.7016393, 842.7861256, 842.8692922, 842.9069063, 842.9207527, 842.9359663, 842.9533521, 842.9605148, 843.099987, 843.2001442, 843.467077, 843.5545311, 843.657943, 843.696733, 843.7495148, 843.8336588, 843.8869706, 843.9069298, 844.1345881, 844.2656454, 844.2886571, 844.3334613, 844.3965966, 844.4545492, 844.6503324, 844.7755959, 845.46535, 845.851068, 846.4413221, 846.5233331, 846.5892224, 846.6594364, 846.8058929, 846.8283309, 846.8546908, 846.8978856, 847.0392173, 847.0792188, 847.4043115, 847.4994034, 847.5290311, 847.7404196, 847.7508672, 847.8878883, 848.0013245, 848.0751792, 848.1196961, 848.404692, 848.5295817, 848.5861349, 848.6511693, 848.6792686, 848.6904689, 848.7935428, 848.9257823, 849.1313569, 849.5648062, 849.5894205, 849.6632241, 849.7188228, 849.7968841, 849.8450987, 849.919392, 849.9852716, 850.0058486, 850.0656169, 850.2730876, 850.3143636, 850.3615158, 850.3854795, 850.4776778, 850.655202, 850.7403472, 851.1023123, 851.1253055, 851.1548517, 851.162766, 851.2446505, 851.334977, 851.5258614, 851.689019, 851.6948659, 851.7774109, 851.8469066, 851.9722277, 852.0785364, 852.0942629, 852.1183263, 852.1509178, 852.1818823, 852.3636828, 852.3719416, 852.4076414, 852.4124014, 852.4285035, 852.5509646, 852.6016967, 852.6295195, 852.778375, 852.9630238, 853.0480618, 853.1853232, 853.2536754, 853.4650902, 853.474556, 853.4803882, 853.4981369, 853.5169506, 853.7498026, 853.7596088, 853.771249, 853.8177267, 853.9648747, 854.0160808, 854.0371607, 854.1469414, 854.1878797, 854.2782155, 854.4959794, 854.5128875, 854.5341941, 854.6029135, 854.635134, 854.6732009, 854.7233832, 854.7251252, 854.8667529, 854.9922837, 855.0582391, 855.1187926, 855.1202825, 855.1394506, 855.2158649, 855.224387, 855.277782, 855.3214245, 855.3606894, 855.4310529, 855.4707688, 855.5728725, 855.7080877, 855.7547884, 855.7788215, 855.7846521, 855.8148306, 855.8377203, 855.8591832, 856.0358198, 856.1468668, 856.2625146, 856.3525789, 856.3710821, 856.4059718, 856.4735757, 856.5394404, 856.5788504, 856.6649837, 856.6866937, 856.7298533, 856.7300913, 856.7835127, 856.8890044, 856.9803916, 857.0230457, 857.0484798, 857.1133314, 857.2304327, 857.2570511, 857.2697741, 857.3049065, 857.3432602, 857.3461793, 857.4165333, 857.5437701, 857.6109487, 857.6277571, 857.639747, 857.727137, 857.9464543, 857.9506384, 858.0427017, 858.1118515, 858.1213896, 858.1335344, 858.2070648, 858.2369164, 858.2565261, 858.2844712, 858.3793605, 858.4157308, 858.4666072, 858.5767237, 858.610572, 858.6609581, 858.8101175, 858.9694253, 858.9717189, 858.9762378, 859.0542378, 859.114204, 859.1832323, 859.1964421, 859.253045, 859.2957552, 859.3754127, 859.3996904, 859.4222159, 859.456602, 859.5289713, 859.5448865, 859.679917, 859.6958147, 859.7800027, 859.7809582, 859.788375, 859.9181919, 859.9629368, 859.9669429, 859.9918573, 860.0009367, 860.0268162, 860.0409976, 860.0689805, 860.0764545, 860.1176702, 860.1634715, 860.1694903, 860.2103333, 860.2460809, 860.3538701, 860.3826605, 860.4005881, 860.4138446, 860.4185801, 860.4638097, 860.4670054, 860.5283644, 860.5931542, 860.6079135, 860.6204238, 860.622259, 860.8544945, 860.8668242, 860.8788494, 860.9821749, 860.9830103, 861.0839108, 861.1113551, 861.3580732, 861.392444, 861.3962901, 861.4033673, 861.5201231, 861.5424699, 861.569766, 861.5737016, 861.6040886, 861.6309877, 861.6310055, 861.6564187, 861.7807512, 861.8008336, 861.8263225, 861.8810661, 861.9628085, 861.9774943, 862.0591369, 862.0911487, 862.1934019, 862.2110969, 862.240667, 862.2508034, 862.2773312, 862.2801743, 862.4273951, 862.4560616, 862.4699879, 862.5850871, 862.5917319, 862.6874776, 862.6886352, 862.7384848, 862.7569398, 862.8529433, 862.8626725, 862.9115471, 863.0274525, 863.1849294, 863.1906987, 863.1970598, 863.2648953, 863.2777406, 863.3290182, 863.3865371, 863.4308369, 863.4595399, 863.5809612, 863.5878549, 863.6074579, 863.6247579, 863.6664467, 863.6738954, 863.717568, 863.7253293, 863.823153, 863.8716101, 863.9300991, 863.9905764, 864.0004003, 864.0209765, 864.0667252, 864.0826488, 864.0893278, 864.1737217, 864.1997249, 864.2046876, 864.229757, 864.2397886, 864.2465886, 864.305131, 864.3389194, 864.3476932, 864.3835192, 864.404353, 864.4997047, 864.5255873, 864.5457004, 864.5592507, 864.7586133, 864.8069879, 864.8118739, 864.8344875, 864.8688667, 864.9462794, 864.9604254, 864.9910742, 865.0163996, 865.022091, 865.0683656, 865.1081238, 865.1583917, 865.2501112, 865.2749089, 865.2851953, 865.325469, 865.328327, 865.430275, 865.4514735, 865.4600855, 865.5040912, 865.5896058, 865.6761783, 865.7396302, 865.7641858, 865.8602111, 865.8624922, 865.9452542, 865.9459657, 865.9462152, 865.9477506, 865.961709, 865.9656759, 865.9978445, 866.0136831, 866.0875596, 866.1238567, 866.1823497, 866.1879169, 866.2497999, 866.267226, 866.2792419, 866.3538624, 866.3790684, 866.4294118, 866.5094482, 866.5472755, 866.5635153, 866.6098511, 866.6173881, 866.6246633, 866.6476948, 866.6561446, 866.6862933, 866.6956207, 866.7156736, 866.7353973, 866.7554796, 866.7675189, 866.8370668, 866.879343, 866.953693, 866.9724811, 867.022321, 867.2248012, 867.2665982, 867.2899444, 867.3301462, 867.331914, 867.3370949, 867.3399766, 867.345396, 867.3542287, 867.429616, 867.4492505, 867.4537017, 867.4930933, 867.5284609, 867.5651521, 867.5880558, 867.6009308, 867.6351897, 867.6890995, 867.7021903, 867.7752852, 867.8026153, 867.8731957, 867.9621609, 867.97682, 867.9836246, 868.0264632, 868.0480631, 868.1787902, 868.2426199, 868.3386618, 868.3503943, 868.3947593, 868.404012, 868.5082333, 868.5350414, 868.5504229, 868.5783449, 868.6340435, 868.7074892, 868.7670459, 868.8085481, 868.8189999, 868.8281739, 868.8307809, 868.8733644, 869.0649608, 869.1204867, 869.1316025, 869.192464, 869.2925762, 869.2956843, 869.324623, 869.3591494, 869.3592932, 869.3726452, 869.3765701, 869.3783265, 869.5025607, 869.5450973, 869.5500504, 869.5610493, 869.590671, 869.7012583, 869.7354967, 869.7394734, 869.7786962, 869.8088289, 869.8542718, 869.8804441, 869.8862735, 869.9443149, 869.9460516, 870.0578012, 870.0666913, 870.1122222, 870.1171013, 870.1348305, 870.1552228, 870.1644474, 870.2066534, 870.21681, 870.2581006, 870.2604022, 870.2708777, 870.2806314, 870.3174869, 870.3355396, 870.39364, 870.4213434, 870.4233218, 870.4338684, 870.4433912, 870.4885629, 870.5082413, 870.5338205, 870.5493134, 870.6151044, 870.6244149, 870.658427, 870.683495, 870.7671641, 870.7685698, 870.7889983, 870.8489869, 870.8704471, 870.9019464, 870.9296165, 870.9640818, 871.0246954, 871.0912496, 871.1085711, 871.254292, 871.2640589, 871.2888834, 871.3106621, 871.4411725, 871.4447261, 871.4843156, 871.6184513, 871.6191847, 871.6912778, 871.6950445, 871.7112015, 871.7130404, 871.729319, 871.7880936, 871.8391008, 871.8635821, 871.8836068, 871.8847767, 871.9395164, 871.9583515, 871.958968, 872.0153485, 872.037776, 872.1005787, 872.2087747, 872.2225163, 872.2274791, 872.2770698, 872.2831565, 872.2851052, 872.331393, 872.4011178, 872.4714698, 872.4971242, 872.5219973, 872.5982144, 872.5993298, 872.6085526, 872.6728447, 872.7756421, 872.8995799, 872.9145417, 872.9207345, 872.9230154, 872.9449416, 872.9755434, 873.0155942, 873.0919728, 873.1167569, 873.1361866, 873.1814808, 873.2183366, 873.2392234, 873.2485468, 873.2702648, 873.2720889, 873.2951493, 873.3865566, 873.4158555, 873.500556, 873.5319464, 873.6855352, 873.6906065, 873.7271928, 873.7583086, 873.816419, 873.8792258, 873.9204321, 873.9208096, 873.940095, 873.9485891, 873.9811986, 873.9952788, 874.087778, 874.0932569, 874.1038451, 874.119552, 874.1295218, 874.161451, 874.1734033, 874.2191899, 874.2746553, 874.3288225, 874.3423884, 874.3488125, 874.3803244, 874.4342655, 874.4367397, 874.4693033, 874.4707547, 874.5863337, 874.6078583, 874.6903967, 874.7481289, 874.8709368, 874.9086792, 874.9746519, 874.9760631, 874.9861585, 875.1012416, 875.1260115, 875.1467516, 875.190557, 875.2584833, 875.2736726, 875.275011, 875.3452632, 875.3537608, 875.4334982, 875.4508048, 875.4969518, 875.4982944, 875.6160241, 875.6258949, 875.634177, 875.6627567, 875.6802332, 875.7099107, 875.7216339, 875.7349581, 875.8132414, 875.8299069, 875.8637355, 875.893718, 875.923427, 876.0554595, 876.0561243, 876.0571265, 876.0922079, 876.101079, 876.1209743, 876.1306606, 876.136072, 876.1668303, 876.2053787, 876.2544616, 876.2579437, 876.3476088, 876.3509242, 876.3513804, 876.4591675, 876.4735279, 876.4864953, 876.5636328, 876.5851535, 876.7597727, 876.7665073, 876.7839597, 876.787057, 876.7937265, 876.7993689, 876.8150221, 876.8169517, 876.8471022, 876.8875677, 876.9282669, 876.9710813, 876.9851825, 876.9949606, 877.0381644, 877.0574657, 877.1169795, 877.1330331, 877.1404848, 877.2007651, 877.285081, 877.2908848, 877.3046894, 877.31101, 877.3547276, 877.3634816, 877.3865706, 877.4222439, 877.4329044, 877.4601132, 877.4854006, 877.4945715, 877.5408161, 877.5417593, 877.5742222, 877.6271825, 877.6307179, 877.6882774, 877.7097935, 877.7566048, 877.7724436, 877.8531696, 877.876906, 877.9067431, 877.9147984, 877.9177926, 878.0040262, 878.0229884, 878.0248847, 878.0477176, 878.0522838, 878.055012, 878.1174972, 878.1676769, 878.2015039, 878.2190838, 878.2422891, 878.3090724, 878.3429301, 878.3741484, 878.3918131, 878.4258675, 878.4875268, 878.646919, 878.6507949, 878.6645655, 878.760361, 878.7609977, 878.8358942, 878.8537124, 878.87946, 878.908014, 879.0540702, 879.0623667, 879.0784496, 879.085164, 879.1661569, 879.167068, 879.1898847, 879.19048, 879.1950193, 879.2267717, 879.2443516, 879.2804033, 879.3542149, 879.3744385, 879.396938, 879.5199186, 879.6014262, 879.622814, 879.6375174, 879.6394698, 879.6768376, 879.7557698, 879.7601091, 879.7712126, 879.7796249, 879.7841481, 879.8043685, 879.8245905, 879.8341876, 879.9594216, 879.9653627, 879.9822428, 880.1282142, 880.1398705, 880.162837, 880.1748014, 880.1807493, 880.1870496, 880.2009292, 880.235968, 880.2450426, 880.2726906, 880.2842302, 880.3214688, 880.3582622, 880.3647906, 880.3709233, 880.3977348, 880.411284, 880.4200699, 880.4381291, 880.4518882, 880.4710647, 880.5177603, 880.527427, 880.5390287, 880.540021, 880.5600933, 880.5735713, 880.5738835, 880.5870468, 880.5906233, 880.6250329, 880.6296337, 880.6343272, 880.6464337, 880.6476791, 880.6634182, 880.6765156, 880.6876663, 880.733272, 880.7426167, 880.759981, 880.7699676, 880.7929376, 880.793621, 880.8300851, 880.8582531, 880.8720513, 880.9600542, 880.9715997, 880.9804228, 881.0049092, 881.0679231, 881.076677, 881.0866773, 881.1153204, 881.1289506, 881.1336735, 881.1565068, 881.1904345, 881.1978718, 881.2141377, 881.2410964, 881.2587872, 881.2749548, 881.332104, 881.3349176, 881.3384084, 881.4093262, 881.438169, 881.4508062, 881.49238, 881.5227655, 881.5893501, 881.6083354, 881.6542917, 881.6582569, 881.6890546, 881.7050085, 881.7147464, 881.7991998, 881.8064411, 881.9246093, 881.942485, 881.9985998, 882.0190624, 882.0249948, 882.0665313, 882.1011165, 882.1327578, 882.1354598, 882.1702949, 882.2013945, 882.2278444, 882.2358369, 882.2495746, 882.2821253, 882.3975979, 882.4409709, 882.4644211, 882.4647359, 882.4777446, 882.4838722, 882.5160373, 882.5435177, 882.5867244, 882.6559987, 882.6954033, 882.7085532, 882.7287336, 882.8271223, 882.8466527, 882.8616746, 882.9118585, 882.9446462, 882.9558845, 883.0317416, 883.0674043, 883.0762778, 883.0859195, 883.1508205, 883.1826474, 883.2859549, 883.300353, 883.3032277, 883.3046086, 883.3183623, 883.3239844, 883.3364634, 883.3553425, 883.3639472, 883.4573078, 883.4881879, 883.5122604, 883.531111, 883.5661984, 883.5784611, 883.5801073, 883.6007569, 883.6271874, 883.6436494, 883.6671969, 883.6922505, 883.7375095, 883.7590087, 883.7663095, 883.7861057, 883.7948863, 883.8002845, 883.8114709, 883.8117659, 883.8194284, 883.8230523, 883.8633183, 883.8743359, 883.9324142, 883.9451261, 883.9466271, 883.9615063, 884.0058444, 884.0154656, 884.0192582, 884.0365996, 884.0513416, 884.0530919, 884.0629919, 884.0787077, 884.087954, 884.1296226, 884.1578377, 884.2954658, 884.3162841, 884.3247916, 884.3425989, 884.3758133, 884.3965611, 884.4236697, 884.5079064, 884.7068506, 884.7166289, 884.7368457, 884.7473131, 884.8213238, 884.8840868, 884.9402885, 884.9660408, 884.977664, 885.1604797, 885.2200021, 885.2779761, 885.3256131, 885.3596937, 885.370054, 885.4052554, 885.4307169, 885.4601886, 885.5100024, 885.5323465, 885.5488229, 885.5665213, 885.5758687, 885.5957289, 885.597766, 885.6205499, 885.6280059, 885.6477598, 885.6851975, 885.6939911, 885.7381134, 885.7569506, 885.8429192, 885.8533163, 885.8674844, 885.9329921, 885.9636556, 885.9877656, 886.0320199, 886.0431425, 886.1067792, 886.1201001, 886.1428073, 886.1459527, 886.1958275, 886.2201081, 886.2568192, 886.2601177, 886.281765, 886.2863002, 886.3409374, 886.3649512, 886.388617, 886.4325098, 886.4347744, 886.4458991, 886.4637025, 886.4946113, 886.5456995, 886.5575286, 886.6079876, 886.6097386, 886.647303, 886.6751305, 886.6855316, 886.7122966, 886.7594417, 886.7672019, 886.7672357, 886.8353837, 886.8467557, 886.8557887, 886.8688785, 886.9293603, 886.9528441, 886.9777796, 886.9790721, 886.9824876, 887.0183222, 887.024093, 887.0656213, 887.0887146, 887.0887445, 887.0893576, 887.1047644, 887.1080161, 887.1393491, 887.1554699, 887.1621051, 887.1816174, 887.2089312, 887.2387944, 887.2515879, 887.2761876, 887.2770555, 887.2807531, 887.281315, 887.2898924, 887.2920651, 887.293797, 887.3033454, 887.3259704, 887.3836914, 887.4399223, 887.450823, 887.5287286, 887.5764584, 887.606971, 887.6097644, 887.6328683, 887.6337398, 887.6450292, 887.6598165, 887.6787459, 887.6787733, 887.6941082, 887.7120302, 887.8366728, 887.8526492, 887.8707368, 887.8972027, 887.9410559, 887.958195, 887.9698014, 887.9711167, 888.0331645, 888.0645098, 888.1346028, 888.1461451, 888.1751073, 888.1797978, 888.2325188, 888.2428044, 888.279416, 888.3003687, 888.316888, 888.3839806, 888.4703156, 888.5058347, 888.5060203, 888.5162328, 888.5364139, 888.5764851, 888.5816954, 888.5821762, 888.5833724, 888.6159206, 888.6189233, 888.6405493, 888.67704, 888.6879409, 888.7022493, 888.7055447, 888.7723846, 888.8059683, 888.8502701, 888.8810273, 888.890289, 888.89071, 889.0200597, 889.0556372, 889.0773218, 889.1145304, 889.1230876, 889.2056815, 889.2205136, 889.2435282, 889.2488072, 889.2957895, 889.3474809, 889.373568, 889.4010399, 889.4254407, 889.4254472, 889.4523269, 889.4538101, 889.4541286, 889.4875207, 889.4975523, 889.5111013, 889.5895742, 889.615238, 889.6368376, 889.6477856, 889.6793873, 889.6809367, 889.7500712, 889.7590507, 889.7600762, 889.8020726, 889.8119168, 889.8409789, 889.8473712, 889.8496589, 889.8598788, 889.8603202, 889.8979006, 889.8986021, 889.9104863, 889.9171215, 889.919348, 889.9249594, 889.9254897, 889.9486683, 890.034532, 890.1141163, 890.115741, 890.1320128, 890.1781133, 890.2308049, 890.2554206, 890.2894303, 890.3769275, 890.4222579, 890.4717265, 890.4754674, 890.5782469, 890.6816568, 890.6898632, 890.6916582, 890.6973432, 890.703591, 890.7427699, 890.743887, 890.7447999, 890.8436663, 890.9997614, 891.0407106, 891.0558675, 891.0582536, 891.0732628, 891.1655512, 891.1953472, 891.1995171, 891.212909, 891.2174065, 891.2705191, 891.2826232, 891.298377, 891.3227383, 891.3967681, 891.4292382, 891.4577809, 891.4606363, 891.5611054, 891.6023162, 891.6164771, 891.6448012, 891.67849, 891.6900524, 891.7348868, 891.7388607, 891.7440362, 891.7656983, 891.7867272, 891.8385851, 891.8437513, 891.8447278, 891.8448697, 891.8616032, 891.8859532, 891.9384412, 891.9416411, 891.9437373, 891.9605029, 891.967867, 891.968273, 891.9725994, 891.9787192, 891.9885078, 891.9953113, 892.0151159, 892.0438879, 892.0879054, 892.1008516, 892.1139413, 892.1990777, 892.2167649, 892.2500578, 892.2505215, 892.2701887, 892.2829834, 892.2953644, 892.3455451, 892.3962515, 892.3983337, 892.4103317, 892.4233026, 892.4238796, 892.4270264, 892.4464271, 892.481909, 892.5361394, 892.54059, 892.5485391, 892.601353, 892.6691471, 892.679399, 892.7326713, 892.7430822, 892.8220908, 892.8486552, 892.8652216, 892.8772855, 892.8949763, 892.8974853, 892.9011978, 892.9042047, 892.9123432, 892.9220852, 892.9400307, 892.963279, 892.9725219, 893.022682, 893.0349285, 893.0870741, 893.1310514, 893.1332703, 893.1386134, 893.1721101, 893.17912, 893.1948263, 893.2437762, 893.2667186, 893.402335, 893.4672432, 893.4966162, 893.5017894, 893.5779986, 893.639394, 893.6517574, 893.7181125, 893.7715267, 893.7895381, 893.8417352, 893.8566465, 893.9103179, 893.9172946, 894.0203375, 894.0385365, 894.0389549, 894.0676295, 894.0729616, 894.0791502, 894.0955757, 894.112641, 894.1204725, 894.1419384, 894.1681853, 894.1769813, 894.2508865, 894.3094656, 894.3098951, 894.322457, 894.3417198, 894.4409344, 894.5213306, 894.5299124, 894.5829505, 894.6071095, 894.6469123, 894.6821165, 894.7108331, 894.7370073, 894.776624, 894.7861041, 894.7913249, 894.8194891, 894.8301659, 894.8304291, 894.8815089, 894.939508, 895.0442227, 895.0589671, 895.0750121, 895.0821013, 895.0823947, 895.1182341, 895.1318765, 895.1378973, 895.1654248, 895.1882584, 895.2226397, 895.2458569, 895.2484016, 895.2518521, 895.3243588, 895.3269164, 895.3591796, 895.364725, 895.3712169, 895.3737035, 895.397426, 895.4123705, 895.43446, 895.4771601, 895.4902263, 895.4978124, 895.4999842, 895.5029089, 895.5603089, 895.5989727, 895.6361632, 895.6476914, 895.6696314, 895.6794662, 895.6886413, 895.7005851, 895.7028268, 895.7097312, 895.7122032, 895.7192673, 895.7257132, 895.8088649, 895.8324811, 895.8536443, 895.8915327, 895.9436994, 895.9640563, 895.9990576, 896.0127408, 896.0523528, 896.0569794, 896.0662836, 896.0757807, 896.0961626, 896.1413303, 896.1699288, 896.1932354, 896.2142678, 896.2524928, 896.3069022, 896.3124275, 896.3834622, 896.4204933, 896.4652681, 896.4761897, 896.4824148, 896.5721113, 896.6276986, 896.6523918, 896.6585213, 896.6856333, 896.7107378, 896.7427303, 896.8228397, 896.8520311, 896.9223492, 896.9231451, 896.9270623, 896.942365, 896.945728, 896.9518687, 896.9915516, 897.0266234, 897.0325964, 897.0731926, 897.3285089, 897.3603801, 897.361806, 897.3839029, 897.3876408, 897.5075973, 897.5233802, 897.5706171, 897.6076083, 897.6297917, 897.6605212, 897.7397179, 897.7584852, 897.8049122, 897.8060626, 897.8148519, 897.9278937, 897.9489603, 898.0003962, 898.0136319, 898.050075, 898.0515119, 898.0932751, 898.1449444, 898.1633453, 898.1639141, 898.1701955, 898.1831816, 898.2159292, 898.2423209, 898.3152917, 898.3236292, 898.3339617, 898.367438, 898.390342, 898.4174269, 898.4345396, 898.4468698, 898.4616088, 898.48712, 898.487621, 898.5334907, 898.5484079, 898.5771893, 898.5869894, 898.6292885, 898.6379138, 898.6634717, 898.6932746, 898.7540218, 898.7958522, 898.8274402, 898.8723178, 898.8794034, 898.8794041, 898.8894314, 898.8982681, 898.9100762, 898.9557457, 899.00263, 899.0093631, 899.0862708, 899.1025951, 899.1445121, 899.1757613, 899.2203011, 899.2244166, 899.229526, 899.2651143, 899.2841141, 899.2873029, 899.4093471, 899.476449, 899.502388, 899.6714585, 899.7395829, 899.8162084, 899.9233127, 899.9901443, 899.9942598, 900.1116102, 900.1125491, 900.1291797, 900.1329038, 900.1344617, 900.2097482, 900.2766165, 900.2773693, 900.3033516, 900.3135958, 900.3494446, 900.3588533, 900.3628824, 900.3835626, 900.4520788, 900.4931674, 900.4971906, 900.506054, 900.5522291, 900.6197565, 900.6483866, 900.7092849, 900.709415, 900.7114779, 900.7237187, 900.7382911, 900.7767414, 900.7858665, 900.8030744, 900.8607497, 900.8705958, 900.9412396, 900.9501378, 900.9671252, 900.9979218, 901.0434152, 901.0627598, 901.0990398, 901.1323016, 901.1328693, 901.1611046, 901.1652202, 901.2467294, 901.2611865, 901.2826935, 901.3029028, 901.3103576, 901.315496, 901.3701373, 901.3844942, 901.3967882, 901.398208, 901.4006631, 901.4085964, 901.5948337, 901.6081592, 901.6421085, 901.667053, 901.6721479, 901.6733114, 901.6809021, 901.6846704, 901.7138236, 901.7358557, 901.7939344, 901.8603814, 901.8699795, 901.9570929, 901.9589088, 901.9745391, 902.0272182, 902.0860045, 902.1088309, 902.1752115, 902.1934363, 902.2328392, 902.3609166, 902.3695855, 902.3717079, 902.3876664, 902.3955054, 902.4044419, 902.4229559, 902.4383004, 902.4455083, 902.5140584, 902.5214115, 902.5235095, 902.5509479, 902.6021237, 902.6099077, 902.6200086, 902.6209301, 902.6343404, 902.6360412, 902.6641587, 902.6989851, 902.7382087, 902.7499785, 902.7769596, 902.7863462, 902.8335253, 902.8393497, 902.8445629, 902.8634073, 902.8786193, 902.894155, 902.9022432, 902.93766, 902.9615891, 903.0149059, 903.089666, 903.1342364, 903.1768114, 903.2041591, 903.2304755, 903.2898804, 903.3444276, 903.3787424, 903.3863479, 903.4465128, 903.4756626, 903.4771877, 903.5385141, 903.7508969, 903.7803873, 903.7974587, 903.8322739, 903.8446915, 903.8639954, 903.8916352, 903.9907742, 904.007145, 904.0082079, 904.1331489, 904.2441541, 904.2536612, 904.2694689, 904.3419049, 904.373089, 904.4126359, 904.4676765, 904.5163191, 904.5265953, 904.5732632, 904.5979707, 904.6037334, 904.6133736, 904.6324168, 904.6559951, 904.754243, 904.7822711, 904.7999681, 904.8646061, 904.8762071, 904.8886065, 904.9062935, 904.9318547, 904.9653376, 904.9799326, 904.9931785, 904.9983149, 905.137418, 905.1560945, 905.19594, 905.201404, 905.225989, 905.2273371, 905.2621844, 905.303947, 905.348164, 905.4334153, 905.4588778, 905.460257, 905.4723355, 905.5302436, 905.6569755, 905.689561, 905.6976947, 905.7562625, 905.7975113, 905.802205, 905.8183842, 905.8405572, 905.8405706, 905.8591788, 905.8975369, 905.9496034, 905.9831573, 906.0830044, 906.1112427, 906.1494164, 906.1510998, 906.2260771, 906.2392914, 906.2449855, 906.2821834, 906.284019, 906.2958529, 906.3243491, 906.3344775, 906.3705772, 906.3730813, 906.3868379, 906.5995429, 906.8230319, 906.9119261, 906.9341949, 906.9592772, 906.9621404, 906.9689581, 906.9998319, 907.0075709, 907.0229988, 907.0653828, 907.12857, 907.1784884, 907.201896, 907.2292191, 907.2906568, 907.3497231, 907.3582152, 907.426522, 907.4599169, 907.4752755, 907.4954864, 907.5258505, 907.633815, 907.7011358, 907.7211735, 907.748167, 907.8460751, 907.9476722, 907.9621109, 908.0350976, 908.0388407, 908.0393228, 908.0493037, 908.0653558, 908.1102469, 908.1722999, 908.2485159, 908.2640138, 908.2656005, 908.2725651, 908.2790678, 908.2857411, 908.400655, 908.4116419, 908.4185085, 908.4195954, 908.4255357, 908.5306883, 908.549186, 908.5914958, 908.6540995, 908.6939522, 908.7045479, 908.7051658, 908.7111827, 908.7341369, 908.7639622, 908.7920531, 908.867048, 908.8973452, 908.9120806, 908.93188, 908.9688863, 909.013755, 909.0881028, 909.2151717, 909.2272985, 909.3135467, 909.3555895, 909.4349605, 909.5561474, 909.5740302, 909.617823, 909.6190973, 909.6689245, 909.6804831, 909.6993531, 909.722541, 909.7627871, 909.7795055, 909.8137075, 909.8286293, 909.8453806, 909.8706203, 909.8977213, 909.8983956, 909.9378658, 909.9817004, 910.0128044, 910.0177102, 910.0195578, 910.0231474, 910.0313568, 910.1493956, 910.1662086, 910.1703876, 910.2270864, 910.2526318, 910.2935325, 910.6453424, 910.738297, 910.7716056, 910.8149708, 910.8590972, 910.8788377, 910.8978674, 910.9422266, 910.9813919, 911.0547164, 911.1920417, 911.2202988, 911.3053698, 911.3172388, 911.400038, 911.4064037, 911.443415, 911.4510954, 911.4789544, 911.5093887, 911.5243282, 911.5382702, 911.612624, 911.6314939, 911.6803939, 911.6890935, 911.7821455, 911.7977209, 911.8275297, 911.8672378, 911.9396266, 911.9846811, 912.0097093, 912.0208783, 912.1929168, 912.2165956, 912.2373545, 912.3213836, 912.3278155, 912.4057646, 912.5418225, 912.5437325, 912.5585737, 912.6612059, 912.6692051, 912.6694686, 912.702797, 912.7106144, 912.9324081, 912.9572995, 912.9990672, 913.1367123, 913.2590401, 913.2597605, 913.311628, 913.3503531, 913.3812907, 913.4053763, 913.6298613, 913.8492496, 913.9419904, 914.0172336, 914.0409501, 914.1779646, 914.3856804, 914.4453795, 914.4736035, 914.4863062, 914.4884061, 914.7182635, 914.7606363, 914.840786, 915.0343354, 915.0364359, 915.1336029, 915.1491603, 915.2075085, 915.2650363, 915.4177328, 915.4244135, 915.4467513, 915.4637361, 915.5666772, 915.6295841, 915.6801651, 915.7282038, 915.8072941, 915.819277, 915.8321706, 915.8760143, 915.9057198, 915.9140011, 916.0392248, 916.0960924, 916.1411279, 916.2451026, 916.2663372, 916.44166, 916.4588722, 916.483015, 916.5206147, 916.5320739, 916.5651427, 916.5966714, 916.6285479, 916.7816444, 916.7871033, 916.803188, 916.8608829, 916.8696123, 916.9354837, 917.0934486, 917.2353639, 917.2980287, 917.333777, 917.3763915, 917.4381859, 917.6803312, 917.6931153, 917.8069148, 917.9101907, 917.920806, 917.9690829, 918.0074915, 918.0291909, 918.0772258, 918.1126609, 918.1235179, 918.1409722, 918.2921142, 918.348081, 918.3777829, 918.4697076, 918.47815, 918.56524, 918.6912766, 918.7168139, 918.7782721, 918.784186, 918.9240528, 919.0796341, 919.1042082, 919.1866983, 919.2244915, 919.2916685, 919.3580434, 919.3784239, 919.6897056, 919.751845, 919.7596854, 919.9463057, 920.0039489, 920.183925, 920.4634921, 920.5017618, 920.5314856, 920.7016564, 920.7059891, 920.8151251, 920.8740009, 920.9590744, 921.0808313, 921.1613823, 921.2072459, 921.3653039, 921.478868, 921.610092, 921.8353794, 921.8887112, 922.0269266, 922.0423677, 922.0504554, 922.0665681, 922.1021354, 922.4199766, 922.4863861, 922.5253403, 922.5696163, 922.7223899, 922.7641562, 923.1479024, 923.3693065, 923.4638082, 923.5025778, 923.6125241, 923.6233058, 923.6276864, 923.9245688, 924.011222, 924.120821, 924.4178273, 924.4265314, 924.4529928, 924.4672934, 924.4802814, 924.7695343, 924.7760339, 924.9897483, 925.0542336, 925.0760239, 925.1103863, 925.1783308, 925.642098, 925.6443887, 925.7136253, 925.9824697, 926.2136724, 926.516041, 926.6365557, 926.9057942, 927.1278621, 927.412003, 927.6539911, 927.8273543, 927.9099347, 928.4870432, 928.5760207, 928.6335355, 928.8010291, 929.2683396, 929.488798, 930.2120345, 930.2409666, 930.869405, 931.2298845, 931.5999795, 931.9604511, 933.0093013, 933.0294647, 933.1323162, 933.2311583, 933.4425496, 933.5988419, 933.7168507, 934.3222121, 934.8730977, 935.010967, 935.0746306, 935.0991934, 935.6479235, 935.6723948, 935.6794025, 935.822326, 936.2687752, 936.5224461, 937.4734664, 937.769119, 938.4283023, 938.9293084, 939.039414, 939.9934511, 941.4844677, 941.5771756, 941.6735601, 943.7874585, 947.5586641, 948.4608989, 949.5634606, 952.7002856]
        data = numpy.array(array)

        ti = time.time()
        print("Ini")
        phitter_continuous = PHITTER_CONTINUOUS(data, num_bins=None, confidence_level=0.95, minimum_sse=100)
        sorted_distributions_sse, not_rejected_distributions = phitter_continuous.fit()
        print(f"Fin: {time.time() - ti}")

        for distribution, results in not_rejected_distributions.items():
            print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
    `)
}

main();

