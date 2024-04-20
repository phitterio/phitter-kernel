import numpy
import scipy.optimize
import scipy.stats


class JOHNSON_SU:
    """
    Johnson SU distribution
    Parameters JOHNSON_SU distribution: {"xi": *, "lambda": *, "gamma": *, "delta": *}
    https://phitter.io/distributions/continuous/johnson_su
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None, init_parameters_examples=False):
        """
        Initializes the JOHNSON_SU distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters JOHNSON_SU distribution: {"xi": *, "lambda": *, "gamma": *, "delta": *}
        https://phitter.io/distributions/continuous/johnson_su
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Continuous Measures [CONTINUOUS_MEASURES] instance or a dictionary of the distribution's parameters.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]

    @property
    def name(self):
        return "johnson_su"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xi": 43, "lambda": 382, "gamma": -16, "delta": 54}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * numpy.arcsinh(z(x)))
        # result, error = scipy.integrate.quad(self.pdf, float(" - inf"), x)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * numpy.sqrt(2 * numpy.pi) * numpy.sqrt(z(x) ** 2 + 1))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.arcsinh(z(x))) ** 2)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.lambda_ * numpy.sinh((scipy.stats.norm.ppf(u) - self.gamma_) / self.delta_) + self.xi_
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        """
        Sample of n elements of ditribution
        """
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        """
        Parametric no central moments. µ[k] = E[Xᵏ] = ∫xᵏ∙f(x) dx
        """
        return None

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x - µ[1])ᵏ f(x) dx
        """
        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        return self.xi_ - self.lambda_ * numpy.exp(self.delta_**-2 / 2) * numpy.sinh(self.gamma_ / self.delta_)

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.lambda_**2 / 2) * (numpy.exp(self.delta_**-2) - 1) * (numpy.exp(self.delta_**-2) * numpy.cosh((2 * self.gamma_) / self.delta_) + 1)

    @property
    def standard_deviation(self) -> float:
        """
        Parametric standard deviation
        """
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """
        Parametric skewness
        """
        return -(
            self.lambda_**3
            * numpy.sqrt(numpy.exp(self.delta_**-2))
            * (numpy.exp(self.delta_**-2) - 1) ** 2
            * (numpy.exp(self.delta_**-2) * (numpy.exp(self.delta_**-2) + 2) * numpy.sinh(3 * (self.gamma_ / self.delta_)) + 3 * numpy.sinh(self.gamma_ / self.delta_))
        ) / (4 * self.standard_deviation**3)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
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
        """
        Parametric median
        """
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        """
        Parametric mode
        """
        return None

    @property
    def num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            ## Variables declaration
            xi_, lambda_, gamma_, delta_ = initial_solution

            ## Help
            w = numpy.exp(1 / delta_**2)
            omega = gamma_ / delta_
            A = w**2 * (w**4 + 2 * w**3 + 3 * w**2 - 3) * numpy.cosh(4 * omega)
            B = 4 * w**2 * (w + 2) * numpy.cosh(2 * omega)
            C = 3 * (2 * w + 1)

            ## Parametric expected expressions
            parametric_mean = xi_ - lambda_ * numpy.sqrt(w) * numpy.sinh(omega)
            parametric_variance = (lambda_**2 / 2) * (w - 1) * (w * numpy.cosh(2 * omega) + 1)
            # parametric_skewness = -((lambda_ ** 3) * numpy.sqrt(w) * (w - 1) ** 2 * (w * (w + 2) * numpy.sinh(3 * omega) + 3 * numpy.sinh(omega)) ) / (4 * numpy.sqrt(parametric_variance) ** 3)
            parametric_kurtosis = ((lambda_**4) * (w - 1) ** 2 * (A + B + C)) / (8 * numpy.sqrt(parametric_variance) ** 4)
            parametric_median = xi_ + lambda_ * numpy.sinh(-omega)

            ## System Equations
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_kurtosis - continuous_measures.kurtosis
            eq4 = parametric_median - continuous_measures.median

            return (eq1, eq2, eq3, eq4)

        solution = scipy.optimize.fsolve(equations, (1, 1, 1, 1), continuous_measures)
        parameters = {"xi": solution[0], "lambda": solution[1], "gamma": solution[2], "delta": solution[3]}
        return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import CONTINUOUS_MEASURES

    ## Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../continuous_distributions_sample/sample_johnson_su.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = JOHNSON_SU(continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.parameters}")
    print(f"CDF: {distribution.cdf(continuous_measures.mean)} {distribution.cdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PDF: {distribution.pdf(continuous_measures.mean)} {distribution.pdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PPF: {distribution.ppf(0.5)} {distribution.ppf(numpy.array([0.5, 0.5]))} - V: {distribution.cdf(distribution.ppf(0.5))}")
    print(f"SAMPLE: {distribution.sample(5)}")
    print(f"\nSTATS")
    print(f"mean: {distribution.mean} - {continuous_measures.mean}")
    print(f"variance: {distribution.variance} - {continuous_measures.variance}")
    print(f"skewness: {distribution.skewness} - {continuous_measures.skewness}")
    print(f"kurtosis: {distribution.kurtosis} - {continuous_measures.kurtosis}")
    print(f"median: {distribution.median} - {continuous_measures.median}")
    print(f"mode: {distribution.mode} - {continuous_measures.mode}")
