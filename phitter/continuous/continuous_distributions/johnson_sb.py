import numpy
import scipy.integrate
import scipy.stats


class JOHNSON_SB:
    """
    Johnson SB distribution
    Parameters JOHNSON_SB distribution: {"xi": *, "lambda": *, "gamma": *, "delta": *}
    https://phitter.io/distributions/continuous/johnson_sb
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the JOHNSON_SB distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters JOHNSON_SB distribution: {"xi": *, "lambda": *, "gamma": *, "delta": *}
        """
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
        """
        Cumulative distribution function
        """
        # result, error = scipy.integrate.quad(self.pdf, self.xi_, x)
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * numpy.log(z(x) / (1 - z(x))))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * numpy.sqrt(2 * numpy.pi) * z(x) * (1 - z(x)))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.log(z(x) / (1 - z(x)))) ** 2)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = (self.lambda_ * numpy.exp((scipy.stats.norm.ppf(u) - self.gamma_) / self.delta_)) / (1 + numpy.exp((scipy.stats.norm.ppf(u) - self.gamma_) / self.delta_)) + self.xi_
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
        f = lambda x: x**k * (self.delta_ / (numpy.sqrt(2 * numpy.pi) * x * (1 - x))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.log(x / (1 - x))) ** 2)
        return scipy.integrate.quad(f, 0, 1)[0]

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x - µ[1])ᵏ f(x) dx
        """
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
        """
        Parametric mean
        """
        µ1 = self.non_central_moments(1)
        return self.xi_ + self.lambda_ * µ1

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.lambda_ * self.lambda_ * (µ2 - µ1**2)

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
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / (µ2 - µ1**2) ** 1.5

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / (µ2 - µ1**2) ** 2

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
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated with the method proposed in [1].

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"xi": *, "lambda": *, "gamma": *, "delta": *}
            {"xi": * , "lambda": * , "gamma": * , "delta": * }

        References
        ==========
        .. [1] George, F., & Ramachandran, K. M. (2011).
               Estimation of parameters of Johnson's system of distributions.
               Journal of Modern Applied Statistical Methods, 10(2), 9.
        """
        # ## Percentiles
        # z = 0.5384
        # percentiles = [scipy.stats.norm.cdf(0.5384 * i) for i in range(-3, 4, 2)]
        # x1, x2, x3, x4 = [scipy.stats.scoreatpercentile(continuous_measures.data, 100 * x) for x in percentiles]

        # ## Calculation m,n,p
        # m = x4 - x3
        # n = x2 - x1
        # p = x3 - x2

        # ## Calculation distribution parameters
        # lambda_ = (p * numpy.sqrt((((1 + p / m) * (1 + p / n) - 2) ** 2-4))) / (p ** 2 / (m * n) - 1)
        # xi_ = 0.5 * (x3 + x2)-0.5 * lambda_ + p * (p / n - p / m) / (2 * (p ** 2 / (m * n) - 1))
        # delta_ = z / numpy.acosh(0.5 *  numpy.sqrt((1 + p / m) * (1 + p / n)))
        # gamma_ = delta_ * numpy.asinh((p / n - p / m) * numpy.sqrt((1 + p / m) * (1 + p / n)-4) / (2 * (p ** 2 / (m * n) - 1)))

        # parameters = {"xi": xi_, "lambda": lambda_, "gamma": gamma_, "delta": delta_}

        scipy_params = scipy.stats.johnsonsb.fit(continuous_measures.data)
        parameters = {"xi": scipy_params[2], "lambda": scipy_params[3], "gamma": scipy_params[0], "delta": scipy_params[1]}
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
    path = "../continuous_distributions_sample/sample_johnson_sb.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = JOHNSON_SB(continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.get_parameters(continuous_measures)}")
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
