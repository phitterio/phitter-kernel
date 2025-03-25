import numpy
import scipy.stats


class InverseGaussian:
    """
    Inverse Gaussian Distribution
    Also known like Wald distribution
    - Parameters InverseGaussian Distribution: {"mu": *, "lambda": *}
    - https://phitter.io/distributions/continuous/inverse_gaussian
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the InverseGaussian Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters InverseGaussian Distribution: {"mu": *, "lambda": *}
        - https://phitter.io/distributions/continuous/inverse_gaussian
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError(
                "You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True."
            )
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.mu = self.parameters["mu"]
        self.lambda_ = self.parameters["lambda"]

    @property
    def name(self):
        return "inverse_gaussian"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 10, "lambda": 19}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        # result = scipy.stats.norm.cdf(numpy.sqrt(self.lambda_ / x) * ((x / self.mu) - 1)) + numpy.exp(2 * self.lambda_ / self.mu) * scipy.stats.norm.cdf(-numpy.sqrt(self.lambda_ / x) * ((x / self.mu) + 1))
        result = scipy.stats.invgauss.cdf(x, self.mu / self.lambda_, scale=self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        # result = numpy.sqrt(self.lambda_ / (2 * numpy.pi * x**3)) * numpy.exp(-(self.lambda_ * (x - self.mu) ** 2) / (2 * self.mu**2 * x))
        result = scipy.stats.invgauss.pdf(x, self.mu / self.lambda_, scale=self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.invgauss.ppf(u, self.mu / self.lambda_, scale=self.lambda_)
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
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx
        """
        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        return self.mu

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.mu**3 / self.lambda_

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
        return 3 * numpy.sqrt(self.mu / self.lambda_)

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 15 * (self.mu / self.lambda_) + 3

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
        return self.mu * (numpy.sqrt(1 + (9 * self.mu * self.mu) / (4 * self.lambda_ * self.lambda_)) - (3 * self.mu) / (2 * self.lambda_))

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
        v1 = self.mu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by formula.

        Parameters
        ==========
        continuous_measures : dict
            {"mu": * , "variance": * , "skewness": * , "kurtosis": * , "data": * }

        Returns
        =======
        parameters: {"mu": *, "lambda": *}
        """
        mu = continuous_measures.mean
        lambda_ = mu**3 / continuous_measures.variance

        parameters = {"mu": mu, "lambda": lambda_}
        return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    ## Import function to get continuous_measures
    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    ## Distribution class
    path = "../continuous_distributions_sample/sample_inverse_gaussian.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = InverseGaussian(continuous_measures=continuous_measures)

    print(distribution.get_parameters(continuous_measures=continuous_measures))
    print(scipy.stats.invgauss.fit(data))
    print(distribution.cdf(continuous_measures.mean))
    print(distribution.cdf(numpy.array([continuous_measures.mean, continuous_measures.mean])))
    print(distribution.pdf(continuous_measures.mean))
    print(distribution.pdf(numpy.array([continuous_measures.mean, continuous_measures.mean])))
