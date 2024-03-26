import numpy
import scipy.special
import scipy.stats


class T_STUDENT_3P:
    """
    T distribution
    Parameters T_STUDENT_3P distribution: {"df": *, "loc": *, "scale": *}
    https://phitter.io/distributions/continuous/t_student_3p
    """

    def __init__(self, continuous_measures=None, parameters: dict[str, int | float] = None):
        """
        Initializes the T_STUDENT_3P distribution by either providing a Continuous Measures instance [CONTINUOUS_MEASURES] or a dictionary with the distribution's parameters.
        Parameters T_STUDENT_3P distribution: {"df": *, "loc": *, "scale": *}
        """
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
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.loc) / self.scale
        # result = scipy.stats.t.cdf(z(x), self.df)
        result = scipy.special.betainc(self.df / 2, self.df / 2, (z(x) + numpy.sqrt(z(x) ** 2 + self.df)) / (2 * numpy.sqrt(z(x) ** 2 + self.df)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.loc) / self.scale
        # result = scipy.stats.t.pdf(z(x), self.df)
        result = (1 / (numpy.sqrt(self.df) * scipy.special.beta(0.5, self.df / 2))) * (1 + z(x) * z(x) / self.df) ** (-(self.df + 1) / 2)
        return result

    def ppf(self, u):
        # result = scipy.stats.t.ppf(u, self.df, loc=self.loc, scale=self.scale)
        result = self.loc + self.scale * numpy.sign(u - 0.5) * numpy.sqrt(
            self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u]))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u]))
        )
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
        return self.loc

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.scale * self.scale * self.df) / (self.df - 2)

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
        return 0

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        return 6 / (self.df - 4) + 3

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
        return self.loc

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
        v1 = self.df > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters: {"df": *, "loc": *, "scale": *}
        """

        scipy_params = scipy.stats.t.fit(continuous_measures.data)
        parameters = {"df": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}

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
    path = "../continuous_distributions_sample/sample_t_student_3p.txt"
    data = get_data(path)
    continuous_measures = CONTINUOUS_MEASURES(data)
    distribution = T_STUDENT_3P(continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.get_parameters(continuous_measures)}")
    print(f"CDF: {distribution.cdf(continuous_measures.mean)} {distribution.cdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PDF: {distribution.pdf(continuous_measures.mean)} {distribution.pdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PPF: {distribution.ppf(0.5)} {distribution.ppf(numpy.array([0.3, 0.7]))} - V: {distribution.cdf(distribution.ppf(0.5))}")
