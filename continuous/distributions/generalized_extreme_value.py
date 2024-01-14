import scipy.stats
import numpy


class GENERALIZED_EXTREME_VALUE:
    """
    Generalized Extreme Value Distribution
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)

        self.xi = self.parameters["xi"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return numpy.exp(-numpy.exp(-z(x)))
        else:
            return numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi)))
        # return scipy.stats.genextreme.cdf(x,  - self.xi, loc=self.mu, scale=self.sigma)

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.genextreme.pdf(x,  - self.xi, loc=self.mu, scale=self.sigma))
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))
        else:
            return (1 / self.sigma) * numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi))) * (1 + self.xi * z(x)) ** (-1 - 1 / self.xi)

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.

        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"xi": * , "mu": * , "sigma": * }
        """
        scipy_params = scipy.stats.genextreme.fit(measurements.data)
        parameters = {"xi": -scipy_params[0], "mu": scipy_params[1], "sigma": scipy_params[2]}
        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys
    import numpy

    sys.path.append("../measurements")
    from measurements_continuous import MEASUREMENTS_CONTINUOUS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_generalized_extreme_value.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = GENERALIZED_EXTREME_VALUE(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))
