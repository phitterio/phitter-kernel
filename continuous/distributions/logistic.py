import scipy.special as sc
import scipy.stats
import math


class LOGISTIC:
    """
    Logistic distribution
    https://en.wikipedia.org/wiki/Logistic_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: math.exp(-(t - self.mu) / self.sigma)
        result = 1 / (1 + z(x))
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: math.exp(-(t - self.mu) / self.sigma)
        result = z(x) / (self.sigma * (1 + z(x)) ** 2)
        return result

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
            {"mu": * , "sigma": * }
        """
        mu = measurements.mean
        sigma = math.sqrt(3 * measurements.variance / (math.pi**2))

        ## Results
        parameters = {"mu": mu, "sigma": sigma}

        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_continuous import MEASUREMENTS_CONTINUOUS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_logistic.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = LOGISTIC(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
