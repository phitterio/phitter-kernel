import scipy.special as sc
import scipy.stats
import math


class NORMAL:
    """
    Normal distribution
    https://en.wikipedia.org/wiki/Normal_distribution
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
        # result = scipy.stats.norm.cdf((x - self.mu) / self.sigma)
        z = lambda t: (t - self.mu) / self.sigma
        result = 0.5 * (1 + sc.erf(z(x) / math.sqrt(2)))
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        result = (1 / (self.sigma * math.sqrt(2 * math.pi))) * math.exp(-(((x - self.mu) ** 2) / (2 * self.sigma**2)))
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
        measurements : dict
            {"mu":  * , "variance":  * , "skewness":  * , "kurtosis":  * , "data":  * }

        Returns
        =======
        parameters : dict
            {"mu":  * , "sigma":  * }
        """

        mu = measurements.mean
        sigma = measurements.standard_deviation

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
    path = "../data/data_normal.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = NORMAL(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
