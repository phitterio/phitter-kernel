import math
import scipy.stats


class LOGNORMAL:
    """
    Lognormal distribution
    https://en.wikipedia.org/wiki/Log - normal_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.miu = self.parameters["miu"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # result, error = scipy.integrate.quad(self.pdf, 1e-15, x)
        result = scipy.stats.norm.cdf((math.log(x) - self.miu) / self.sigma)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        return (1 / (x * self.sigma * math.sqrt(2 * math.pi))) * math.exp(-(((math.log(x) - self.miu) ** 2) / (2 * self.sigma**2)))

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.miu > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, measurements) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample measurements.
        The parameters are calculated by formula.

        Parameters
        ==========
        measurements : dict
            {"miu":  * , "variance":  * , "skewness":  * , "kurtosis":  * , "data":  * }

        Returns
        =======
        parameters : dict
            {"miu":  * , "sigma":  * }
        """

        μ = math.log(measurements.mean**2 / math.sqrt(measurements.mean**2 + measurements.variance))
        σ = math.sqrt(math.log((measurements.mean**2 + measurements.variance) / (measurements.mean**2)))

        parameters = {"miu": μ, "sigma": σ}
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
    path = "../data/data_lognormal.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = LOGNORMAL(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
