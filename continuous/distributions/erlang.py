import scipy.integrate
import math
import scipy.special as sc
import scipy.stats


class ERLANG:
    """
    Erlang distribution
    https://en.wikipedia.org/wiki/Erlang_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        result = sc.gammainc(self.k, x / self.beta)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = ((self.beta**-self.k) * (x ** (self.k - 1)) * math.exp(-(x / self.beta))) / math.factorial(self.k - 1)
        result = scipy.stats.erlang.pdf(x, self.k, scale=self.beta)
        return result

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restriction
        """
        v1 = self.k > 0
        v2 = self.beta > 0
        v3 = type(self.k) == int
        return v1 and v2 and v3

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
            {"m":  * , "beta":  * }
        """
        k = round(measurements.mean**2 / measurements.variance)
        β = measurements.variance / measurements.mean
        parameters = {"k": k, "beta": β}
        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements import MEASUREMENTS

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_erlang.txt"
    data = get_data(path)
    measurements = MEASUREMENTS(data)
    distribution = ERLANG(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
