import scipy.stats
import math


class POISSON:
    """
    Poisson distribution
    https://en.wikipedia.org/wiki/Poisson_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.lambda_ = self.parameters["lambda"]

    def cdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        Alternative: scipy cdf method
        """
        result = scipy.stats.poisson.cdf(x, self.lambda_)
        return result

    def pmf(self, x: int) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        """
        # result = scipy.stats.poisson.pmf(x, self.lambda_)
        # print(result)
        result = (self.lambda_**x) * math.exp(-self.lambda_) / math.factorial(x)
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
        v1 = self.lambda_ > 0
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
            {"alpha":  * , "beta":  * , "gamma":  * }
        """
        lambda_ = measurements.mean
        parameters = {"lambda": lambda_}
        return parameters


if __name__ == "__main__":
    ## Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_discrete import MEASUREMENTS_DISCRETE

    ## Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        return data

    ## Distribution class
    path = "../data/data_poisson.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = POISSON(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(round(measurements.mean)))
    print(distribution.pmf(round(measurements.mean)))
