import scipy.stats
import scipy.special
import numpy
import numpy


class GENERALIZED_NORMAL:
    """
    Generalized normal distribution
    https://en.wikipedia.org/wiki/Generalized_normal_distribution
    This distribution is known whit the following names:
    * Error Distribution
    * Exponential Power Distribution
    * Generalized Error Distribution (GED)
    * Generalized Gaussian distribution (GGD)
    * Subbotin distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)

        self.beta = self.parameters["beta"]
        self.mu = self.parameters["mu"]
        self.alpha = self.parameters["alpha"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        # print(scipy.stats.gennorm.cdf(x , self.beta, loc=self.mu, scale=self.alpha))
        return 0.5 + (numpy.sign(x - self.mu) / 2) * scipy.special.gammainc(1 / self.beta, abs((x - self.mu) / self.alpha) ** self.beta)

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # print(scipy.stats.gennorm.pdf(x , self.beta, loc=self.mu, scale=self.alpha))
        return self.beta / (2 * self.alpha * scipy.special.gamma(1 / self.beta)) * numpy.exp(-((abs(x - self.mu) / self.alpha) ** self.beta))

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restriction
        """
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

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
            {"beta": * , "mu": * , "alpha": * }
        """
        scipy_params = scipy.stats.gennorm.fit(measurements.data)
        parameters = {"beta": scipy_params[0], "mu": scipy_params[1], "alpha": scipy_params[2]}
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
    path = "../data/data_generalized_normal.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = GENERALIZED_NORMAL(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.cdf(numpy.array([measurements.mean, measurements.mean])))
    print(distribution.pdf(measurements.mean))
    print(distribution.pdf(numpy.array([measurements.mean, measurements.mean])))
