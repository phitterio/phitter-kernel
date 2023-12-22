import math
import scipy.stats


class RAYLEIGH:
    """
    Rayleigh distribution
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.gamma = self.parameters["gamma"]
        self.sigma = self.parameters["sigma"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.gamma) / self.sigma
        return 1 - math.exp(-0.5 * (z(x) ** 2))

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.gamma) / self.sigma
        return z(x) * math.exp(-0.5 * (z(x) ** 2)) / self.sigma

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
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        measurements: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, length, num_bins, data

        Returns
        =======
        parameters : dict
            {"gamma": * , "sigma": * }
        """
        ## Scipy Rayleigh estimation
        # scipy_params = scipy.stats.rayleigh.fit(measurements.data)
        # parameters = {"gamma": scipy_params[0], "sigma": scipy_params[1]}

        ## Location and sigma solve system
        sigma = math.sqrt(measurements.variance * 2 / (4 - math.pi))
        gamma = measurements.mean - sigma * math.sqrt(math.pi / 2)

        parameters = {"gamma": gamma, "sigma": sigma}
        return parameters


if __name__ == "__main__":
    # Import function to get measurements
    import sys

    sys.path.append("../measurements")
    from measurements_continuous import MEASUREMENTS_CONTINUOUS

    # Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    # Distribution class
    path = "../data/data_rayleigh.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = RAYLEIGH(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
