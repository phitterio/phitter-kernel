import math
import scipy.stats


class RAYLEIGH:
    """
    Rayleigh distribution
    https://en.wikipedia.org/wiki/Rayleigh_distribution    
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.loc) / self.scale
        return 1 - math.exp(-0.5 * (z(x) ** 2))

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.loc) / self.scale
        return z(x) * math.exp(-0.5 * (z(x) ** 2)) / self.scale

    def get_num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.scale > 0
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
            {"alpha":  * , "beta":  * , "min":  * , "max":  * }
        """
        ## Scipy Rayleigh estimation
        # scipy_params = scipy.stats.rayleigh.fit(measurements.data)
        # parameters = {"loc": scipy_params[0], "scale": scipy_params[1]}

        ## Location and scale solve system
        scale = math.sqrt(measurements.variance * 2 / (4 - math.pi))
        loc = measurements.mean - scale * math.sqrt(math.pi / 2)

        parameters = {"loc": loc, "scale": scale}
        return parameters


if __name__ == "__main__":
    # Import function to get measurements
    import sys
    sys.path.append("../measurements")
    from measurements import MEASUREMENTS

    # Import function to get measurements
    def get_data(direction):
        sample_distribution_file = open(direction, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        return data

    # Distribution class
    path = "../data/data_rayleigh.txt"
    data = get_data(path)
    measurements = MEASUREMENTS(data)
    distribution = RAYLEIGH(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
