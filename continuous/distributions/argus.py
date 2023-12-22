import math
import scipy.stats
import scipy.special as sc


class ARGUS:
    """
    Argus distribution
    https://en.wikipedia.org/wiki/ARGUS_distribution
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.chi = self.parameters["chi"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        z = lambda t: (t - self.loc) / self.scale
        # Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t)-0.5
        # print(scipy.stats.argus.cdf(x, self.chi, loc=self.loc, scale=self.scale))
        # print(1 - Ψ(self.chi * math.sqrt(1 - z(x) * z(x))) / Ψ(self.chi))
        result = 1 - sc.gammainc(1.5, self.chi * self.chi * (1 - z(x) ** 2) / 2) / sc.gammainc(1.5, self.chi * self.chi / 2)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        z = lambda t: (t - self.loc) / self.scale
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        # print(scipy.stats.argus.pdf(x, self.chi, loc=self.loc, scale=self.scale))
        result = (1 / self.scale) * ((self.chi**3) / (math.sqrt(2 * math.pi) * Ψ(self.chi))) * z(x) * math.sqrt(1 - z(x) * z(x)) * math.exp(-0.5 * self.chi**2 * (1 - z(x) * z(x)))
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
        v1 = self.chi > 0
        v2 = self.scale > 0
        return v1 and v2

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
            {"alpha": * , "beta": * , "min": * , "max": * }
        """
        scipy_params = scipy.stats.argus.fit(measurements.data)
        parameters = {"chi": scipy_params[0], "loc": scipy_params[1], "scale": scipy_params[2]}
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
    path = "../data/data_argus.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = ARGUS(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
