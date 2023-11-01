import math
import scipy.special as sc
import scipy.stats


class GAMMA_3P:
    """
    Gamma distribution
    https://en.wikipedia.org/wiki/Gamma_distribution
    Compendium of Common Probability Distributions (pag.39) ... Michael P. McLaughlin
    """

    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    def cdf(self, x: float) -> float:
        """
        Cumulative distribution function
        Calculated using the definition of the function
        Alternative: quadrature integration method
        """
        ## Method 1: Integrate PDF function
        # result, error = scipy.integrate.quad(self.pdf, 0, x)
        # print(result)

        ## Method 2: Scipy Gamma Distribution class
        # result = scipy.stats.gamma.cdf(x, a=self.alpha, scale=self.beta)
        # print(result)
        result = sc.gammainc(self.alpha, (x - self.loc) / self.beta)
        return result

    def pdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using definition of the function in the documentation
        """
        # result = ((self.beta**-self.alpha) * ((x - self.loc) ** (self.alpha - 1)) * math.exp(-((x - self.loc) / self.beta))) / math.gamma(self.alpha)
        result = scipy.stats.gamma.pdf(x, self.alpha, loc=self.loc, scale=self.beta)
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
            {"alpha":  * , "beta":  * }
        """
        alpha = (2 / measurements.skewness) ** 2
        beta = math.sqrt(measurements.variance / alpha)
        loc = measurements.mean - alpha * beta
        parameters = {"alpha": alpha, "loc": loc, "beta": beta}
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
    path = "../data/data_gamma_3p.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_CONTINUOUS(data)
    distribution = GAMMA_3P(measurements)

    print(distribution.get_parameters(measurements))
    print(distribution.cdf(measurements.mean))
    print(distribution.pdf(measurements.mean))
