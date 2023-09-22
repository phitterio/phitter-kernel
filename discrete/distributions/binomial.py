import scipy.stats
import scipy.special as sc

class BINOMIAL:
    """
    Binomial distribution
    https://en.wikipedia.org/wiki/Binomial_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.n = self.parameters["n"]
        self.p = self.parameters["p"]
                
    def cdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        Alternative: scipy cdf method
        """
        result = scipy.stats.binom.cdf(x, self.n, self.p)
        return result

    
    def pmf(self, x: int) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        """
        # result = scipy.stats.binom.pmf(x, self.n, self.p)
        result = sc.comb(self.n, x) * (self.p ** x) * ((1 - self.p) ** (self.n - x))
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
        v1 = self.p > 0 and self.p < 1
        v2 = self.n > 0
        v3 = type(self.n) == int
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
            {"alpha":  * , "beta":  * , "gamma":  * }
        """
        p = 1 - measurements.variance / measurements.mean
        n = int(round(measurements.mean / p, 0))
        parameters = {"n": n, "p": p}
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
    path = "../data/data_binomial.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = BINOMIAL(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(round(measurements.mean)))
    print(distribution.pmf(round(measurements.mean)))