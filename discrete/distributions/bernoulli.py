class BERNOULLI:
    """
    Bernoulli distribution
    https://en.wikipedia.org/wiki/Bernoulli_distribution
    """
    def __init__(self, measurements):
        self.parameters = self.get_parameters(measurements)
        self.p = self.parameters["p"]
                
    def cdf(self, x: float) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        Alternative: scipy cdf method
        """
        if ( x < 0 ):
            result = 0
        elif( x >= 0 and x < 1 ):
            result = 1 - self.p
        else:
            result = 1
        return result
    
    def pmf(self, x: int) -> float:
        """
        Probability density function
        Calculated using the definition of the function
        """
        result = (self.p ** x) * (1 - self.p) ** (1 - x)
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
        p = measurements.mean
        parameters = {"p": p}
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
    path = "../data/data_bernoulli.txt"
    data = get_data(path)
    measurements = MEASUREMENTS_DISCRETE(data)
    distribution = BERNOULLI(measurements)
    
    print(distribution.get_parameters(measurements))
    print(distribution.cdf(round(measurements.mean)))
    print(distribution.pmf(round(measurements.mean)))